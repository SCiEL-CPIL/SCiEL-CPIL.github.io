import numpy as np
import time
import threading

_TABLE_CACHE = {}
_TABLE_LOCK = threading.Lock()

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def _check_deadline():
    return time.time() >= _DEADLINE

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    max_q = Q
    min_q = -Q
    n_states = 2 * Q + 1
    
    # Precompute interpolation grid for the optimality condition
    # phi(x) = g(x)/h(x) - x
    # g(x) = 1 - exp(-A exp(-kx) dt)
    # h(x) = k A dt exp(-kx) exp(-A exp(-kx) dt)
    # We solve phi(x) = C for x
    
    x_max = 10.0 / k
    n_grid = 5000
    x_grid = np.linspace(0, x_max, n_grid)
    
    # Precompute p(x) = 1 - exp(-A exp(-kx) dt) on the grid
    exp_kx = np.exp(-k * x_grid)
    exp_Au = np.exp(-A * exp_kx * dt)
    p_grid = 1.0 - exp_Au
    
    # Calculate phi(x) = g(x)/h(x) - x where g = p, h = k*A*dt*exp(-kx)*exp(-A*exp(-kx)*dt)
    # h = k * A * dt * exp_kx * (1 - p_grid)
    g = p_grid
    h = k * A * dt * exp_kx * (1.0 - p_grid)
    
    # Avoid division by zero at large x where h is tiny
    phi = np.empty(n_grid)
    mask = h > 1e-300
    phi[mask] = g[mask] / h[mask] - x_grid[mask]
    # For very large x, phi(x) ≈ 1/k - x
    phi[~mask] = 1.0/k - x_grid[~mask]
    
    neg_phi = -phi
    
    # Precompute p lookup: given a depth value, find its grid index and interpolate p
    # We'll use np.searchsorted + linear interp for p lookup
    
    V_next = np.zeros(n_states)
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    q_vals = np.arange(n_states) - Q
    risk = (gamma / 2.0) * (sigma**2) * (q_vals**2) * dt
    
    def _p_lookup(depths):
        """Lookup p(x) for array of depth values via grid interpolation."""
        # Clip to grid range
        d = np.clip(depths, 0.0, x_max)
        # Find positions
        idx = np.searchsorted(x_grid, d, side='right') - 1
        idx = np.clip(idx, 0, n_grid - 2)
        t = (d - x_grid[idx]) / (x_grid[idx+1] - x_grid[idx])
        t = np.clip(t, 0.0, 1.0)
        return p_grid[idx] * (1.0 - t) + p_grid[idx+1] * t
    
    for n in range(N - 1, -1, -1):
        if _check_deadline():
            depth_default = 1.0 / k
            for n_fill in range(n, N):
                opt_bid[n_fill, :] = depth_default
                opt_ask[n_fill, :] = depth_default
            return opt_bid, opt_ask
        
        # Delta values (constant across iterations for given V_next)
        delta_plus = np.zeros(n_states)
        delta_minus = np.zeros(n_states)
        delta_plus[:-1] = V_next[1:] - V_next[:-1]
        delta_minus[1:] = V_next[:-1] - V_next[1:]
        
        # Initialize b and a for fixed point
        b = np.zeros(n_states)
        a = np.zeros(n_states)
        p_b = np.zeros(n_states)
        p_a = np.zeros(n_states)
        
        for _iter in range(60):
            if _check_deadline():
                 break
            
            # C_b = (1 - p_a) * Delta_plus - p_a * Delta_minus
            C_b = (1.0 - p_a) * delta_plus - p_a * delta_minus
            b_new = np.interp(-C_b, neg_phi, x_grid)
            p_b_new = _p_lookup(b_new)
            
            # C_a = (1 - p_b_new) * Delta_minus - p_b_new * Delta_plus
            C_a = (1.0 - p_b_new) * delta_minus - p_b_new * delta_plus
            a_new = np.interp(-C_a, neg_phi, x_grid)
            p_a_new = _p_lookup(a_new)
            
            # Convergence check
            if np.allclose(b_new, b, atol=1e-10) and np.allclose(a_new, a, atol=1e-10):
                b = b_new
                a = a_new
                p_b = p_b_new
                p_a = p_a_new
                break
            
            b = b_new
            a = a_new
            p_b = p_b_new
            p_a = p_a_new
        
        # Store optimal depths
        opt_bid[n, :] = b
        opt_ask[n, :] = a
        
        # Compute V_curr
        prob_q_plus = p_b * (1.0 - p_a)
        prob_q_minus = (1.0 - p_b) * p_a
        
        prob_q_plus[-1] = 0.0
        prob_q_minus[0] = 0.0
        
        reward = p_b * b + p_a * a
        reward[-1] = p_a[-1] * a[-1]
        reward[0] = p_b[0] * b[0]
        
        V_next_plus = np.zeros(n_states)
        V_next_minus = np.zeros(n_states)
        V_next_plus[:-1] = V_next[1:]
        V_next_minus[1:] = V_next[:-1]
        
        exp_val = prob_q_plus * V_next_plus + prob_q_minus * V_next_minus
        prob_q = 1.0 - prob_q_plus - prob_q_minus
        exp_val += prob_q * V_next
        
        V_next = reward - risk + exp_val

    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    
    with _TABLE_LOCK:
        if key not in _TABLE_CACHE:
            start_time = time.time()
            opt_bid, opt_ask = _precompute_regime(
                params['sigma'], params['A'], params['k'], params['gamma'], 
                params['T'], params['steps'], params['Q']
            )
            _TABLE_CACHE[key] = (opt_bid, opt_ask)
            
            if time.time() - start_time > 290:
                 depth = 1.0 / params["k"]
                 return np.full(len(inventory), depth), np.full(len(inventory), depth)

    opt_bid, opt_ask = _TABLE_CACHE[key]
    
    q_indices = np.asarray(inventory) + params['Q']
    
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    min_allowed = -3.0 / params['k']
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths < min_allowed), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths < min_allowed), np.inf, ask_depths)
    
    return bid_depths, ask_depths
