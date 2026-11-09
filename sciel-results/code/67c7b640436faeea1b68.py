import numpy as np
from scipy.optimize import brentq
import time

# Cache for precomputed tables
_TABLE_CACHE = {}
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE

def _solve_depths_closed_form(C, A, k, dt):
    """
    Vectorized closed-form solve for optimal depths.
    For small A*dt, p(d) ≈ A*exp(-k*d)*dt
    J(d) = p(d)*d + C*p(d) ≈ A*dt*exp(-k*d)*(d+C)
    J'(d) = A*dt*exp(-k*d)*(1 - k*(d+C)) = 0
    => d* = 1/k - C, clamped to [0, d_max]
    
    For larger A*dt, falls back to brentq per element.
    """
    C = np.asarray(C, dtype=np.float64)
    d_max = 10.0 / k  # upper bound for depth
    
    # Closed-form: d* = 1/k - C, clamped to [0, d_max]
    d_cf = np.clip(1.0 / k - C, 0.0, d_max)
    
    # Check if closed-form is valid: A*dt should be small enough
    # The approximation p(d) ≈ A*exp(-k*d)*dt is good when A*exp(-k*d)*dt << 1
    # At d=0: A*dt. If A*dt < 0.1, the approximation is within ~5%
    # Let's use a threshold: if A*dt < 0.5, use closed-form; else brentq
    if A * dt < 0.5:
        return d_cf
    
    # Fallback: brentq per element (only needed for large A*dt)
    d_out = np.zeros_like(C)
    p_0 = 1.0 - np.exp(-A * dt)
    p_0_prime = -k * A * dt * (1 - p_0)
    
    for i in range(len(C)):
        c_val = C[i]
        val_0 = p_0_prime * c_val + p_0
        if val_0 <= 0:
            d_out[i] = 0.0
        else:
            def f(x):
                u = np.exp(-k * x)
                p = 1.0 - np.exp(-A * u * dt)
                dp = -k * A * dt * u * (1 - p)
                return dp * (x + c_val) + p
            try:
                d_out[i] = brentq(f, 0, d_max, xtol=1e-8)
            except ValueError:
                if f(d_max) > 0:
                    d_out[i] = d_max
                else:
                    d_out[i] = 0.0
    return d_out

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for all steps and inventories.
    Returns:
        opt_bid: np.array of shape (steps, 2*Q+1)
        opt_ask: np.array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    N = steps
    max_q = Q
    min_q = -Q
    n_states = 2 * Q + 1
    
    V_next = np.zeros(n_states)
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    # Precompute constants
    p_0 = 1.0 - np.exp(-A * dt)
    p_0_prime = -k * A * dt * (1 - p_0)
    
    # Precompute inventory values
    q_vals = np.arange(n_states) - Q  # array of q values
    
    for n in range(N - 1, -1, -1):
        if _check_deadline():
            # Fill remaining with closed-form approximation
            for nn in range(n, -1, -1):
                C_b = np.zeros(n_states)
                C_a = np.zeros(n_states)
                for qi in range(n_states):
                    q = qi - Q
                    if q < max_q:
                        C_b[qi] = (V_next[qi+1] - V_next[qi])
                    if q > min_q:
                        C_a[qi] = (V_next[qi-1] - V_next[qi])
                opt_bid[nn] = _solve_depths_closed_form(C_b, A, k, dt)
                opt_ask[nn] = _solve_depths_closed_form(C_a, A, k, dt)
            break
        
        V_curr = np.zeros(n_states)
        
        # Fixed-point iteration for coupled b, a
        b = np.zeros(n_states)
        a = np.zeros(n_states)
        
        for _iter in range(30):
            # Compute p_a from current a
            p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
            # Compute p_b from current b
            p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
            
            # Compute C_b and C_a
            C_b = np.zeros(n_states)
            C_a = np.zeros(n_states)
            
            for qi in range(n_states):
                q = qi - Q
                if q < max_q:
                    C_b[qi] = (V_next[qi+1] - V_next[qi]) * (1.0 - p_a[qi])
                if q > min_q:
                    C_a[qi] = (V_next[qi-1] - V_next[qi]) * (1.0 - p_b[qi])
            
            # Solve for new b and a using closed-form or brentq
            b_new = _solve_depths_closed_form(C_b, A, k, dt)
            a_new = _solve_depths_closed_form(C_a, A, k, dt)
            
            # Enforce boundaries
            b_new[max_q - min_q] = 0.0  # q == max_q -> no bid
            a_new[0] = 0.0  # q == min_q -> no ask
            
            # Check convergence
            if np.max(np.abs(b_new - b)) < 1e-7 and np.max(np.abs(a_new - a)) < 1e-7:
                b = b_new
                a = a_new
                break
            b = b_new
            a = a_new
        
        opt_bid[n] = b
        opt_ask[n] = a
        
        # Calculate V_curr
        p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
        p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
        
        # At boundaries, force no fill
        p_b[max_q - min_q] = 0.0
        p_a[0] = 0.0
        
        # Transitions
        prob_q_plus = np.zeros(n_states)
        prob_q_minus = np.zeros(n_states)
        
        # q -> q+1 via bid fill (only if q < max_q)
        mask_b = np.zeros(n_states, dtype=bool)
        mask_b[:-1] = True  # q < max_q
        prob_q_plus[mask_b] = p_b[mask_b] * (1.0 - p_a[mask_b])
        
        # q -> q-1 via ask fill (only if q > min_q)
        mask_a = np.zeros(n_states, dtype=bool)
        mask_a[1:] = True  # q > min_q
        prob_q_minus[mask_a] = (1.0 - p_b[mask_a]) * p_a[mask_a]
        
        prob_q = 1.0 - prob_q_plus - prob_q_minus
        
        # Reward: depth earned
        reward = p_b * b + p_a * a
        
        # Expected value from next state
        exp_val = np.zeros(n_states)
        for qi in range(n_states):
            q = qi - Q
            if q < max_q:
                exp_val[qi] += prob_q_plus[qi] * V_next[qi+1]
            else:
                exp_val[qi] += prob_q_plus[qi] * V_next[qi]  # stays
            if q > min_q:
                exp_val[qi] += prob_q_minus[qi] * V_next[qi-1]
            else:
                exp_val[qi] += prob_q_minus[qi] * V_next[qi]  # stays
            exp_val[qi] += prob_q[qi] * V_next[qi]
        
        # Risk term
        risk = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        V_curr = reward - risk + exp_val
        V_next = V_curr
    
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    # Determine step index
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    # Check cache
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    if key not in _TABLE_CACHE:
        # Precompute
        opt_bid, opt_ask = _precompute_regime(
            params['sigma'], params['A'], params['k'], params['gamma'], 
            params['T'], params['steps'], params['Q']
        )
        _TABLE_CACHE[key] = (opt_bid, opt_ask)
        
        # Time guard (safety)
        if _check_deadline():
            depth = 1.0 / params["k"]
            return np.full(len(inventory), depth), np.full(len(inventory), depth)

    opt_bid, opt_ask = _TABLE_CACHE[key]
    
    # Map inventory to indices
    q_indices = np.array(inventory) + params['Q']
    
    # Look up depths
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    return bid_depths, ask_depths
