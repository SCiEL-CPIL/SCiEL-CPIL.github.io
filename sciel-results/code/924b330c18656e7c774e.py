import numpy as np
import time
from scipy.optimize import minimize_scalar, brentq

_DEADLINE = time.time() + 285
_POLICY_CACHE = {}
_FALLBACK_START = time.time()

def _fallback_quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    dt = T / steps
    
    remaining = max(0.0, T - t)
    time_frac = remaining / T
    
    q = inventory.astype(np.float64)
    base = 1.0 / k
    
    # Heuristic: spread proportional to risk aversion, scaled by inventory
    # Risk aversion term: gamma * sigma^2
    # Capacity term: A * k
    # Base spread should be around 1/k
    # Inventory adjustment: shift quotes to discourage building inventory
    # If q > 0 (long), raise bid (worse bid), lower ask (better ask)
    # bid_depth increases (further from mid), ask_depth decreases (closer to mid)
    # But wait: ask_depth is distance above mid. Lower ask_depth = closer to mid = better ask.
    # If long, we want to sell (ask) more easily, so lower ask_depth.
    # If long, we want to buy less, so raise bid_depth.
    
    risk = gamma * sigma**2
    cap = A * k
    scale = risk / cap
    beta = base * scale * Q * 0.5  # Scaled down to be reasonable
    
    q_norm = np.clip(q / Q, -1, 1)
    adjust = np.tanh(2.0 * q_norm)
    
    bid_depth = base + beta * adjust
    ask_depth = base - beta * adjust
    
    # Ensure positive
    min_depth = 0.01
    bid_depth = np.maximum(bid_depth, min_depth)
    ask_depth = np.maximum(ask_depth, min_depth)
    
    # Enforce inventory limits
    bid_depth[inventory >= Q] = np.inf
    ask_depth[inventory <= -Q] = np.inf
    
    return bid_depth, ask_depth

def _solve_dp(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    alpha = A * dt  # A * exp(0) * dt, max intensity factor
    
    # Precompute optimal depth for a range of dv
    # dv = V_next[q'] - V_next[q]
    # The optimal d maximizes: p(d) * (d + dv) + (1-p(d)) * 0
    # where p(d) = 1 - exp(-alpha * exp(-k*d))
    # Let u = exp(-k*d), so d = -ln(u)/k, u in (0, 1]
    # p(u) = 1 - exp(-alpha * u)
    # f(u) = (1 - exp(-alpha*u)) * (-ln(u)/k + dv)
    # df/du = [alpha*exp(-alpha*u)] * (-ln(u)/k + dv) + (1 - exp(-alpha*u)) * (-1/(k*u))
    # Set to 0:
    # alpha*exp(-alpha*u)*(-ln(u)/k + dv) - (1 - exp(-alpha*u))/(k*u) = 0
    # alpha*u*exp(-alpha*u)*(-ln(u)/k + dv) = 1 - exp(-alpha*u)
    
    def find_opt_d(dv):
        def obj(u):
            if u <= 0: u = 1e-10
            if u > 1: u = 1.0
            p = 1.0 - np.exp(-alpha * u)
            d = -np.log(u) / k
            return p * (d + dv)
        
        # Minimize negative of objective
        res = minimize_scalar(lambda u: -obj(u), bounds=(1e-8, 1.0), method='bounded', 
                              options={'xatol': 1e-8})
        u_opt = res.x
        d_opt = -np.log(u_opt) / k
        # Clamp to reasonable range
        d_opt = np.clip(d_opt, 0.0, 10.0 / k)
        return d_opt
    
    # We need to solve this for many dv values.
    # dv ranges roughly from -C to C where C ~ Q * gamma * sigma^2 * T * something.
    # Let's estimate the range of V.
    # V ~ O(Q^2 * gamma * sigma^2 * T).
    # dv ~ O(Q * gamma * sigma^2 * T).
    
    # Let's create a grid of dv values and solve for each.
    # To speed up, we can solve for a small number of points and interpolate,
    # or solve for all states at each step? No, dv depends on step.
    
    # Better approach: Solve for d as a function of dv on a coarse grid, then interpolate.
    # Range of dv:
    # Max |dv| ~ Q * (gamma * sigma**2 * T) / (A * dt * k)? No.
    # Let's just use a wide range: [-50, 50] or similar.
    # With Q=30, gamma=0.05, sigma=2, T=1.
    # gamma*sigma^2*T = 0.2. Q*0.2 = 6.
    # dv might be around a few units.
    
    dv_min = -50.0
    dv_max = 50.0
    num_dv = 200
    dv_grid = np.linspace(dv_min, dv_max, num_dv)
    d_opt_grid = np.zeros(num_dv)
    
    # Check deadline before heavy computation
    if time.time() > _DEADLINE - 10:
        return None
    
    for i, dv in enumerate(dv_grid):
        if time.time() > _DEADLINE - 5:
            return None
        d_opt_grid[i] = find_opt_d(dv)
        
    # Ensure monotonicity (optional, but good for sanity)
    # d_opt should be decreasing in dv?
    # If dv is large positive (reward for filling is high), we want to fill more easily (smaller d).
    # If dv is large negative (penalty for filling), we want to fill less easily (larger d).
    # So d_opt should decrease as dv increases.
    # Let's check if our grid is monotonic.
    
    def get_opt_depth(dv_val):
        if dv_val < dv_min:
            return d_opt_grid[0]
        if dv_val > dv_max:
            return d_opt_grid[-1]
        return np.interp(dv_val, dv_grid, d_opt_grid)
    
    # DP Loop
    num_states = 2 * Q + 1
    V_next = np.zeros(num_states)
    
    # We don't need to store all tables if we can compute on the fly?
    # But we need to return quote(t, ...) which is called at each step.
    # We can precompute bid/ask depths for each step and inventory state.
    # bid_table[step, q_index] = depth
    # ask_table[step, q_index] = depth
    
    bid_table = np.full((steps, num_states), np.inf)
    ask_table = np.full((steps, num_states), np.inf)
    
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    penalty = 0.5 * gamma * (sigma ** 2) * (q_vals ** 2) * dt
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 2:
            return None
            
        V_curr = np.zeros(num_states)
        
        # Bid: q -> q+1
        # Valid for q < Q
        idx_bid = np.arange(num_states - 1)
        dv_bid = V_next[idx_bid + 1] - V_next[idx_bid]
        d_bid = get_opt_depth(dv_bid)
        
        p_bid = 1.0 - np.exp(-A * np.exp(-k * d_bid) * dt)
        val_bid = p_bid * (d_bid + V_next[idx_bid + 1]) + (1.0 - p_bid) * V_next[idx_bid]
        
        bid_table[n, :num_states-1] = d_bid
        # bid_table[n, num_states-1] remains inf
        V_curr[:num_states-1] = val_bid
        V_curr[num_states-1] = V_next[num_states-1]
        
        # Ask: q -> q-1
        # Valid for q > -Q
        idx_ask = np.arange(1, num_states)
        dv_ask = V_next[idx_ask - 1] - V_next[idx_ask]
        d_ask = get_opt_depth(dv_ask)
        
        p_ask = 1.0 - np.exp(-A * np.exp(-k * d_ask) * dt)
        val_ask = p_ask * (d_ask + V_next[idx_ask - 1]) + (1.0 - p_ask) * V_next[idx_ask]
        
        ask_table[n, 1:num_states] = d_ask
        # ask_table[n, 0] remains inf
        V_curr[1:num_states] = val_ask
        V_curr[0] = V_next[0]
        
        V_curr -= penalty
        V_next = V_curr
        
    return bid_table, ask_table

def quote(t, inventory, mid, params):
    # Check deadline
    if time.time() > _DEADLINE:
        return _fallback_quote(t, inventory, mid, params)
        
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _POLICY_CACHE:
        # Try to solve DP
        if time.time() < _DEADLINE - 15:
            result = _solve_dp(params)
            if result is not None:
                _POLICY_CACHE[key] = result
            else:
                _POLICY_CACHE[key] = None
        else:
            _POLICY_CACHE[key] = None
            
    cached = _POLICY_CACHE[key]
    
    if cached is None:
        return _fallback_quote(t, inventory, mid, params)
    
    bid_table, ask_table = cached
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    Q = params['Q']
    indices = inventory + Q
    
    bid_depths = bid_table[n, indices].copy()
    ask_depths = ask_table[n, indices].copy()
    
    # Ensure valid values
    bid_depths[np.isnan(bid_depths)] = np.inf
    ask_depths[np.isnan(ask_depths)] = np.inf
    
    return bid_depths, ask_depths
