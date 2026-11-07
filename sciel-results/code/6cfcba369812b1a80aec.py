import numpy as np
from scipy.optimize import minimize_scalar

_cache = {}

def _get_tables(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Inventory indices: 0..2Q corresponding to -Q..Q
    n_states = 2 * Q + 1
    
    # V_next and V_curr
    V_next = np.zeros(n_states)
    
    # Tables for depths
    d_b_table = np.zeros((steps, n_states))
    d_a_table = np.zeros((steps, n_states))
    V_table = np.zeros((steps, n_states))
    
    d_max = 10.0 / k  # Upper bound for depth
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states)
        for idx in range(n_states):
            q = idx - Q
            
            # Calculate cost
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Optimize d_b
            # Term: p_b * (d_b + V_next[idx+1] - V_next[idx])
            # If q == Q, bid is forbidden (p_b = 0 effectively, or d_b = inf)
            if q == Q:
                opt_d_b = np.inf
                val_b = 0.0
            else:
                target_b = V_next[idx + 1] - V_next[idx]
                def obj_b(d):
                    if d <= 0: return 0.0
                    p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                    return p * (d + target_b)
                res_b = minimize_scalar(lambda d: -obj_b(d), bounds=(0, d_max), method='bounded')
                opt_d_b = res_b.x
                val_b = -res_b.fun
            
            # Optimize d_a
            # Term: p_a * (d_a + V_next[idx-1] - V_next[idx])
            if q == -Q:
                opt_d_a = np.inf
                val_a = 0.0
            else:
                target_a = V_next[idx - 1] - V_next[idx]
                def obj_a(d):
                    if d <= 0: return 0.0
                    p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                    return p * (d + target_a)
                res_a = minimize_scalar(lambda d: -obj_a(d), bounds=(0, d_max), method='bounded')
                opt_d_a = res_a.x
                val_a = -res_a.fun
            
            # Calculate total value
            # Approximation: ignore p_b p_a V_next[idx] coupling term
            # Exact: p_b(d_b) d_b + p_a(d_a) d_a + p_b V_{q+1} + p_a V_{q-1} + (1-p_b)(1-p_a) V_q
            # Using optimized d_b, d_a:
            p_b = 1.0 - np.exp(-A * np.exp(-k * opt_d_b) * dt) if opt_d_b != np.inf else 0.0
            p_a = 1.0 - np.exp(-A * np.exp(-k * opt_d_a) * dt) if opt_d_a != np.inf else 0.0
            
            V_curr[idx] = -cost + p_b * (opt_d_b + V_next[idx + 1] if q != Q else 0) + \
                           p_a * (opt_d_a + V_next[idx - 1] if q != -Q else 0) + \
                           (1 - p_b) * (1 - p_a) * V_next[idx]
            
            # Note: The above formula is slightly inconsistent with the separable optimization.
            # The separable optimization maximized p_b(d_b+delta_b) + p_a(d_a+delta_a).
            # The true objective is p_b d_b + p_a d_a + p_b V_{q+1} + p_a V_{q-1} + (1-p_b)(1-p_a) V_q.
            # = p_b(d_b + V_{q+1}) + p_a(d_a + V_{q-1}) + (1-p_b)(1-p_a) V_q
            # = p_b(d_b + V_{q+1} - V_q) + p_a(d_a + V_{q-1} - V_q) + V_q + p_b p_a V_q
            # We optimized the first two terms. The last term V_q is constant. p_b p_a V_q is small.
            # So V_curr[idx] should be close to V_next[idx] + max_val_b + max_val_a - cost.
            # Let's just compute the exact value using the chosen d_b, d_a to be safe for V_next.
            
            if q != Q:
                term_b = p_b * (opt_d_b + V_next[idx + 1])
            else:
                term_b = 0.0
            if q != -Q:
                term_a = p_a * (opt_d_a + V_next[idx - 1])
            else:
                term_a = 0.0
            term_no_fill = (1 - p_b) * (1 - p_a) * V_next[idx]
            
            V_curr[idx] = -cost + term_b + term_a + term_no_fill
            
            d_b_table[n, idx] = opt_d_b
            d_a_table[n, idx] = opt_d_a
            V_table[n, idx] = V_curr[idx]
            
        V_next = V_curr
        
    _cache[key] = (d_b_table, d_a_table)
    return _cache[key]

def quote(t, inventory, mid, params):
    d_b_table, d_a_table = _get_tables(params)
    steps = params['steps']
    Q = params['Q']
    
    # Determine step index
    # t = n * dt. dt = T/steps.
    # n = t / dt = t * steps / T
    n = int(t * steps / params['T'])
    if n >= steps: n = steps - 1
    if n < 0: n = 0
    
    # Map inventory to index
    # inventory is array of ints in [-Q, Q]
    idx = inventory + Q
    
    bid_depths = d_b_table[n, idx]
    ask_depths = d_a_table[n, idx]
    
    return bid_depths, ask_depths
