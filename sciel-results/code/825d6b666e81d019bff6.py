"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0

# Cache for pre-computed optimal depths
# Key: (sigma, A, k, gamma, Q, steps)
# Value: dict with 'bid' and 'ask' arrays of shape (steps, 2*Q+1)
_OPT_TABLES = {}

def _solve_1d_max(A, k, dt, delta):
    """
    Solve max_{x >= 0} p(x) * (x + delta) where p(x) = 1 - exp(-A exp(-kx) dt).
    Uses Newton-Raphson with fallback to grid search.
    Returns the optimal depth (float).
    """
    # If delta is very large negative, check if no quote is better
    # p(0) * (0 + delta) = (1 - exp(-A*dt)) * delta
    # If this is negative and the function doesn't recover, return inf
    
    # First-order condition:
    # f(x) = p(x) * (x + delta)
    # f'(x) = p'(x)(x + delta) + p(x)
    # p(x) = 1 - exp(-c(x)) where c(x) = A*exp(-kx)*dt
    # p'(x) = exp(-c(x)) * c'(x) = exp(-c(x)) * (-k) * A*exp(-kx)*dt = -k * c(x) * exp(-c(x))
    # Let E = exp(-c(x))
    # p(x) = 1 - E
    # p'(x) = -k * c(x) * E
    # f'(x) = -k * c(x) * E * (x + delta) + (1 - E)
    # f'(x) = 1 - E - k * c(x) * E * (x + delta)
    # f'(x) = 1 - E * (1 + k * c(x) * (x + delta))
    
    # Set f'(x) = 0 => E * (1 + k * c(x) * (x + delta)) = 1
    # Note: E = exp(-c(x)) <= 1, so LHS <= 1 + k*c(x)*(x+delta)
    # If x+delta is very negative, the term in parenthesis can be negative, making LHS negative, so no solution?
    # Wait, if x+delta < 0, f(x) might be decreasing initially.
    
    # Let's just use a robust hybrid approach.
    # Coarse grid to find region, then Newton.
    
    max_depth = 15.0 / k
    n_coarse = 100
    x_coarse = np.linspace(0, max_depth, n_coarse)
    c_coarse = A * np.exp(-k * x_coarse) * dt
    E_coarse = np.exp(-c_coarse)
    p_coarse = 1.0 - E_coarse
    val_coarse = p_coarse * (x_coarse + delta)
    
    # Check if max is negative
    max_val_coarse = np.max(val_coarse)
    if max_val_coarse <= 1e-12:
        return np.inf
    
    idx_coarse = np.argmax(val_coarse)
    
    # Newton refinement
    x = x_coarse[idx_coarse]
    if idx_coarse == 0:
        x = 0.0
    elif idx_coarse == n_coarse - 1:
        x = max_depth
    
    for _ in range(20):
        c = A * np.exp(-k * x) * dt
        E = np.exp(-c)
        p = 1.0 - E
        f_val = p * (x + delta)
        
        # Derivative
        # f'(x) = 1 - E * (1 + k * c * (x + delta))
        f_prime = 1.0 - E * (1.0 + k * c * (x + delta))
        
        # Second derivative for stability check (optional)
        # c' = -k*c
        # E' = -c' * E = k*c*E
        # f'' = - (E' * (1 + k*c*(x+delta)) + E * (k*c'*(x+delta) + k*c))
        # f'' = - (k*c*E * (1 + k*c*(x+delta)) + E * (-k^2*c*(x+delta) + k*c))
        # f'' = -k*c*E * (1 + k*c*(x+delta) - k*c*(x+delta) + c)  <-- wait, k*c' = -k^2*c
        # Let's just do secant or damped Newton if f_prime is small.
        
        if abs(f_prime) < 1e-12:
            break
            
        x_new = x - f_val / f_prime
        
        # Damping
        if x_new < 0:
            x_new = x * 0.5
        if x_new > max_depth:
            x_new = (x + max_depth) / 2
            
        if abs(x_new - x) < 1e-9:
            x = x_new
            break
        x = x_new
        
    # Final check
    c = A * np.exp(-k * x) * dt
    E = np.exp(-c)
    p = 1.0 - E
    best_val = p * (x + delta)
    
    if best_val <= 1e-12:
        return np.inf
        
    return x

def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    # Initialize value function
    V_next = np.zeros(2*Q + 1, dtype=np.float64)
    
    # Store optimal depths
    opt_bid = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    opt_ask = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 5.0:
            # Fallback: fill remaining with constant depth
            base_depth = 1.0 / k
            for nn in range(n, -1, -1):
                opt_bid[nn, :] = base_depth
                opt_ask[nn, :] = base_depth
                # Handle boundaries
                opt_bid[nn, -1] = np.inf
                opt_ask[nn, 0] = np.inf
            break
            
        V_curr = np.zeros(2*Q + 1, dtype=np.float64)
        
        for i, q in enumerate(range(-Q, Q + 1)):
            # Inventory cost
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Deltas
            if q < Q:
                delta_b = V_next[q + 1 + Q] - V_next[q + Q]
            else:
                delta_b = 0.0 # Will be inf
                
            if q > -Q:
                delta_a = V_next[q - 1 + Q] - V_next[q + Q]
            else:
                delta_a = 0.0 # Will be inf
                
            # Solve for b
            if q < Q:
                best_b = _solve_1d_max(A, k, dt, delta_b)
            else:
                best_b = np.inf
                
            # Solve for a
            if q > -Q:
                best_a = _solve_1d_max(A, k, dt, delta_a)
            else:
                best_a = np.inf
                
            # Compute contributions
            contrib_b = 0.0
            if best_b != np.inf:
                p_b = 1.0 - np.exp(-A * np.exp(-k * best_b) * dt)
                contrib_b = p_b * (best_b + delta_b)
                
            contrib_a = 0.0
            if best_a != np.inf:
                p_a = 1.0 - np.exp(-A * np.exp(-k * best_a) * dt)
                contrib_a = p_a * (best_a + delta_a)
            
            V_curr[i] = -cost + V_next[q + Q] + contrib_b + contrib_a
            
            opt_bid[n, i] = best_b
            opt_ask[n, i] = best_a
            
        V_next = V_curr
    
    table = {'bid': opt_bid, 'ask': opt_ask}
    _OPT_TABLES[key] = table
    return table

def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed HJB solution.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to simple safe heuristic if running out of time
        sigma = params["sigma"]
        k = params["k"]
        Q = params["Q"]
        q = np.asarray(inventory, dtype=np.float64)
        base_depth = 1.0 / k
        bid_depth = np.full_like(q, base_depth)
        ask_depth = np.full_like(q, base_depth)
        bid_depth[q >= Q] = np.inf
        ask_depth[q <= -Q] = np.inf
        return bid_depth, ask_depth

    # Get or compute table
    table = _get_optimal_table(params)
    
    # Map t to step index
    steps = params["steps"]
    T = params["T"]
    dt = T / steps
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
    
    # Map inventory to index
    Q = params["Q"]
    q = np.asarray(inventory, dtype=np.int64)
    # Clamp to [-Q, Q] just in case, though contract says it's in range
    q_clamped = np.clip(q, -Q, Q)
    indices = q_clamped + Q
    
    bid_depths = table['bid'][n, indices]
    ask_depths = table['ask'][n, indices]
    
    # Ensure no invalid values
    bid_depths = np.where(bid_depths < 0, np.inf, bid_depths)
    ask_depths = np.where(ask_depths < 0, np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
