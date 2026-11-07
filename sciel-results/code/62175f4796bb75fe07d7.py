# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state
_policy_cache = {}
_global_start_time = None
_DEADLINE_MARGIN = 15.0  # seconds to spare before the 330s limit

def _get_deadline():
    """Return the global deadline (330s wall-clock limit minus margin)."""
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    return _global_start_time + 330.0 - _DEADLINE_MARGIN

def _solve_depth(lam, k, coeff, deadline):
    """
    Solve max_d P(d) * (coeff + d) where P(d) = 1 - exp(-lam * exp(-k*d)).
    d is in [0, 3/k].
    Uses vectorized grid search + Newton refinement.
    """
    if coeff < -1e6:
        return 3.0 / k  # very negative -> avoid fills -> max depth
    
    max_depth = 3.0 / k
    # Coarse grid
    n_grid = 500
    grid = np.linspace(0, max_depth, n_grid)
    P = 1.0 - np.exp(-lam * np.exp(-k * grid))
    vals = P * (coeff + grid)
    
    idx = np.argmax(vals)
    best_d = grid[idx]
    best_val = vals[idx]
    
    # Refine with Newton's method if interior
    if 0 < idx < n_grid - 1:
        # Newton iteration
        d = best_d
        for _ in range(20):
            if time.time() > deadline:
                break
            e = np.exp(-k * d)
            P = 1.0 - np.exp(-lam * e)
            # f(d) = P(d) * (coeff + d)
            # f'(d) = P'(d)(coeff + d) + P(d)
            # P'(d) = -exp(-lam*e) * (-lam * e * (-k)) = -k * lam * e * exp(-lam*e)
            # Actually: d/da exp(-a) = -exp(-a), a = lam*exp(-kd)
            # da/dd = -k*lam*exp(-kd) = -k*a
            # dP/dd = -exp(-a) * (-k*a) = k*a*exp(-a)
            # So P'(d) = k * lam * exp(-k*d) * exp(-lam*exp(-k*d))
            a = lam * e
            P_prime = k * a * np.exp(-a)
            
            f = P * (coeff + d)
            f_prime = P_prime * (coeff + d) + P
            
            if abs(f_prime) < 1e-12:
                break
            d_new = d - f / f_prime
            d_new = np.clip(d_new, 0, max_depth)
            if abs(d_new - d) < 1e-10:
                d = d_new
                break
            d = d_new
        
        # Check if Newton improved
        e = np.exp(-k * d)
        P_new = 1.0 - np.exp(-lam * e)
        val_new = P_new * (coeff + d)
        if val_new > best_val:
            best_d = d
            best_val = val_new
    
    return best_d

def _compute_optimal_policy(params, deadline):
    """
    Compute optimal bid/ask depths via DP.
    Returns (bid_table, ask_table) each of shape (steps, 2*Q+1).
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    lam = A * dt
    num_q = 2 * Q + 1
    max_depth = 3.0 / k

    # V[t, q_idx] where q_idx = q + Q
    V = np.zeros((steps + 1, num_q), dtype=np.float64)
    bid_table = np.full((steps, num_q), max_depth, dtype=np.float64)
    ask_table = np.full((steps, num_q), max_depth, dtype=np.float64)

    cost = (gamma / 2.0) * sigma**2 * dt

    # Precompute grid for vectorized optimization
    n_grid = 200
    grid = np.linspace(0, max_depth, n_grid)
    P_grid = 1.0 - np.exp(-lam * np.exp(-k * grid))

    for t in range(steps - 1, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use simple heuristic for remaining steps
            for tt in range(t, -1, -1):
                for qi in range(num_q):
                    bid_table[tt, qi] = 1.0 / k
                    ask_table[tt, qi] = 1.0 / k
                    V[tt, qi] = V[tt+1, qi] - cost
            break

        V_next = V[t + 1]

        # Initialize depths
        b_vals = np.full(num_q, 1.0 / k, dtype=np.float64)
        a_vals = np.full(num_q, 1.0 / k, dtype=np.float64)

        # Boundary conditions
        b_vals[2 * Q] = max_depth  # q = Q, no bid
        a_vals[0] = max_depth      # q = -Q, no ask

        # Iterate to fixed point (coupled optimization)
        for iteration in range(8):
            if time.time() > deadline - 5:
                break

            # Compute P_a
            P_a = np.zeros(num_q)
            valid_a = a_vals < max_depth - 1e-10
            if np.any(valid_a):
                P_a[valid_a] = 1.0 - np.exp(-lam * np.exp(-k * a_vals[valid_a]))

            # Compute C_b for bid optimization
            # For q_idx < 2Q (q < Q):
            # C_b = (1-P_a)(V_next[q+1] - V_next[q]) + P_a*(a + V_next[q] - V_next[q-1])
            # Wait, let me re-derive.
            # X = (1-P_a)*V_next[q_idx+1] + P_a*(a + V_next[q_idx])
            # Y = P_a*(a + V_next[q_idx-1]) + (1-P_a)*V_next[q_idx]
            # C_b = X - Y = (1-P_a)*(V_next[q_idx+1] - V_next[q_idx]) + P_a*(V_next[q_idx] - V_next[q_idx-1])
            
            V_q = V_next
            V_q_plus = np.roll(V_next, -1)
            V_q_minus = np.roll(V_next, 1)
            V_q_plus[-1] = 0
            V_q_minus[0] = 0

            C_b = (1.0 - P_a) * (V_q_plus - V_q) + P_a * (V_q - V_q_minus)

            # Optimize b for q_idx < 2Q
            for qi in range(2 * Q):
                coeff = C_b[qi]
                b_vals[qi] = _solve_depth(lam, k, coeff, deadline)
            
            # Compute P_b
            P_b = np.zeros(num_q)
            valid_b = b_vals < max_depth - 1e-10
            if np.any(valid_b):
                P_b[valid_b] = 1.0 - np.exp(-lam * np.exp(-k * b_vals[valid_b]))

            # Compute C_a for ask optimization
            # For q_idx > 0 (q > -Q):
            # U = P_b*(b + V_next[q_idx]) + (1-P_b)*V_next[q_idx-1]
            # V_term = P_b*(b + V_next[q_idx+1]) + (1-P_b)*V_next[q_idx]
            # C_a = U - V_term = P_b*(V_next[q_idx] - V_next[q_idx+1]) + (1-P_b)*(V_next[q_idx-1] - V_next[q_idx])
            
            C_a = P_b * (V_q - V_q_plus) + (1.0 - P_b) * (V_q_minus - V_q)

            # Optimize a for q_idx > 0
            for qi in range(1, num_q):
                coeff = C_a[qi]
                a_vals[qi] = _solve_depth(lam, k, coeff, deadline)

            # Re-enforce boundaries
            b_vals[2 * Q] = max_depth
            a_vals[0] = max_depth

        # Store results
        bid_table[t, :] = b_vals
        ask_table[t, :] = a_vals

        # Compute V[t, :]
        P_b = np.zeros(num_q)
        valid_b = b_vals < max_depth - 1e-10
        if np.any(valid_b):
            P_b[valid_b] = 1.0 - np.exp(-lam * np.exp(-k * b_vals[valid_b]))

        P_a = np.zeros(num_q)
        valid_a = a_vals < max_depth - 1e-10
        if np.any(valid_a):
            P_a[valid_a] = 1.0 - np.exp(-lam * np.exp(-k * a_vals[valid_a]))

        b_safe = np.where(b_vals < max_depth - 1e-10, b_vals, 0.0)
        a_safe = np.where(a_vals < max_depth - 1e-10, a_vals, 0.0)

        # EV = P_b*P_a*(b+a+V_q) + P_b*(1-P_a)*(b+V_q_plus) + (1-P_b)*P_a*(a+V_q_minus) + (1-P_b)*(1-P_a)*V_q
        term1 = P_b * P_a * (b_safe + a_safe + V_q)
        term2 = P_b * (1.0 - P_a) * (b_safe + V_q_plus)
        term3 = (1.0 - P_b) * P_a * (a_safe + V_q_minus)
        term4 = (1.0 - P_b) * (1.0 - P_a) * V_q

        V[t, :] = term1 + term2 + term3 + term4 - cost

    return bid_table, ask_table

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    deadline = _get_deadline()
    
    if key not in _policy_cache:
        if time.time() < deadline:
            try:
                bid_table, ask_table = _compute_optimal_policy(params, deadline)
                _policy_cache[key] = (bid_table, ask_table)
            except Exception:
                # Fallback
                depth = np.full(len(inventory), 1.0 / params['k'], dtype=np.float64)
                return depth, depth.copy()
        else:
            depth = np.full(len(inventory), 1.0 / params['k'], dtype=np.float64)
            return depth, depth.copy()
    
    bid_table, ask_table = _policy_cache[key]
    
    dt = params['T'] / params['steps']
    step = int(round(t / dt))
    step = np.clip(step, 0, params['steps'] - 1)
    
    Q = params['Q']
    q_indices = inventory + Q
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    bid_depths = bid_table[step, q_indices].astype(np.float64)
    ask_depths = ask_table[step, q_indices].astype(np.float64)
    
    # Clamp to valid range [0, 3/k]
    max_depth = 3.0 / params['k']
    bid_depths = np.clip(bid_depths, 0.0, max_depth)
    ask_depths = np.clip(ask_depths, 0.0, max_depth)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
