import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320  # safety margin before 330s kill
_cache = {}


def _vectorized_optimize_batch(A_dt, k, deltas, D_max, n_grid=2000, n_refine=8):
    """
    Vectorized 1D optimization of g(d) = p(d) * (d + delta) over d in [0, D_max]
    for a batch of delta values.
    p(d) = 1 - exp(-A_dt * exp(-k*d)).
    
    Returns: (d_opts, vals) arrays of shape (len(deltas),).
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)
    
    # Grid search
    d_grid = np.linspace(0.0, D_max, n_grid)
    p_grid = 1.0 - np.exp(-A_dt * np.exp(-k * d_grid))  # shape (n_grid,)
    
    # g_grid[i, j] = p_grid[j] * (d_grid[j] + deltas[i])
    # Shape: (n, n_grid)
    g_grid = p_grid[None, :] * (d_grid[None, :] + deltas[:, None])
    
    # For each row, find argmax
    idx_best = np.argmax(g_grid, axis=1)  # shape (n,)
    
    # Bisection refinement
    d_lo = np.zeros(n, dtype=np.float64)
    d_hi = np.zeros(n, dtype=np.float64)
    
    # Left boundary case
    left_mask = (idx_best == 0)
    d_lo[left_mask] = 0.0
    d_hi[left_mask] = d_grid[1] if n_grid > 1 else D_max
    
    # Right boundary case
    right_mask = (idx_best == n_grid - 1)
    d_lo[right_mask] = d_grid[-2] if n_grid > 1 else 0.0
    d_hi[right_mask] = D_max
    
    # Interior case
    int_mask = ~(left_mask | right_mask)
    d_lo[int_mask] = d_grid[idx_best[int_mask] - 1]
    d_hi[int_mask] = d_grid[idx_best[int_mask] + 1]
    
    # Bisection: find where g' changes sign
    # g'(d) = p'(d)*(d+delta) + p(d)
    # p'(d) = -k * A_dt * exp(-k*d) * exp(-A_dt * exp(-k*d))
    for _ in range(n_refine):
        d_mid = 0.5 * (d_lo + d_hi)
        exp_kd = np.exp(-k * d_mid)  # shape (n,)
        u = A_dt * exp_kd
        p = 1.0 - np.exp(-u)
        dp_dd = -k * u * np.exp(-u)
        gp = dp_dd * (d_mid + deltas) + p
        
        # Where gp > 0, move lo up; else move hi down
        d_lo = np.where(gp > 0, d_mid, d_lo)
        d_hi = np.where(gp > 0, d_hi, d_mid)
    
    d_opt = 0.5 * (d_lo + d_hi)
    
    # Evaluate at refined point
    p_opt = 1.0 - np.exp(-A_dt * np.exp(-k * d_opt))
    val_opt = p_opt * (d_opt + deltas)
    
    # Evaluate at d=0
    p_zero = 1.0 - np.exp(-A_dt)
    val_zero = p_zero * deltas
    
    # Evaluate at d=D_max
    p_dmax = 1.0 - np.exp(-A_dt * np.exp(-k * D_max))
    val_dmax = p_dmax * (D_max + deltas)
    
    # Take the best: max of (0, val_opt, val_zero, val_dmax)
    best_val = np.maximum.reduce([np.zeros(n), val_opt, val_zero, val_dmax])
    
    # Determine which point achieved the best
    d_result = np.where(val_opt >= val_zero, d_opt, 0.0)
    d_result = np.where(val_dmax >= np.maximum(val_opt, val_zero), D_max, d_result)
    val_result = np.where(best_val <= 0, 0.0, best_val)
    
    return d_result, val_result


def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth tables via dynamic programming.
    Uses vectorized grid-search + bisection for high-precision 1D optimization.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # V[n, q_idx] where q_idx = q + Q
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    opt_bid = np.full((steps, n_inv), np.inf, dtype=np.float64)
    opt_ask = np.full((steps, n_inv), np.inf, dtype=np.float64)

    q_vals = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma * sigma * q_vals * q_vals * dt

    # Search bounds for depth
    D_max = 10.0 / k
    
    # Pre-compute constants for speed
    A_dt = A * dt
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback: simple heuristic
            for nn in range(n, -1, -1):
                for qi in range(n_inv):
                    q = qi - Q
                    base = 1.0 / k
                    skew = gamma * sigma * sigma * (T - nn * dt) * q * 0.05
                    b = max(1e-6, base * (1.0 + skew))
                    a = max(1e-6, base * (1.0 - skew))
                    if q >= Q:
                        b = np.inf
                    if q <= -Q:
                        a = np.inf
                    opt_bid[nn, qi] = b
                    opt_ask[nn, qi] = a
                    V[nn, qi] = 0.0
            break

        Vn1 = V[n + 1]
        
        # Vectorized computation for all inventory levels
        # Bid side: delta_b[qi] = Vn1[qi+1] - Vn1[qi] for qi in 0..Q-1
        # Ask side: delta_a[qi] = Vn1[qi-1] - Vn1[qi] for qi in Q+1..2Q
        
        # Bid deltas: shape (Q,)
        delta_b = Vn1[1:Q+1] - Vn1[0:Q]  # qi=0 to Q-1
        # Ask deltas: shape (Q,)
        delta_a = Vn1[1:Q+1] - Vn1[0:Q]  # qi=Q+1 to 2Q, delta = Vn1[qi-1] - Vn1[qi]
        # Wait, let me re-derive:
        # For ask at qi (where qi > Q, i.e., q > -Q): delta_a = Vn1[qi-1] - Vn1[qi]
        # qi ranges from Q+1 to 2Q
        # Let j = qi - (Q+1), so j ranges from 0 to Q-1
        # delta_a[j] = Vn1[Q+1+j-1] - Vn1[Q+1+j] = Vn1[Q+j] - Vn1[Q+1+j]
        delta_a = Vn1[Q:2*Q] - Vn1[Q+1:2*Q+1]
        
        # Vectorized optimization for bid side
        bid_depths = np.zeros(Q, dtype=np.float64)
        bid_rewards = np.zeros(Q, dtype=np.float64)
        
        for i in range(Q):
            d_opt, val = _vectorized_optimize(A_dt, k, delta_b[i], D_max)
            bid_depths[i] = d_opt
            bid_rewards[i] = val
        
        # Vectorized optimization for ask side
        ask_depths = np.zeros(Q, dtype=np.float64)
        ask_rewards = np.zeros(Q, dtype=np.float64)
        
        for i in range(Q):
            d_opt, val = _vectorized_optimize(A_dt, k, delta_a[i], D_max)
            ask_depths[i] = d_opt
            ask_rewards[i] = val
        
        # Fill in the tables
        # Bid: qi = 0 to Q-1 (q = -Q to -1)
        opt_bid[n, 0:Q] = bid_depths
        opt_bid[n, Q] = np.inf  # q = 0, can still bid... wait
        # Actually: no bid fill at q = Q, meaning qi = 2Q
        # No ask fill at q = -Q, meaning qi = 0
        
        # Let me re-check the boundaries:
        # "no bid fill at q = Q" → qi = Q + Q = 2Q → opt_bid[n, 2Q] = inf
        # "no ask fill at q = -Q" → qi = -Q + Q = 0 → opt_ask[n, 0] = inf
        
        # Bid side: valid for qi = 0 to 2Q-1 (q = -Q to Q-1)
        # delta_b for qi: Vn1[qi+1] - Vn1[qi]
        # qi=0: Vn1[1] - Vn1[0]
        # qi=2Q-1: Vn1[2Q] - Vn1[2Q-1]
        # So delta_b has 2Q entries, not Q.
        
        # Let me redo this properly.
        # Bid: qi from 0 to 2Q-1
        delta_b_full = Vn1[1:] - Vn1[:2*Q]  # shape (2Q,)
        
        bid_depths_full = np.full(2*Q, np.inf, dtype=np.float64)
        bid_rewards_full = np.zeros(2*Q, dtype=np.float64)
        
        for i in range(2*Q):
            d_opt, val = _vectorized_optimize(A_dt, k, delta_b_full[i], D_max)
            bid_depths_full[i] = d_opt
            bid_rewards_full[i] = val
        
        # Ask: qi from 1 to 2Q
        delta_a_full = Vn1[:2*Q] - Vn1[1:]  # shape (2Q,)
        
        ask_depths_full = np.full(2*Q, np.inf, dtype=np.float64)
        ask_rewards_full = np.zeros(2*Q, dtype=np.float64)
        
        for i in range(2*Q):
            d_opt, val = _vectorized_optimize(A_dt, k, delta_a_full[i], D_max)
            ask_depths_full[i] = d_opt
            ask_rewards_full[i] = val
        
        # Assemble opt_bid[n, :]
        # opt_bid[n, qi] for qi=0..2Q-1 is bid_depths_full[qi]
        # opt_bid[n, 2Q] = inf (no bid at q=Q)
        opt_bid[n, 0:2*Q] = bid_depths_full
        opt_bid[n, 2*Q] = np.inf
        
        # Assemble opt_ask[n, :]
        # opt_ask[n, qi] for qi=1..2Q is ask_depths_full[qi-1]
        # opt_ask[n, 0] = inf (no ask at q=-Q)
        opt_ask[n, 0] = np.inf
        opt_ask[n, 1:2*Q+1] = ask_depths_full
        
        # Compute rewards for each qi
        # bid_reward[qi] = bid_rewards_full[qi] for qi=0..2Q-1, else 0
        # ask_reward[qi] = ask_rewards_full[qi-1] for qi=1..2Q, else 0
        total_reward = np.zeros(n_inv, dtype=np.float64)
        total_reward[0:2*Q] += bid_rewards_full
        total_reward[1:2*Q+1] += ask_rewards_full
        
        V[n, :] = Vn1 - risk + total_reward

    return opt_bid, opt_ask

def _get_dp_tables(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key not in _cache:
        sigma, A, k, gamma, T, steps, Q = key
        opt_bid, opt_ask = _compute_dp(sigma, A, k, gamma, T, steps, Q)
        _cache[key] = (opt_bid, opt_ask)
    return _cache[key]

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    opt_bid, opt_ask = _get_dp_tables(params)

    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, q_idx].astype(np.float64)
    ask_depths = opt_ask[n, q_idx].astype(np.float64)

    # Safety: ensure valid values
    min_valid = -3.0 / k + 1e-6
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, np.inf)

    # Boundary: no bid at Q, no ask at -Q
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)

    return bid_depths, ask_depths
