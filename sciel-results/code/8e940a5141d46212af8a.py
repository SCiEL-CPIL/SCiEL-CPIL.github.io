"""
Optimal market making with inventory risk.
Uses a high-resolution discrete DP with vectorized Bellman updates.
The depth action space is discretized on a fine log-spaced grid, and the
Bellman backup is computed via NumPy broadcasting over all inventory states
simultaneously within each time step.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 320.0  # safety margin before 330s kill


def _hmm_depths(params, t, inventory):
    """
    Hopenhaver-Sussmann closed-form approximation.
    bid_depth = 1/k + (gamma * sigma^2 * q * tau) / (A * k)
    ask_depth = 1/k - (gamma * sigma^2 * q * tau) / (A * k)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']

    q = inventory.astype(np.float64)
    tau = np.maximum(T - t, 0.0)

    base = 1.0 / k
    skew = (gamma * sigma * sigma * q * tau) / (A * k) if A > 0 and k > 0 else np.zeros_like(q)

    bid_depth = base + skew
    ask_depth = base - skew

    # Clip to valid range
    min_depth = 1e-6
    max_depth = 15.0 / k
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)

    # Boundary: no bid fill at q=Q, no ask fill at q=-Q
    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)

    # Ensure validity
    min_valid = -3.0 / k
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= min_valid), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= min_valid), ask_depth, np.inf)

    return bid_depth, ask_depth


def _compute_vectorized_dp(params, n_depths=200):
    """
    Vectorized DP: for each time step, compute the Bellman backup for all
    inventory states simultaneously using broadcasting.
    Optimized for speed with a smaller depth grid.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_q = 2 * Q + 1
    q_range = np.arange(-Q, Q + 1, dtype=np.float64)

    depth_min = 0.001
    depth_max = 12.0 / k
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)

    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    depth_p = depth_grid * p_fill

    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    inv_cost_factor = (gamma / 2.0) * sigma * sigma * dt
    cost_arr = inv_cost_factor * q_range * q_range  # shape (n_q,)

    # Precompute pf_outer once
    pf_outer = p_fill[:, None] * p_fill[None, :]  # (n_depths, n_depths)
    term1 = depth_p[:, None] + depth_p[None, :]  # (n_depths, n_depths)

    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            # Fallback: fill remaining with HMM
            for n2 in range(n, -1, -1):
                tau2 = T - n2 * dt
                base = 1.0 / k
                skew = (gamma * sigma * sigma * q_range * tau2) / (A * k)
                bid_table[n2, :] = np.clip(base + skew, depth_min, depth_max)
                ask_table[n2, :] = np.clip(base - skew, depth_min, depth_max)
                bid_table[n2, n_q - 1] = np.inf
                ask_table[n2, 0] = np.inf
            break

        Vn1 = V[n + 1]  # shape (n_q,)

        # V_plus[i] = V(q+1) when at q=i, but capped at Q
        # V_minus[i] = V(q-1) when at q=i, but capped at -Q
        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]

        D1 = V_plus - Vn1       # shape (n_q,)
        Dm1 = V_minus - Vn1     # shape (n_q,)
        D2 = V_plus + V_minus - 2.0 * Vn1  # shape (n_q,)

        # Build J efficiently using precomputed terms
        # term2: Vn1[qi] + p_fill[i]*D1[qi] + p_fill[j]*Dm1[qi]
        # term3: -p_fill[i]*p_fill[j]*D2[qi]
        
        # J[i,j,q] = term1[i,j] + Vn1[q] + p_fill[i]*D1[q] + p_fill[j]*Dm1[q] - pf_outer[i,j]*D2[q] - cost[q]
        
        # Reshape for efficient computation
        # We can compute this as:
        # J = term1[:, :, None] + (Vn1 - cost_arr)[None, None, :] 
        #     + (p_fill[:, None, None] * D1[None, None, :])
        #     + (p_fill[None, :, None] * Dm1[None, None, :])
        #     - (pf_outer[:, :, None] * D2[None, None, :])
        
        J = (term1[:, :, None] 
             + (Vn1 - cost_arr)[None, None, :]
             + p_fill[:, None, None] * D1[None, None, :]
             + p_fill[None, :, None] * Dm1[None, None, :]
             - pf_outer[:, :, None] * D2[None, None, :])

        # Find argmax over (i, j) for each qi
        flat_J = J.reshape(n_depths * n_depths, n_q)
        idx_max = np.argmax(flat_J, axis=0)  # shape (n_q,)

        i_max = idx_max // n_depths
        j_max = idx_max % n_depths

        V[n, :] = flat_J[idx_max, np.arange(n_q)]
        bid_table[n, :] = depth_grid[i_max]
        ask_table[n, :] = depth_grid[j_max]

    # Boundary conditions
    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf

    return bid_table, ask_table


def _compute_dp_policy(params):
    """
    Compute optimal bid/ask depths via backward DP on a depth grid.
    Uses vectorized operations where possible.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_q = 2 * Q + 1
    q_range = np.arange(-Q, Q + 1, dtype=np.float64)

    depth_min = 0.001
    depth_max = 12.0 / k
    n_depths = 500
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)

    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    depth_p = depth_grid * p_fill

    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    inv_cost_factor = (gamma / 2.0) * sigma * sigma * dt
    cost_arr = inv_cost_factor * q_range * q_range

    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            for n2 in range(n, -1, -1):
                tau2 = T - n2 * dt
                base = 1.0 / k
                skew = (gamma * sigma * sigma * q_range * tau2) / (A * k)
                bid_table[n2, :] = np.clip(base + skew, depth_min, depth_max)
                ask_table[n2, :] = np.clip(base - skew, depth_min, depth_max)
                bid_table[n2, n_q - 1] = np.inf
                ask_table[n2, 0] = np.inf
            break

        Vn1 = V[n + 1]

        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]

        D1 = V_plus - Vn1
        Dm1 = V_minus - Vn1
        D2 = V_plus + V_minus - 2.0 * Vn1

        # Vectorized over all qi
        # For interior states (1..n_q-2), both bid and ask can fill
        # For qi=0: no ask fill; for qi=n_q-1: no bid fill

        # Build J for interior states using broadcasting
        # term1: depth_p[i] + depth_p[j] -> (n_depths, n_depths)
        term1 = depth_p[:, None] + depth_p[None, :]

        # For interior qi (1 to n_q-2):
        # term2: Vn1[qi] + p_fill[i]*D1[qi] + p_fill[j]*Dm1[qi]
        # term3: -p_fill[i]*p_fill[j]*D2[qi]

        # Process in chunks to manage memory if needed
        # For now, process all interior states at once
        interior = slice(1, n_q - 1)
        n_int = n_q - 2

        if n_int > 0:
            D1_int = D1[interior]  # (n_int,)
            Dm1_int = Dm1[interior]
            D2_int = D2[interior]
            Vn1_int = Vn1[interior]
            cost_int = cost_arr[interior]

            # J shape: (n_depths, n_depths, n_int)
            J = (term1[:, :, None]
                 + Vn1_int[None, None, :]
                 + p_fill[:, None, None] * D1_int[None, None, :]
                 + p_fill[None, :, None] * Dm1_int[None, None, :]
                 - (p_fill[:, None] * p_fill[None, :])[:, :, None] * D2_int[None, None, :]
                 - cost_int[None, None, :])

            flat_J = J.reshape(n_depths * n_depths, n_int)
            idx_max = np.argmax(flat_J, axis=0)

            i_max = idx_max // n_depths
            j_max = idx_max % n_depths

            V[n, interior] = flat_J[idx_max, np.arange(n_int)]
            bid_table[n, interior] = depth_grid[i_max]
            ask_table[n, interior] = depth_grid[j_max]

        # Boundary: qi = 0 (no ask fill)
        if n_q > 0:
            # Only bid can fill: maximize over i only
            # J[i, 0] = depth_p[i] + Vn1[0] + p_fill[i]*D1[0] - cost[0]
            J_bid = depth_p + Vn1[0] + p_fill * D1[0] - cost_arr[0]
            i_max_bid = np.argmax(J_bid)
            V[n, 0] = J_bid[i_max_bid]
            bid_table[n, 0] = depth_grid[i_max_bid]
            ask_table[n, 0] = np.inf

        # Boundary: qi = n_q-1 (no bid fill)
        if n_q > 1:
            # Only ask can fill: maximize over j only
            J_ask = depth_p + Vn1[n_q-1] + p_fill * Dm1[n_q-1] - cost_arr[n_q-1]
            j_max_ask = np.argmax(J_ask)
            V[n, n_q-1] = J_ask[j_max_ask]
            bid_table[n, n_q-1] = np.inf
            ask_table[n, n_q-1] = depth_grid[j_max_ask]

    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf

    return bid_table, ask_table


_policy_cache = {}


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT:
        return _hmm_depths(params, t, inventory)

    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key not in _policy_cache:
        try:
            # Use vectorized DP with optimized grid size
            bid_table, ask_table = _compute_vectorized_dp(params, n_depths=200)
        except Exception:
            try:
                bid_table, ask_table = _compute_dp_policy(params)
            except Exception:
                bid_table, ask_table = None, None
                # Ultimate fallback: HMM
                if bid_table is None:
                    return _hmm_depths(params, t, inventory)
        _policy_cache[key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[key]

    if bid_table is None:
        return _hmm_depths(params, t, inventory)

    steps = params['steps']
    Q = params['Q']
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    q_index = inventory + Q
    q_index = np.clip(q_index, 0, 2 * Q)

    bid_depths = bid_table[n, q_index].copy()
    ask_depths = ask_table[n, q_index].copy()

    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths), ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
