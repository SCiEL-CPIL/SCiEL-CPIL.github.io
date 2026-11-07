"""
Optimal market making with inventory risk.
Fully vectorized Bellman DP: the update across all inventory states is
performed simultaneously via NumPy broadcasting, eliminating the per-inventory
Python loop. This permits a much finer depth grid (reducing discretization
bias) within the same wall-clock budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 310.0  # safety margin before 330s kill


def _hmm_depths(params, t, inventory):
    """Hopenhaver-Sussmann closed-form fallback."""
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

    min_depth = 1e-6
    max_depth = 15.0 / k
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)

    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)

    min_valid = -3.0 / k
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= min_valid), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= min_valid), ask_depth, np.inf)

    return bid_depth, ask_depth


def _compute_vectorized_dp(params, n_depths=800):
    """
    Fully vectorized backward DP.
    
    Key structural change: instead of looping over inventory states (61 Python
    iterations per step), we build the full (n_q, n_d, n_d) cost tensor and
    do a single argmax per step. This reduces 61,000 Python-loop iterations
    to 1,000, freeing time for a finer depth grid.
    
    The Bellman update for state (n, q) with bid depth d_b and ask depth d_a:
      J = d_b*p_b + d_a*p_a 
          + p_b*(1-p_a)*V[q+1] + (1-p_b)*p_a*V[q-1] 
          + (p_b*p_a + (1-p_b)*(1-p_a))*V[q]
          - (gamma/2)*sigma^2*q^2*dt
    
    Vectorized over q: build (n_q, n_d, n_d) tensor, argmax over last two axes.
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

    # Depth grid: log-spaced for fine resolution at small depths
    depth_min = 0.0005
    depth_max = 12.0 / k
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)

    # Precompute fill probabilities and depth*prob
    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    depth_p = depth_grid * p_fill  # expected immediate gain

    # DP tables
    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    # Inventory cost
    inv_cost = (gamma / 2.0) * sigma * sigma * q_range * q_range * dt

    # Precompute shifted V arrays for all q (these change each step)
    # V_plus[q] = V[n+1, q+1] if q < Q else V[n+1, Q]
    # V_minus[q] = V[n+1, q-1] if q > -Q else V[n+1, -Q]
    
    # For vectorization: 
    # D1[q] = V_plus[q] - V[q]  (value of moving to q+1)
    # Dm1[q] = V_minus[q] - V[q]  (value of moving to q-1)
    # D2[q] = V_plus[q] + V_minus[q] - 2*V[q]
    
    # Boundary masks
    bid_mask = np.ones(n_q, dtype=bool)  # can place bid
    bid_mask[-1] = False  # q = Q: no bid
    ask_mask = np.ones(n_q, dtype=bool)  # can place ask
    ask_mask[0] = False   # q = -Q: no ask

    for n in range(steps - 1, -1, -1):
        # Time guard
        if time.time() - _START_TIME > TIME_LIMIT - 8.0:
            # Fill remaining steps with HMM
            for n2 in range(n, -1, -1):
                tau2 = T - n2 * dt
                skew = (gamma * sigma * sigma * q_range * tau2) / (A * k)
                base = 1.0 / k
                bt = np.clip(base + skew, depth_min, depth_max)
                at = np.clip(base - skew, depth_min, depth_max)
                bt[-1] = np.inf
                at[0] = np.inf
                bid_table[n2, :] = bt
                ask_table[n2, :] = at
            break

        Vn1 = V[n + 1]  # shape (n_q,)

        # Compute shifted values
        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]

        D1 = V_plus - Vn1       # shape (n_q,)
        Dm1 = V_minus - Vn1     # shape (n_q,)
        D2 = V_plus + V_minus - 2.0 * Vn1  # shape (n_q,)

        # Build the (n_q, n_d, n_d) cost tensor
        # For state q, bid depth i, ask depth j:
        #   J[q,i,j] = depth_p[i] + depth_p[j]
        #             + p_fill[i] * D1[q] * (1 - p_fill[j])   [bid fills, ask doesn't -> q+1]
        #             + p_fill[j] * Dm1[q] * (1 - p_fill[i])   [ask fills, bid doesn't -> q-1]
        #             + (p_fill[i]*p_fill[j] + (1-p_fill[i])*(1-p_fill[j])) * 0  [both or neither -> q]
        #             - inv_cost[q]
        #
        # Wait, let me re-derive more carefully:
        # E[V_next] = p_b*(1-p_a)*V[q+1] + (1-p_b)*p_a*V[q-1] + (p_b*p_a + (1-p_b)*(1-p_a))*V[q]
        #           = V[q] + p_b*(1-p_a)*(V[q+1]-V[q]) + (1-p_b)*p_a*(V[q-1]-V[q])
        #           = V[q] + p_b*(1-p_a)*D1[q] + (1-p_b)*p_a*Dm1[q]
        #
        # J = depth_p[i] + depth_p[j] + V[q] + p_b*(1-p_a)*D1[q] + (1-p_b)*p_a*Dm1[q] - cost[q]
        #
        # Expanding:
        # J = depth_p[i] + depth_p[j] + V[q] + p_b*D1 - p_b*p_a*D1 + p_a*Dm1 - p_b*p_a*Dm1 - cost
        #   = depth_p[i] + depth_p[j] + V[q] + p_b*D1 + p_a*Dm1 - p_b*p_a*(D1+Dm1) - cost
        #
        # Note: D1+Dm1 = D2 (the second difference)
        #
        # So: J[q,i,j] = depth_p[i] + depth_p[j] + V[q] + p_b[i]*D1[q] + p_a[j]*Dm1[q] - p_b[i]*p_a[j]*D2[q] - cost[q]

        # Shapes:
        # depth_p: (n_d,)
        # p_fill: (n_d,)
        # D1, Dm1, D2, Vn1, inv_cost: (n_q,)
        
        # term_dp: (n_q, n_d, n_d) = depth_p[i] + depth_p[j]
        term_dp = depth_p[None, :, None] + depth_p[None, None, :]  # (1, n_d, n_d) broadcast to (n_q, n_d, n_d)
        
        # term_v: (n_q, 1, 1) = V[q] - cost[q]
        term_v = (Vn1 - inv_cost)[:, None, None]
        
        # term_p_b: (n_q, n_d, 1) = p_b[i] * D1[q]
        term_pb = (p_fill[None, :, None] * D1[:, None, None])  # (1, n_d, 1) * (n_q, 1, 1) -> (n_q, n_d, 1)
        
        # term_p_a: (n_q, 1, n_d) = p_a[j] * Dm1[q]
        term_pa = (p_fill[None, None, :] * Dm1[:, None, None])  # (1, 1, n_d) * (n_q, 1, 1) -> (n_q, 1, n_d)
        
        # term_cross: (n_q, n_d, n_d) = -p_b[i]*p_a[j]*D2[q]
        term_cross = -(p_fill[None, :, None] * p_fill[None, None, :] * D2[:, None, None])
        
        # Total: (n_q, n_d, n_d)
        J = term_dp + term_v + term_pb + term_pa + term_cross

        # Apply boundary masks: for q=Q, only allow "no bid" (set bid cost to -inf for all finite depths)
        # Actually, we need to handle this differently. For q=Q, bid is forbidden.
        # We can set the bid dimension to a single "no quote" option with 0 gain and 0 prob.
        # Simpler: after argmax, override boundary states.
        
        # Find optimal (i, j) for each q
        # Reshape to (n_q, n_d*n_d), argmax along last axis
        J_flat = J.reshape(n_q, -1)
        flat_idx = np.argmax(J_flat, axis=1)  # shape (n_q,)
        
        i_max = flat_idx // n_depths  # shape (n_q,)
        j_max = flat_idx % n_depths   # shape (n_q,)
        
        V[n, :] = J_flat[np.arange(n_q), flat_idx]
        bid_table[n, :] = depth_grid[i_max]
        ask_table[n, :] = depth_grid[j_max]

    # Boundary conditions
    bid_table[:, -1] = np.inf  # q = Q
    ask_table[:, 0] = np.inf   # q = -Q

    return bid_table, ask_table


def _compute_vectorized_dp_fine(params, n_depths=1200):
    """Finer grid version for when time allows."""
    return _compute_vectorized_dp(params, n_depths=n_depths)


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
        # Choose grid resolution based on remaining time
        remaining = TIME_LIMIT - elapsed
        if remaining > 200.0:
            n_d = 1200
        elif remaining > 120.0:
            n_d = 800
        elif remaining > 60.0:
            n_d = 500
        else:
            n_d = 300
        
        try:
            bid_table, ask_table = _compute_vectorized_dp(params, n_depths=n_d)
        except Exception:
            # Fallback: try coarser
            try:
                bid_table, ask_table = _compute_vectorized_dp(params, n_depths=200)
            except Exception:
                # Last resort: HMM for all steps
                steps = params['steps']
                Q = params['Q']
                dt = params['T'] / steps
                bid_table = np.full((steps, 2*Q+1), 1.0/params['k'])
                ask_table = np.full((steps, 2*Q+1), 1.0/params['k'])
                bid_table[:, -1] = np.inf
                ask_table[:, 0] = np.inf
                for n2 in range(steps):
                    tau2 = params['T'] - n2 * dt
                    q_range = np.arange(-Q, Q+1, dtype=np.float64)
                    skew = (params['gamma'] * params['sigma']**2 * q_range * tau2) / (params['A'] * params['k'])
                    bid_table[n2, :] = np.clip(1.0/params['k'] + skew, 1e-6, 15.0/params['k'])
                    ask_table[n2, :] = np.clip(1.0/params['k'] - skew, 1e-6, 15.0/params['k'])
                    bid_table[n2, -1] = np.inf
                    ask_table[n2, 0] = np.inf
        _policy_cache[key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[key]

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
