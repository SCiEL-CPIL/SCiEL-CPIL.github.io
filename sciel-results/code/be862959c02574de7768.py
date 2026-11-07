"""Market-making quote strategy using fully vectorized backward DP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_dp_cache = {}
_DEADLINE = 295.0
_start_time = None


def _precompute_dp_vectorized(sigma, A, k, gamma, T, steps, Q):
    """
    Fully vectorized backward DP. All inventory states processed simultaneously
    at each time step. Uses a fine log-spaced depth grid.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _dp_cache:
        return _dp_cache[key]

    dt = T / steps
    n_q = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1, dtype=np.float64)

    # Fine log-spaced depth grid: 200 points from 0.005/k to 15/k
    n_d = 200
    kd_min = 0.005
    kd_max = 15.0
    depth_grid = np.geomspace(kd_min, kd_max, n_d) / k

    # Precompute fill probabilities and rewards for each depth
    exp_kd = np.exp(-k * depth_grid)
    fill_prob = 1.0 - np.exp(-A * exp_kd * dt)
    depth_reward = depth_grid * fill_prob  # d * p(d)

    # Inventory cost per state
    inv_cost = 0.5 * gamma * sigma**2 * q_indices**2 * dt

    # V_next[q] for all q simultaneously
    V_next = np.zeros(n_q)

    # Precompute masks for boundary conditions
    can_bid = np.array([q < Q for q in q_indices])  # (n_q,)
    can_ask = np.array([q > -Q for q in q_indices])  # (n_q,)

    # For interior states (can_bid AND can_ask), we need 2D optimization.
    # For boundary states, 1D optimization.
    # To keep it fully vectorized, we handle all states with the 2D approach
    # but mask out invalid sides.

    # Store optimal depths: (steps, n_q)
    opt_bid = np.full((steps, n_q), np.inf)
    opt_ask = np.full((steps, n_q), np.inf)

    # For the 2D case, we need to evaluate over all (d_b, d_a) pairs.
    # Shape: (n_q, n_d, n_d)
    # V_next contributions:
    #   V_q[i] = V_next[i]
    #   V_qp1[i] = V_next[i+1] if i+1 < n_q else 0
    #   V_qm1[i] = V_next[i-1] if i-1 >= 0 else 0

    V_q = V_next  # (n_q,)
    V_qp1 = np.zeros(n_q)
    V_qm1 = np.zeros(n_q)
    V_qp1[:-1] = V_next[1:]  # V_qp1[i] = V_next[i+1] for i < n_q-1
    V_qm1[1:] = V_next[:-1]  # V_qm1[i] = V_next[i-1] for i > 0

    # Now expand to (n_q, n_d, n_d) for the 2D optimization
    # pb shape: (n_d,) -> broadcast to (n_q, n_d, n_d)
    # pa shape: (n_d,) -> broadcast to (n_q, n_d, n_d)

    # For each state i:
    #   total[i, ib, ia] = depth_reward[ib]*can_bid[i] + depth_reward[ia]*can_ask[i] - inv_cost[i]
    #                      + (1-pb[ib])*(1-pa[ia])*V_q[i]
    #                      + pb[ib]*(1-pa[ia])*V_qp1[i]*can_bid[i]
    #                      + (1-pb[ib])*pa[ia]*V_qm1[i]*can_ask[i]

    # We can't fully vectorize the 2D max over (n_d, n_d) for all n_q at once
    # without enormous memory (61 * 200 * 200 * 8 bytes = ~200 MB per step, times 1000 steps is too much).
    # But we can do it step by step.

    # Actually, let's just do it. 61 * 200 * 200 = 2.44M floats = ~20 MB. That's fine per step.

    for t in range(steps - 1, -1, -1):
        if time.time() - _start_time > _DEADLINE:
            # Fill remaining with default
            default_d = 1.0 / k
            opt_bid[:t+1] = default_d
            opt_ask[:t+1] = default_d
            _dp_cache[key] = (opt_bid, opt_ask)
            return (opt_bid, opt_ask)

        # Recompute V_qp1 and V_qm1 from current V_next
        V_q = V_next.copy()
        V_qp1 = np.zeros(n_q)
        V_qm1 = np.zeros(n_q)
        V_qp1[:-1] = V_next[1:]
        V_qm1[1:] = V_next[:-1]

        # Build the full (n_q, n_d, n_d) array
        # bid_reward part: (n_q, n_d, 1)
        br = depth_reward[None, :, None] * can_bid[:, None, None]  # (n_q, n_d, 1)
        # ask_reward part: (n_q, 1, n_d)
        ar = depth_reward[None, None, :] * can_ask[:, None, None]  # (n_q, 1, n_d)
        # inv_cost: (n_q, 1, 1)
        ic = inv_cost[:, None, None]

        # pb: (1, n_d, 1)
        pb = fill_prob[None, :, None]
        # pa: (1, 1, n_d)
        pa = fill_prob[None, None, :]

        # V terms: (n_q, 1, 1)
        Vq = V_q[:, None, None]
        Vqp1 = V_qp1[:, None, None]
        Vqm1 = V_qm1[:, None, None]

        # Mask for bid/ask validity in the 2D grid
        # If can_bid[i] is False, pb should effectively be 0 and Vqp1 term should be 0
        # If can_ask[i] is False, pa should effectively be 0 and Vqm1 term should be 0

        pb_eff = pb * can_bid[:, None, None]
        pa_eff = pa * can_ask[:, None, None]

        V_next_state = ((1 - pb_eff) * (1 - pa_eff) * Vq +
                        pb_eff * (1 - pa_eff) * Vqp1 +
                        (1 - pb_eff) * pa_eff * Vqm1)

        total = br + ar - ic + V_next_state

        # Find argmax over (n_d, n_d) for each state
        # Flatten last two dims
        total_flat = total.reshape(n_q, -1)
        best_idx = np.argmax(total_flat, axis=1)  # (n_q,)

        # Unflatten
        best_ib = best_idx // n_d
        best_ia = best_idx % n_d

        V_curr = total_flat[np.arange(n_q), best_idx]

        opt_bid[t, :] = depth_grid[best_ib]
        opt_ask[t, :] = depth_grid[best_ia]

        # Set inf for invalid sides
        opt_bid[t, ~can_bid] = np.inf
        opt_ask[t, ~can_ask] = np.inf

        V_next = V_curr

    _dp_cache[key] = (opt_bid, opt_ask)
    return (opt_bid, opt_ask)


def quote(t, inventory, mid, params):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    dt = T / steps
    t_idx = int(round(t / dt))
    t_idx = min(t_idx, steps - 1)
    t_idx = max(t_idx, 0)

    try:
        opt_bid, opt_ask = _precompute_dp_vectorized(sigma, A, k, gamma, T, steps, Q)
    except Exception:
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()

    inv_idx = np.clip(inventory, -Q, Q) + Q
    inv_idx = inv_idx.astype(int)

    bid_depths = opt_bid[t_idx, inv_idx]
    ask_depths = opt_ask[t_idx, inv_idx]

    # Ensure valid: replace non-positive with inf
    bid_depths = np.where(bid_depths > 0, bid_depths, np.inf)
    ask_depths = np.where(ask_depths > 0, ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
