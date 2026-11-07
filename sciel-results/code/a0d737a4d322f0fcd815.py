"""
Optimal market making via DP with vectorized state processing and adaptive depth refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 320.0  # safety margin before 330s kill


def _heuristic_depths(params, t, inventory):
    """Analytical fallback: time-varying base depth with inventory skew."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    q = inventory.astype(np.float64)
    t_remaining = np.maximum(T - t, 0.0)
    d0 = 1.0 / k
    skew_scale = gamma * sigma * sigma * t_remaining
    alpha = skew_scale / (A * k) if A > 0 and k > 0 else 0.0
    alpha = min(alpha, 6.0 / k)
    bid_depth = d0 + alpha * q
    ask_depth = d0 - alpha * q
    min_depth = 0.01 / k
    max_depth = 10.0 / k
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)
    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= -3.0 / k), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= -3.0 / k), ask_depth, np.inf)
    return bid_depth, ask_depth


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP
    with vectorized state processing and adaptive refinement.
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
    q_range = np.arange(-Q, Q + 1)

    # Depth grid: log-spaced for fine resolution at small depths
    depth_min = 0.005
    depth_max = 12.0 / k
    n_depths = 80
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)

    # Precompute fill probabilities and depth*prob for the grid
    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    depth_p = depth_grid * p_fill

    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    inv_cost_factor = (gamma / 2.0) * sigma**2 * dt

    # First pass: full backward DP with coarse grid (vectorized over states)
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 30:
            # Fallback: fill remaining with heuristic
            for n2 in range(n, -1, -1):
                for qi in range(n_q):
                    q = q_range[qi]
                    d = 1.0 / k + gamma * sigma**2 * q * (T - n2 * dt) / (A * k)
                    d = max(d, depth_min)
                    bid_table[n2, qi] = d
                    ask_table[n2, qi] = d
            bid_table[:, n_q - 1] = np.inf
            ask_table[:, 0] = np.inf
            return bid_table, ask_table

        Vn1 = V[n + 1]

        # Precompute V_plus and V_minus for all states
        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]

        D1 = V_plus - Vn1
        Dm1 = V_minus - Vn1
        D2 = V_plus + V_minus - 2.0 * Vn1

        # Process all states at this step
        for qi in range(n_q):
            q_val = q_range[qi]
            cost = inv_cost_factor * q_val * q_val

            if qi == n_q - 1:  # q = Q, no bid
                Pb = np.zeros(1)
                DPb = np.zeros(1)
            else:
                Pb = p_fill
                DPb = depth_p

            if qi == 0:  # q = -Q, no ask
                Pa = np.zeros(1)
                DPa = np.zeros(1)
            else:
                Pa = p_fill
                DPa = depth_p

            term1 = DPb[:, None] + DPa[None, :]
            term2 = Vn1[qi] + Pb[:, None] * D1[qi] + Pa[None, :] * Dm1[qi]
            term3 = -(Pb[:, None] * Pa[None, :]) * D2[qi]

            J = term1 + term2 + term3 - cost

            idx_max = np.unravel_index(np.argmax(J), J.shape)
            i_max, j_max = idx_max

            V[n, qi] = J[i_max, j_max]
            bid_table[n, qi] = depth_grid[i_max]
            ask_table[n, qi] = depth_grid[j_max]

    # Refinement passes: narrow the grid around previously best depths
    n_refine = 4
    for refine_pass in range(n_refine):
        if time.time() - _START_TIME > TIME_LIMIT - 20:
            break

        new_V = np.zeros((steps + 1, n_q))
        new_bid_table = np.zeros((steps, n_q))
        new_ask_table = np.zeros((steps, n_q))

        n_local = 60  # Number of local points per side

        for n in range(steps - 1, -1, -1):
            if time.time() - _START_TIME > TIME_LIMIT - 15:
                # Copy remaining from old tables
                new_V[n + 1:] = V[n + 1:]
                new_bid_table[n:] = bid_table[n:]
                new_ask_table[n:] = ask_table[n:]
                break

            Vn1 = new_V[n + 1]

            V_plus = np.empty(n_q)
            V_minus = np.empty(n_q)
            V_plus[:-1] = Vn1[1:]
            V_plus[-1] = Vn1[-1]
            V_minus[1:] = Vn1[:-1]
            V_minus[0] = Vn1[0]

            D1 = V_plus - Vn1
            Dm1 = V_minus - Vn1
            D2 = V_plus + V_minus - 2.0 * Vn1

            for qi in range(n_q):
                q_val = q_range[qi]
                cost = inv_cost_factor * q_val * q_val

                prev_bid = bid_table[n, qi]
                prev_ask = ask_table[n, qi]

                if qi == n_q - 1:  # q = Q, no bid
                    local_bid_depths = np.array([depth_min])
                    local_ask_depths = np.geomspace(
                        max(prev_ask * 0.5, depth_min),
                        min(prev_ask * 2.0, depth_max),
                        n_local
                    )
                else:
                    local_bid_depths = np.geomspace(
                        max(prev_bid * 0.5, depth_min),
                        min(prev_bid * 2.0, depth_max),
                        n_local
                    )
                    if qi == 0:  # q = -Q, no ask
                        local_ask_depths = np.array([depth_min])
                    else:
                        local_ask_depths = np.geomspace(
                            max(prev_ask * 0.5, depth_min),
                            min(prev_ask * 2.0, depth_max),
                            n_local
                        )

                # Compute for local bid grid
                exp_b = np.exp(-k * local_bid_depths)
                p_b = 1.0 - np.exp(-A * exp_b * dt)
                dp_b = local_bid_depths * p_b

                # Compute for local ask grid
                exp_a = np.exp(-k * local_ask_depths)
                p_a = 1.0 - np.exp(-A * exp_a * dt)
                dp_a = local_ask_depths * p_a

                nb = len(local_bid_depths)
                na = len(local_ask_depths)

                term1 = dp_b[:, None] + dp_a[None, :]
                term2 = Vn1[qi] + p_b[:, None] * D1[qi] + p_a[None, :] * Dm1[qi]
                term3 = -(p_b[:, None] * p_a[None, :]) * D2[qi]

                J = term1 + term2 + term3 - cost

                idx_max = np.unravel_index(np.argmax(J), J.shape)
                i_max, j_max = idx_max

                new_V[n, qi] = J[i_max, j_max]
                new_bid_table[n, qi] = local_bid_depths[i_max]
                new_ask_table[n, qi] = local_ask_depths[j_max]

        V = new_V
        bid_table = new_bid_table
        ask_table = new_ask_table

    # Post-process boundaries to return inf for no quote
    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf

    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    if time.time() - _START_TIME > TIME_LIMIT:
        return _heuristic_depths(params, t, inventory)

    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key not in _policy_cache:
        bid_table, ask_table = _compute_optimal_policy(params)
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

    # Ensure valid values
    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths), ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
