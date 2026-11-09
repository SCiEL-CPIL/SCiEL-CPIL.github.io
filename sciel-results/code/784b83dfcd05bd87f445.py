"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = _START_TIME + 270.0


def _solve_pde_policy(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    n_steps = steps
    q_max = Q

    # Depth grid - use a non-uniform grid that's dense near 0 and sparse at large depths
    # Optimal depth is typically around 1/k to 3/k
    max_depth = 4.0 / k
    n_depth = 48
    # Use a grid that's denser near small depths
    depth_grid = np.linspace(0.0, max_depth, n_depth)

    # Fill probability for each depth
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)

    # Value function: (n_steps+1, 2*Q+1)
    V = np.zeros((n_steps + 1, 2 * Q + 1), dtype=np.float64)
    best_bid_idx = np.zeros((n_steps + 1, 2 * Q + 1), dtype=np.int32)
    best_ask_idx = np.zeros((n_steps + 1, 2 * Q + 1), dtype=np.int32)

    # Backward induction
    for n in range(n_steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            break

        # Vectorize over inventory for speed
        # For each inventory q, we need to find the best (bi, ai) pair
        # This is a 2D grid search, so we can vectorize over q
        for q in range(-Q, Q + 1):
            qi = q + Q
            can_bid = q < Q
            can_ask = q > -Q

            # Build the value matrix over (bi, ai)
            # Shape: (n_depth, n_depth)
            # For each (bi, ai):
            #   reward = depth_grid[bi] * pb + depth_grid[ai] * pa - holding_cost
            #   expected_V = ...

            pb_all = fill_probs.copy() if can_bid else np.zeros(n_depth)
            pa_all = fill_probs.copy() if can_ask else np.zeros(n_depth)

            # PB[i] for bid depth i, PA[j] for ask depth j
            PB = pb_all  # shape (n_depth,)
            PA = pa_all  # shape (n_depth,)

            # reward matrix: (n_depth, n_depth)
            # reward[i,j] = depth_grid[i]*PB[i] + depth_grid[j]*PA[j] - holding_cost
            holding_cost = 0.5 * gamma * sigma * sigma * q * q * dt

            reward_mat = np.outer(depth_grid, PB) + np.outer(depth_grid, PA) - holding_cost
            # Wait, that's wrong. reward[i,j] = depth_grid[i]*PB[i] + depth_grid[j]*PA[j] - holding_cost
            # outer(depth_grid, PB) gives [i,j] = depth_grid[i]*PB[j] -- that's wrong
            # Let me fix:
            # reward[i,j] = depth_grid[i] * PB[i] + depth_grid[j] * PA[j] - holding_cost
            reward_mat = np.outer(depth_grid, np.ones(n_depth)) * PB[:, None] + \
                         np.ones((n_depth, 1)) * (depth_grid * PA)[None, :] - holding_cost

            # Next inventory:
            # P(q-1) = PB[i] * (1 - PA[j])
            # P(q+1) = PA[j] * (1 - PB[i])
            # P(q)   = (1-PB[i]) * (1-PA[j])
            pqm1 = np.outer(PB, 1.0 - PA)
            pqp1 = np.outer(PA, 1.0 - PB)
            pqq = (1.0 - PB[:, None]) * (1.0 - PA[None, :])

            # expected_V matrix
            EV = np.zeros((n_depth, n_depth), dtype=np.float64)
            if q - 1 >= -Q:
                EV += pqm1 * V[n + 1, q - 1 + Q]
            if q + 1 <= Q:
                EV += pqp1 * V[n + 1, q + 1 + Q]
            EV += pqq * V[n + 1, qi]

            total = reward_mat + EV
            # Find max
            idx = np.unravel_index(np.argmax(total), total.shape)
            best_val = total[idx]
            V[n, qi] = best_val
            best_bid_idx[n, qi] = idx[0]
            best_ask_idx[n, qi] = idx[1]

    policy_bid = np.zeros((n_steps, 2 * Q + 1), dtype=np.float64)
    policy_ask = np.zeros((n_steps, 2 * Q + 1), dtype=np.float64)
    for n in range(n_steps):
        for q in range(-Q, Q + 1):
            qi = q + Q
            policy_bid[n, qi] = depth_grid[best_bid_idx[n, qi]]
            policy_ask[n, qi] = depth_grid[best_ask_idx[n, qi]]

    return policy_bid, policy_ask


def _get_policy(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _solve_pde_policy(sigma, A, k, gamma, T, steps, Q)
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    n_paths = len(inventory)
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    policy_bid, policy_ask = _get_policy(sigma, A, k, gamma, T, steps, Q)

    inv_clipped = np.clip(inventory, -Q, Q).astype(np.int64)

    bid_depths = policy_bid[n, inv_clipped + Q]
    ask_depths = policy_ask[n, inv_clipped + Q]

    bid_depths = np.where(inventory >= Q, 0.0, bid_depths)
    ask_depths = np.where(inventory <= -Q, 0.0, ask_depths)

    min_valid = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, 0.0)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, 0.0)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
