"""
Optimal market making with inventory risk via HJB discretization and lookup table.

Solves the finite-horizon MDP by backward induction with vectorized depth-pair
evaluation. At runtime, O(1) table lookup per path per step.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_DEADLINE = None


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 310  # 20s safety margin before 330s kill
    return _DEADLINE


def _solve_policy(sigma, A, k, gamma, T, steps, Q, depth_grid, deadline):
    """
    Solve the discrete HJB equation for the optimal policy using vectorized
    evaluation over all (bid_depth, ask_depth) pairs per time step.
    """
    dt = T / steps
    n_levels = 2 * Q + 1
    n_depths = len(depth_grid)

    V = np.zeros((steps + 1, n_levels))
    bid_opt = np.zeros((steps, n_levels))
    ask_opt = np.zeros((steps, n_levels))

    # Precompute fill probabilities
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)  # (n_depths,)

    # Precompute index arrays
    q_idx = np.arange(n_levels)
    bid_next_idx = np.minimum(q_idx + 1, n_levels - 1)
    ask_next_idx = np.maximum(q_idx - 1, 0)
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)

    # Precompute all probability combinations: (n_depths, n_depths)
    p_b = p_fill[:, np.newaxis]  # (n_depths, 1)
    p_a = p_fill[np.newaxis, :]  # (1, n_depths)
    p_ba = p_b * p_a            # (n_depths, n_depths)
    p_bn = p_b * (1.0 - p_a)    # (n_depths, n_depths)
    p_na = (1.0 - p_b) * p_a    # (n_depths, n_depths)
    p_nn = (1.0 - p_b) * (1.0 - p_a)  # (n_depths, n_depths)

    # Immediate reward components: p_b * bid_d + p_a * ask_d
    # Shape: (n_depths, n_depths)
    imm_base = p_fill[:, np.newaxis] * depth_grid[:, np.newaxis] + \
               p_fill[np.newaxis, :] * depth_grid[np.newaxis, :]

    completed = 0
    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            break

        V_next = V[n + 1]  # (n_levels,)
        inv_cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt  # (n_levels,)

        # Compute exp_V_next for each (di, ai) pair: shape (n_depths, n_depths, n_levels)
        # exp_V_next[di, ai, q] = p_ba[di,ai]*V_next[q] + p_bn[di,ai]*V_next[bid_next[q]]
        #                        + p_na[di,ai]*V_next[ask_next[q]] + p_nn[di,ai]*V_next[q]
        #
        # We can compute this as:
        #   (p_ba + p_nn) * V_next[q] + p_bn * V_next[bid_next[q]] + p_na * V_next[ask_next[q]]
        #
        # Shape: (n_depths, n_depths) * (n_levels,) -> (n_depths, n_depths, n_levels)
        p_same = p_ba + p_nn  # (n_depths, n_depths)
        # V_next terms: (n_levels,) -> broadcast to (n_depths, n_depths, n_levels)
        exp_V_next = (p_same[:, :, np.newaxis] * V_next[np.newaxis, np.newaxis, :] +
                      p_bn[:, :, np.newaxis] * V_next[np.newaxis, np.newaxis, bid_next_idx] +
                      p_na[:, :, np.newaxis] * V_next[np.newaxis, np.newaxis, ask_next_idx])

        # Total value: (n_depths, n_depths, n_levels)
        total = imm_base[:, :, np.newaxis] - inv_cost[np.newaxis, np.newaxis, :] + exp_V_next

        # Find best (di, ai) for each q
        # Reshape to (n_depths * n_depths, n_levels)
        total_flat = total.reshape(-1, n_levels)  # (n_depths^2, n_levels)
        best_idx = np.argmax(total_flat, axis=0)  # (n_levels,)
        V[n] = total_flat[best_idx, np.arange(n_levels)]

        # Decode best (di, ai)
        di = best_idx // n_depths
        ai = best_idx % n_depths
        bid_opt[n] = depth_grid[di]
        ask_opt[n] = depth_grid[ai]
        completed += 1

    return bid_opt, ask_opt, completed


def _build_policy(params, deadline):
    """Build or retrieve cached policy for given params."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _policy_cache:
        return _policy_cache[key]

    # Depth grid: centered around 1/k with reasonable spread
    min_d = 0.005
    max_d = 4.0 / k
    n_grid = 15
    depth_grid = np.linspace(min_d, max_d, n_grid)
    depth_grid = np.maximum(depth_grid, 1e-6)

    bid_opt, ask_opt, completed = _solve_policy(sigma, A, k, gamma, T, steps, Q, depth_grid, deadline)

    # If not all steps completed, fill remaining with last computed step or default
    if completed < steps:
        # Fill incomplete steps with the last fully computed step's policy
        if completed > 0:
            for n in range(completed, steps):
                bid_opt[n] = bid_opt[completed - 1]
                ask_opt[n] = ask_opt[completed - 1]
        else:
            # No steps completed; use risk-neutral depth
            default_d = 1.0 / k
            bid_opt[:] = default_d
            ask_opt[:] = default_d

    _policy_cache[key] = (bid_opt, ask_opt, depth_grid)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (4000,).
    Each value is the distance below/above mid to quote.
    np.inf means no quote.
    """
    deadline = _get_deadline()

    n_paths = len(inventory)

    try:
        bid_opt, ask_opt, depth_grid = _build_policy(params, deadline)
    except Exception:
        depth = np.full(n_paths, 1.0 / params["k"])
        return depth, depth.copy()

    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = min(n, params['steps'] - 1)
    n = max(n, 0)

    Q = params['Q']

    q_idx = np.clip(inventory, -Q, Q) + Q
    q_idx = np.array(q_idx, dtype=np.int32)

    bid_depths = bid_opt[n, q_idx]
    ask_depths = ask_opt[n, q_idx]

    min_valid = -3.0 / params['k']
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_valid), ask_depths, np.inf)

    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
