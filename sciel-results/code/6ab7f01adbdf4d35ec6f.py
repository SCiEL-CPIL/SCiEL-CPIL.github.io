"""Rewritten program: vectorized DP with log-spaced depth grid and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed policies per regime
_policy_cache = {}
_DEADLINE = 320.0  # safety margin before 330s kill
_start_time = None


def _compute_optimal_depths(params):
    """
    Compute optimal bid/ask depths via fully vectorized dynamic programming.
    Returns (bid_depths, ask_depths) arrays of shape (steps, 2*Q+1).
    """
    global _start_time, _DEADLINE
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

    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)

    # Log-spaced depth grid for better resolution at small depths
    d_min = 0.005
    d_max = 12.0 / k
    n_depths = 80
    depth_grid = np.logspace(np.log10(d_min), np.log10(d_max), n_depths)

    # Precompute p_fill and p_fill * depth
    lam = A * np.exp(-k * depth_grid) * dt
    p_fill = 1.0 - np.exp(-lam)
    p_depth = p_fill * depth_grid  # expected earnings per fill

    # Precompute cost per state
    cost = 0.5 * gamma * sigma**2 * (q_values**2) * dt

    # Value function: V_next[i] where i is index into q_values
    V_next = np.zeros(n_states)

    # Store optimal actions
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))

    # For vectorization, we'll iterate over depth pairs but vectorize over states.
    # To keep memory manageable, we process in chunks of depth pairs.
    # n_depths^2 = 6400 pairs, n_states = 61. 6400*61 = 390400 floats per step. Fine.

    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10.0:
            # Return what we have so far (remaining steps get default depths)
            for nn in range(n, steps):
                opt_bid[nn, :] = 1.0 / k
                opt_ask[nn, :] = 1.0 / k
            break

        # Compute Delta_b, Delta_a, Curv for current V_next
        Delta_b = np.zeros(n_states)
        Delta_a = np.zeros(n_states)
        Curv = np.zeros(n_states)

        # Interior states
        if n_states > 2:
            Delta_b[1:-1] = V_next[2:] - V_next[1:-1]
            Delta_a[1:-1] = V_next[:-2] - V_next[1:-1]
            Curv[1:-1] = V_next[2:] + V_next[:-2] - 2.0 * V_next[1:-1]

        # Boundaries
        Delta_b[0] = V_next[1] - V_next[0]
        Delta_a[-1] = V_next[-2] - V_next[-1]

        # Initialize max_val
        max_val = np.full(n_states, -np.inf)
        best_b = np.full(n_states, 1.0 / k)
        best_a = np.full(n_states, 1.0 / k)

        # Vectorized loop over depth pairs
        # For each pair (ib, ia), compute value for all states
        # p_be: pb for all states except last (which is 0)
        # p_ae: pa for all states except first (which is 0)

        # Precompute p_be and p_ae templates
        # p_be_template[ib] = [pb, pb, ..., pb, 0]
        # p_ae_template[ia] = [0, pa, pa, ..., pa]

        # We'll iterate over ib, ia but vectorize the state computation
        for ib in range(n_depths):
            pb = p_fill[ib]
            db = depth_grid[ib]
            pd_b = p_depth[ib]

            # p_be: all pb except last is 0
            p_be = np.full(n_states, pb)
            p_be[-1] = 0.0

            for ia in range(n_depths):
                pa = p_fill[ia]
                da = depth_grid[ia]
                pd_a = p_depth[ia]

                # p_ae: all pa except first is 0
                p_ae = np.full(n_states, pa)
                p_ae[0] = 0.0

                # Compute value for all states
                # val = p_be*(db + Delta_b) + p_ae*(da + Delta_a) + V_next - cost - p_be*p_ae*Curv
                # Simplify: p_be*db + p_be*Delta_b + p_ae*da + p_ae*Delta_a + V_next - cost - p_be*p_ae*Curv
                val = (p_be * (db + Delta_b) + p_ae * (da + Delta_a) + V_next - cost - p_be * p_ae * Curv)

                # Update max
                mask = val > max_val
                if np.any(mask):
                    max_val[mask] = val[mask]
                    best_b[mask] = db
                    best_a[mask] = da

        # Store optimal actions for step n
        opt_bid[n, :] = best_b
        opt_ask[n, :] = best_a

        # Update V_next
        V_next = max_val

    return opt_bid, opt_ask


def _get_policy(params):
    """Get cached policy or compute it."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_depths(params)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote depths based on precomputed optimal policy.
    """
    global _start_time, _DEADLINE
    if _start_time is None:
        _start_time = time.time()

    steps = params['steps']
    Q = params['Q']
    k = params['k']

    # Check deadline - if exceeded, return default
    if time.time() - _start_time > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / k, dtype=np.float64)
        return depth, depth.copy()

    # Get or compute policy
    try:
        opt_bid, opt_ask = _get_policy(params)
    except Exception:
        depth = np.full(len(inventory), 1.0 / k, dtype=np.float64)
        return depth, depth.copy()

    # Determine current step index
    dt = params['T'] / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0

    # Map inventory to index
    idx = np.clip(inventory + Q, 0, 2 * Q)

    # Get depths
    bid_depths = opt_bid[n, idx].astype(np.float64)
    ask_depths = opt_ask[n, idx].astype(np.float64)

    # Ensure no invalid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, 1.0 / k)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, 1.0 / k)

    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
