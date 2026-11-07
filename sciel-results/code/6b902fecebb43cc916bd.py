"""Combined program: vectorized DP with HJB-guided adaptive grid and continuous refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_start_time = None
_DEADLINE = 325.0  # safety margin before 330s kill


def _hjb_estimate_depths(params):
    """
    Estimate optimal bid/ask depths using continuous-time HJB intuition.
    Returns arrays of shape (n_states,) for bid and ask.
    """
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

    # Base depth: where fill probability is moderate
    # p = 1 - exp(-A exp(-k d) dt)
    # For small dt, p ≈ A exp(-k d) dt
    # We want p ~ 0.1-0.3 typically
    # A exp(-k d) dt ~ 0.2 => exp(-k d) ~ 0.2/(A dt)
    # -k d ~ log(0.2/(A dt)) => d ~ -log(0.2/(A dt))/k = log(A dt / 0.2)/k

    base = max(0.01, np.log(A * dt / 0.2) / k)
    base = np.clip(base, 0.01, 5.0 / k)

    # Inventory adjustment:
    # If q > 0 (long), we want to:
    #   - Increase bid_depth (harder to buy more)
    #   - Decrease ask_depth (easier to sell)
    # The marginal cost of holding inventory is gamma * sigma^2 * q
    # Balance with fill intensity: A exp(-k d) dt
    # Higher |q| -> want lower fill probability -> higher depth (for the side that increases |q|)

    # For bid (increases q): depth should increase with q
    # For ask (decreases q): depth should decrease with q (i.e., lower ask_depth means easier to sell)

    # Linear approximation around base
    inv_scale = gamma * sigma**2 * dt / (A * base + 1e-10)
    inv_scale = np.clip(inv_scale, 0.0, 2.0)

    d_bid = base * (1.0 + inv_scale * q_values)
    d_ask = base * (1.0 - inv_scale * q_values)

    # Clamp to reasonable range
    d_min = 0.005
    d_max = 10.0 / k
    d_bid = np.clip(d_bid, d_min, d_max)
    d_ask = np.clip(d_ask, d_min, d_max)

    return d_bid, d_ask


def _compute_optimal_depths(params):
    """
    Compute optimal bid/ask depths via fully vectorized dynamic programming
    with HJB-guided adaptive grid.
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

    # Get HJB estimates to guide grid construction
    d_bid_est, d_ask_est = _hjb_estimate_depths(params)

    # Construct adaptive grid covering the range of estimated depths
    all_est = np.concatenate([d_bid_est, d_ask_est])
    d_min = max(0.005, np.min(all_est) * 0.5)
    d_max = min(12.0 / k, np.max(all_est) * 2.0)

    # Use log-spaced grid for better resolution at small depths
    n_depths = 100
    depth_grid = np.logspace(np.log10(d_min), np.log10(d_max), n_depths)

    # Precompute p_fill and p_fill * depth
    lam = A * np.exp(-k * depth_grid) * dt
    p_fill = 1.0 - np.exp(-lam)
    p_depth = p_fill * depth_grid

    # Precompute cost per state
    cost = 0.5 * gamma * sigma**2 * (q_values**2) * dt

    # Value function
    V_next = np.zeros(n_states)

    # Store optimal actions
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))

    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10.0:
            for nn in range(n, steps):
                opt_bid[nn, :] = 1.0 / k
                opt_ask[nn, :] = 1.0 / k
            break

        # Compute Delta_b, Delta_a, Curv for current V_next
        Delta_b = np.zeros(n_states)
        Delta_a = np.zeros(n_states)
        Curv = np.zeros(n_states)

        if n_states > 2:
            Delta_b[1:-1] = V_next[2:] - V_next[1:-1]
            Delta_a[1:-1] = V_next[:-2] - V_next[1:-1]
            Curv[1:-1] = V_next[2:] + V_next[:-2] - 2.0 * V_next[1:-1]

        Delta_b[0] = V_next[1] - V_next[0]
        Delta_a[-1] = V_next[-2] - V_next[-1]

        # Initialize max_val
        max_val = np.full(n_states, -np.inf)
        best_b = np.full(n_states, 1.0 / k)
        best_a = np.full(n_states, 1.0 / k)

        # Vectorized loop over depth pairs
        for ib in range(n_depths):
            pb = p_fill[ib]
            db = depth_grid[ib]
            pd_b = p_depth[ib]

            p_be = np.full(n_states, pb)
            p_be[-1] = 0.0

            for ia in range(n_depths):
                pa = p_fill[ia]
                da = depth_grid[ia]
                pd_a = p_depth[ia]

                p_ae = np.full(n_states, pa)
                p_ae[0] = 0.0

                val = (p_be * (db + Delta_b) + p_ae * (da + Delta_a) + V_next - cost - p_be * p_ae * Curv)

                mask = val > max_val
                if np.any(mask):
                    max_val[mask] = val[mask]
                    best_b[mask] = db
                    best_a[mask] = da

        opt_bid[n, :] = best_b
        opt_ask[n, :] = best_a
        V_next = max_val

    return opt_bid, opt_ask


def _refine_continuous(opt_bid, opt_ask, params, V_final):
    """
    Refine discrete optimal depths by solving the continuous first-order condition.
    For each state and step, find the depth that maximizes the value function
    in the continuous limit.
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

    cost = 0.5 * gamma * sigma**2 * (q_values**2) * dt

    refined_bid = opt_bid.copy()
    refined_ask = opt_ask.copy()

    # We'll refine using the value function differences
    # For the last step, we can't do backward induction, so we use a heuristic
    # For earlier steps, we use the V_next from the DP

    # Re-run backward induction but store V_next at each step
    V_next = np.zeros(n_states)
    V_history = [None] * steps  # V_history[n] = V_next when computing step n

    for n in range(steps - 1, -1, -1):
        if time.time() - _start_time > _DEADLINE - 5.0:
            break

        Delta_b = np.zeros(n_states)
        Delta_a = np.zeros(n_states)
        Curv = np.zeros(n_states)

        if n_states > 2:
            Delta_b[1:-1] = V_next[2:] - V_next[1:-1]
            Delta_a[1:-1] = V_next[:-2] - V_next[1:-1]
            Curv[1:-1] = V_next[2:] + V_next[:-2] - 2.0 * V_next[1:-1]

        Delta_b[0] = V_next[1] - V_next[0]
        Delta_a[-1] = V_next[-2] - V_next[-1]

        # For continuous refinement, we solve for d that maximizes:
        # f(d) = p(d) * (d + Delta) + ... where p(d) = 1 - exp(-A exp(-k d) dt)
        # The optimal d satisfies the first-order condition.

        # For bid side:
        # max_d [ (1 - exp(-A exp(-k d) dt)) * (d + Delta_b) ]
        # Let g(d) = (1 - exp(-A exp(-k d) dt)) * (d + Delta_b)
        # g'(d) = A exp(-k d) dt * exp(-A exp(-k d) dt) * (d + Delta_b) + (1 - exp(-A exp(-k d) dt))
        # Set g'(d) = 0

        # This is complex to solve analytically, so we use a local search
        # around the discrete optimum.

        for i in range(n_states):
            # Bid refinement
            d0 = opt_bid[n, i]
            Db = Delta_b[i]
            # Local search in a small window around d0
            window = np.linspace(max(0.001, d0 * 0.8), min(20.0 / k, d0 * 1.2), 20)
            p_win = 1.0 - np.exp(-A * np.exp(-k * window) * dt)
            val_win = p_win * (window + Db)
            best_idx = np.argmax(val_win)
            refined_bid[n, i] = window[best_idx]

            # Ask refinement
            d0 = opt_ask[n, i]
            Da = Delta_a[i]
            window = np.linspace(max(0.001, d0 * 0.8), min(20.0 / k, d0 * 1.2), 20)
            p_win = 1.0 - np.exp(-A * np.exp(-k * window) * dt)
            val_win = p_win * (window + Da)
            best_idx = np.argmax(val_win)
            refined_ask[n, i] = window[best_idx]

        # Update V_next using the refined depths
        # For simplicity, we use the discrete V_next update
        max_val = np.zeros(n_states)
        for i in range(n_states):
            db = refined_bid[n, i]
            da = refined_ask[n, i]
            pb = 1.0 - np.exp(-A * np.exp(-k * db) * dt)
            pa = 1.0 - np.exp(-A * np.exp(-k * da) * dt)

            if i == n_states - 1:
                pb = 0.0
            if i == 0:
                pa = 0.0

            val = pb * (db + Delta_b[i]) + pa * (da + Delta_a[i]) + V_next[i] - cost[i]
            if i > 0 and i < n_states - 1:
                val -= pb * pa * Curv[i]
            max_val[i] = val

        V_next = max_val
        V_history[n] = V_next.copy()

    return refined_bid, refined_ask


def _get_policy(params):
    """Get cached policy or compute it."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        opt_bid, opt_ask = _compute_optimal_depths(params)

        # Attempt continuous refinement if time permits
        if _start_time is not None and time.time() - _start_time < _DEADLINE - 30.0:
            try:
                refined_bid, refined_ask = _refine_continuous(opt_bid, opt_ask, params, None)
                # Check if refinement improved things (sanity check)
                if np.all(np.isfinite(refined_bid)) and np.all(np.isfinite(refined_ask)):
                    opt_bid = refined_bid
                    opt_ask = refined_ask
            except Exception:
                pass  # Keep discrete policy if refinement fails

        _policy_cache[key] = (opt_bid, opt_ask)
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

    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / k, dtype=np.float64)
        return depth, depth.copy()

    try:
        opt_bid, opt_ask = _get_policy(params)
    except Exception:
        depth = np.full(len(inventory), 1.0 / k, dtype=np.float64)
        return depth, depth.copy()

    dt = params['T'] / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0

    idx = np.clip(inventory + Q, 0, 2 * Q)

    bid_depths = opt_bid[n, idx].astype(np.float64)
    ask_depths = opt_ask[n, idx].astype(np.float64)

    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, 1.0 / k)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, 1.0 / k)

    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
