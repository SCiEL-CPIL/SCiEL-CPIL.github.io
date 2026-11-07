import numpy as np
import time

_T0 = time.time()
_DEADLINE = 290.0
_DP_CACHE = {}

def _solve_dp_grid(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _DP_CACHE:
        return _DP_CACHE[key]

    dt = T / steps
    n_states = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)
    d_max = 5.0 / k
    d_min = 0.0

    # Discrete depth grid
    n_grid = 512
    grid = np.linspace(d_min, d_max, n_grid)

    # Pre-compute probabilities and immediate rewards for all grid points
    # p[d] = 1 - exp(-A exp(-k d) dt)
    # r[d] = p[d] * d
    lam = A * np.exp(-k * grid) * dt
    p = 1.0 - np.exp(-lam)
    r = p * grid

    # V[t, q_idx] for t in 0..n_steps (t=n_steps is initial)
    V = np.zeros((steps + 1, n_states))
    opt_bid = np.zeros((steps + 1, n_states))
    opt_ask = np.zeros((steps + 1, n_states))

    costs = 0.5 * gamma * sigma**2 * q_indices**2 * dt

    for t in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            break

        V_next = V[t + 1]

        # Shifted value arrays
        V_p = np.empty(n_states)
        V_m = np.empty(n_states)
        V_0 = V_next.copy()

        V_p[1:] = V_next[:-1]
        V_p[0] = V_next[0]
        V_m[:-1] = V_next[1:]
        V_m[-1] = V_next[-1]

        D1 = V_p - V_0  # V_{q+1} - V_q
        D2 = V_m - V_0  # V_{q-1} - V_q
        D3 = V_p + V_m - 2 * V_0

        best_bid = np.zeros(n_states)
        best_ask = np.zeros(n_states)
        best_val = np.full(n_states, -np.inf)

        # For each state, find optimal (b_idx, a_idx)
        # E[V_next] = V_0 + p_b D1 + p_a D2 - p_b p_a D3
        # Total = r[b] + r[a] - cost + V_0 + p_b D1 + p_a D2 - p_b p_a D3
        #       = V_0 - cost + [r[b] + p_b D1] + [r[a] + p_a D2] - p_b p_a D3

        # Pre-compute per-grid contributions
        # For bid: contrib_b[i] = r[i] + p[i] * D1[q]  -> shape (n_grid, n_states)
        # For ask: contrib_a[i] = r[i] + p[i] * D2[q]
        # Cross:   cross[i,j] = p[i] * p[j] * D3[q]

        # Vectorized over states:
        # contrib_b shape (n_grid, n_states)
        contrib_b = r[:, None] + p[:, None] * D1[None, :]
        contrib_a = r[:, None] + p[:, None] * D2[None, :]
        # cross shape (n_grid, n_grid, n_states)
        cross = (p[:, None, :] * p[None, :, :]) * D3[None, None, :]

        # total[b_idx, a_idx, q] = contrib_b[b_idx, q] + contrib_a[a_idx, q] - cross[b_idx, a_idx, q]
        total = contrib_b[:, None, :] + contrib_a[None, :, :] - cross

        # Add V_0 - cost
        total += (V_0 - costs)[None, None, :]

        # Boundary masks
        # q = Q (index 2Q): no bid allowed -> force bid depth to 0 (index 0)
        # q = -Q (index 0): no ask allowed -> force ask depth to 0 (index 0)
        q_is_max = np.zeros(n_states, dtype=bool)
        q_is_max[2 * Q] = True
        q_is_min = np.zeros(n_states, dtype=bool)
        q_is_min[0] = True

        # For q_is_max: only consider a_idx (bid fixed at 0)
        # For q_is_min: only consider b_idx (ask fixed at 0)
        # For others: full 2D search

        # To keep memory manageable, process in chunks or use argmax with masking
        # Mask: for q_is_max, set all b_idx > 0 to -inf
        # For q_is_min, set all a_idx > 0 to -inf

        mask = np.ones((n_grid, n_grid, n_states), dtype=bool)
        if np.any(q_is_max):
            mask[1:, :, q_is_max] = False
        if np.any(q_is_min):
            mask[:, 1:, q_is_min] = False

        total_masked = np.where(mask, total, -np.inf)

        # Find best (b_idx, a_idx) per state
        flat = total_masked.reshape(n_grid * n_grid, n_states)
        best_flat = np.argmax(flat, axis=0)
        best_b_idx = best_flat // n_grid
        best_a_idx = best_flat % n_grid

        best_bid[t + 1] = grid[best_b_idx]
        best_ask[t + 1] = grid[best_a_idx]
        V[t + 1] = flat[best_flat, np.arange(n_states)]

    _DP_CACHE[key] = (opt_bid, opt_ask)
    return _DP_CACHE[key]

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE:
        k = params['k']
        d = 1.0 / k
        return np.full(len(inventory), d), np.full(len(inventory), d)

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    opt_bid, opt_ask = _solve_dp_grid(sigma, A, k, gamma, T, steps, Q)

    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    n = int(t / (T / steps))
    n = np.clip(n, 0, steps - 1)

    bid_depths = opt_bid[n, q_idx].copy()
    ask_depths = opt_ask[n, q_idx].copy()

    # Error-repair pass: clamp to valid range
    d_min_valid = 0.0
    d_max_valid = 5.0 / k
    bid_depths = np.clip(bid_depths, d_min_valid, d_max_valid)
    ask_depths = np.clip(ask_depths, d_min_valid, d_max_valid)

    # Ensure no NaN, inf, or values below -3/k
    bad_bid = ~np.isfinite(bid_depths) | (bid_depths < -3.0 / k)
    if np.any(bad_bid):
        bid_depths[bad_bid] = d_min_valid
    bad_ask = ~np.isfinite(ask_depths) | (ask_depths < -3.0 / k)
    if np.any(bad_ask):
        ask_depths[bad_ask] = d_min_valid

    return bid_depths, ask_depths
