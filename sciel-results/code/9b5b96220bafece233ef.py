import numpy as np
import time

_solution_cache = {}

def _precompute_solution(sigma, A, k, gamma, T, steps, Q):
    start_time = time.time()
    deadline = 300.0

    dt = T / steps
    n_states = 2 * Q + 1

    # Precompute the optimal probability p* for a range of marginal value C
    # The objective for choosing probability p (which determines depth d) is:
    # max_p p * (d(p) + C)
    # where d(p) = -1/k * log( -log(1-p) / (A*dt) )

    # Estimate range of C
    max_cost = 0.5 * gamma * sigma**2 * Q**2
    # C is the difference in value function. It's bounded by the max possible reward/cost.
    # A safe range is usually a few times the max cost.
    C_min, C_max = -2.5 * max_cost, 2.5 * max_cost

    N_C = 4000
    C_grid = np.linspace(C_min, C_max, N_C)

    p_max = 1 - np.exp(-A * dt)

    # We want to find p* for each C.
    # Instead of a 2D grid of p and C which is O(N_C * N_P), we can use the fact
    # that the optimal p is monotonic in C?
    # Let f(p, C) = p * d(p) + p * C.
    # d(p) is increasing in p.
    # f_p = d(p) + p * d'(p) + C.
    # d'(p) > 0. So f_p is increasing in p?
    # Let's check d(p).
    # d = -1/k log( -log(1-p) / (A dt) )
    # Let u = -log(1-p). u' = 1/(1-p).
    # d = -1/k ( log(u) - log(A dt) )
    # d' = -1/k * u'/u = -1/k * (1/(1-p)) / (-log(1-p))
    # d' = 1 / ( k (1-p) (-log(1-p)) )
    # p * d' = p / ( k (1-p) (-log(1-p)) )
    # As p -> 1, d' -> inf. So f_p -> inf.
    # Thus the maximum is always at the boundary p_max?
    # Wait. d(p) goes to infinity as p -> 1.
    # So p * d(p) goes to infinity?
    # Let's check the model.
    # "bid hit with probability 1 - exp(-A exp(-k bid_depth) dt)"
    # If depth -> 0, prob -> 1.
    # If depth -> inf, prob -> 0.
    # d(p) is the depth corresponding to probability p.
    # If p is high, d is low.
    # If p is low, d is high.
    # Objective: p * (d + C).
    # If C is very negative, we want p=0 (d=inf).
    # If C is very positive, we want p=p_max (d=0).
    # Is there an interior maximum?
    # f(p) = p * d(p) + p * C.
    # f'(p) = d(p) + p d'(p) + C.
    # As p -> 0, d(p) -> inf. So f'(p) -> inf.
    # As p -> 1, d(p) -> 0, d'(p) -> inf. So f'(p) -> inf.
    # So f(p) is convex? It has a minimum?
    # Let's check values.
    # A=140, dt=0.001. A*dt = 0.14.
    # p=0.5. 1-p=0.5. -log(0.5)=0.693. ratio = 0.693/0.14 = 4.95. d = -log(4.95)/1.5 = -1.6/1.5 = -1.06.
    # Wait, depth must be positive?
    # "distances below and above mid... Values below -3/k... make the whole result invalid."
    # This implies depths can be negative? "distances below ... to quote".
    # Usually depth is positive distance.
    # But if d < 0, it means we are quoting *inside* the mid?
    # The formula d = -1/k log( -log(1-p) / (A dt) ).
    # If -log(1-p) / (A dt) > 1, then log(...) > 0, so d < 0.
    # -log(1-p) > A dt.
    # 1-p < exp(-A dt).
    # p > 1 - exp(-A dt) = p_max.
    # But p is capped at p_max by the model (prob cannot exceed 1 - exp(-A*0*dt) = 1 - 1 = 0? No).
    # p = 1 - exp(-A exp(-k d) dt).
    # If d=0, p = 1 - exp(-A dt).
    # If d < 0, exp(-k d) > 1, so p > 1 - exp(-A dt).
    # Max p is 1 (as d -> inf? No, as d -> -inf, exp(-kd) -> inf, p -> 1).
    # So p can go up to 1.
    # d(p) formula:
    # p = 1 - exp(-A exp(-k d) dt)
    # 1-p = exp(-A exp(-k d) dt)
    # -ln(1-p) = A exp(-k d) dt
    # exp(-k d) = -ln(1-p) / (A dt)
    # -k d = ln( -ln(1-p) / (A dt) )
    # d = -1/k ln( -ln(1-p) / (A dt) )
    # For d to be real, -ln(1-p) / (A dt) > 0.
    # Also, if -ln(1-p) / (A dt) = 1, d=0.
    # If -ln(1-p) / (A dt) < 1, ln(...) < 0, d > 0.
    # If -ln(1-p) / (A dt) > 1, ln(...) > 0, d < 0.
    # So p can range from 0 to 1.
    # p=0 => d=inf.
    # p=1 => d=-inf.

    # So we can choose any p in [0, 1].
    # f(p) = p d(p) + p C.
    # Let's sample p in [0, 1] and find argmax for each C.

    N_P = 2000
    p_grid = np.linspace(0, 1, N_P)
    # Handle p=1 carefully for log(1-p)
    p_safe = np.clip(p_grid, 1e-15, 1 - 1e-15)
    log_term = -np.log(1 - p_safe)
    ratio = log_term / (A * dt)
    # ratio > 0 always
    d_grid = -np.log(ratio) / k

    vals = p_grid[None, :] * (d_grid[None, :] + C_grid[:, None])
    idx_max = np.argmax(vals, axis=1)
    p_opt_grid = p_grid[idx_max]

    # Map p to d
    # d(p)
    p_opt_safe = np.clip(p_opt_grid, 1e-15, 1 - 1e-15)
    log_term_opt = -np.log(1 - p_opt_safe)
    ratio_opt = log_term_opt / (A * dt)
    d_opt_grid = -np.log(ratio_opt) / k

    # DP
    V = np.zeros((steps, n_states))
    d_b_opt = np.full((steps, n_states), np.inf)
    d_a_opt = np.full((steps, n_states), np.inf)

    q_vals = np.arange(-Q, Q + 1)

    for n in range(steps - 2, -1, -1):
        if time.time() > deadline - 20:
            # Fallback: simple constant depth
            depth = 1.0 / k
            d_b_opt[n:, :] = depth
            d_a_opt[n:, :] = depth
            break

        V_next = V[n + 1]

        delta_plus = np.zeros(n_states)
        delta_minus = np.zeros(n_states)

        if n_states > 1:
            delta_plus[:-1] = V_next[1:] - V_next[:-1]
            delta_minus[1:] = V_next[:-1] - V_next[1:]

        # Look up optimal p
        dp_clipped = np.clip(delta_plus, C_min, C_max)
        dm_clipped = np.clip(delta_minus, C_min, C_max)

        idx_b = np.searchsorted(C_grid, dp_clipped)
        idx_a = np.searchsorted(C_grid, dm_clipped)

        idx_b = np.clip(idx_b, 0, N_C - 1)
        idx_a = np.clip(idx_a, 0, N_C - 1)

        p_b = p_opt_grid[idx_b]
        p_a = p_opt_grid[idx_a]

        # Boundary conditions
        p_b[-1] = 0.0  # q = Q, no bid
        p_a[0] = 0.0   # q = -Q, no ask

        # Depths
        d_b = np.full(n_states, np.inf)
        d_a = np.full(n_states, np.inf)

        mask_b = p_b > 0
        mask_a = p_a > 0

        if np.any(mask_b):
            d_b[mask_b] = d_opt_grid[idx_b[mask_b]]
        if np.any(mask_a):
            d_a[mask_a] = d_opt_grid[idx_a[mask_a]]

        # Ensure depths are valid (not -inf or nan)
        # If d is very negative, it might be invalid?
        # "Values below -3/k ... make the whole result invalid"
        # So we must clip d to be >= -3/k.
        min_d = -3.0 / k
        d_b = np.where(d_b < min_d, min_d, d_b)
        d_a = np.where(d_a < min_d, min_d, d_a)

        d_b_opt[n, :] = d_b
        d_a_opt[n, :] = d_a

        # Reward
        reward_b = np.where(p_b > 0, p_b * d_b, 0.0)
        reward_a = np.where(p_a > 0, p_a * d_a, 0.0)

        # Cost
        cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt

        # Expected Value
        # E[V_next] = V_next + p_b delta_plus + p_a delta_minus - p_b p_a (delta_plus + delta_minus)
        # Note: delta_plus[-1] and delta_minus[0] are 0, and p_b[-1]=0, p_a[0]=0.
        ev_term = V_next + p_b * delta_plus + p_a * delta_minus - p_b * p_a * (delta_plus + delta_minus)

        V_curr = reward_b + reward_a - cost + ev_term
        V[n] = V_curr

    return d_b_opt, d_a_opt

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    key = (sigma, A, k, gamma, T, steps, Q)

    if key not in _solution_cache:
        try:
            d_b_opt, d_a_opt = _precompute_solution(sigma, A, k, gamma, T, steps, Q)
            _solution_cache[key] = (d_b_opt, d_a_opt)
        except Exception:
            depth = 1.0 / k
            return np.full(len(inventory), depth), np.full(len(inventory), depth)

    d_b_opt, d_a_opt = _solution_cache[key]

    dt = T / steps
    n = int(t / dt)
    if n >= steps:
        n = steps - 1

    idx = inventory + Q

    bid_depths = d_b_opt[n, idx]
    ask_depths = d_a_opt[n, idx]

    # Final safety check for invalid values
    min_d = -3.0 / k
    bid_depths = np.where(bid_depths < min_d, min_d, bid_depths)
    ask_depths = np.where(ask_depths < min_d, min_d, ask_depths)

    return bid_depths, ask_depths
