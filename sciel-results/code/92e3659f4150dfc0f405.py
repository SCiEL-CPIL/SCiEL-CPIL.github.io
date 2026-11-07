"""Combined program: exact DP with vectorized depth optimization and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325  # safety margin before 330s kill
_cache = {}


def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth tables via dynamic programming.
    
    For each (time, inventory) state, we solve:
    V(n, q) = -risk(q) + max_{d_b, d_a} { p_b(d_b)*(d_b + delta_b) + p_a(d_a)*(d_a + delta_a) }
    
    where p(d) = 1 - exp(-A*exp(-k*d)*dt) and delta_b, delta_a are continuation value differences.
    
    The 1D maximization is done via a precomputed grid + local refinement.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma * sigma * q_vals * q_vals * dt

    # Value and policy tables
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    opt_bid = np.full((steps, n_inv), np.inf, dtype=np.float64)
    opt_ask = np.full((steps, n_inv), np.inf, dtype=np.float64)

    # Precompute depth grid for fast fill-probability lookup
    d_max = 15.0 / k
    n_grid = 512
    d_grid = np.linspace(1e-8, d_max, n_grid)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)

    # Precompute the objective on the grid for a given delta_V
    # obj(d) = p(d) * (d + delta_V)
    # We'll vectorize: for a batch of delta_V values, compute best grid index

    def batch_max_one_side(deltas, can_fill_mask):
        """
        For an array of delta_V values, find optimal depth and value.
        Returns (values, depths) arrays.
        """
        n = len(deltas)
        vals = np.zeros(n, dtype=np.float64)
        deps = np.full(n, np.inf, dtype=np.float64)

        active = can_fill_mask
        if not np.any(active):
            return vals, deps

        # For active entries, compute objective on grid
        # obj[i, j] = p_grid[j] * (d_grid[j] + deltas[i])
        # This is (n_active x n_grid) which might be large; do it in chunks
        active_indices = np.where(active)[0]
        n_active = len(active_indices)

        # Process in chunks to avoid memory issues
        chunk_size = 200
        for start in range(0, n_active, chunk_size):
            end = min(start + chunk_size, n_active)
            idx_chunk = active_indices[start:end]
            delta_chunk = deltas[idx_chunk]

            # obj shape: (chunk_len, n_grid)
            obj = p_grid[None, :] * (d_grid[None, :] + delta_chunk[:, None])
            best_idx = np.argmax(obj, axis=1)
            best_val = obj[np.arange(len(idx_chunk)), best_idx]
            best_d = d_grid[best_idx]

            # Local refinement for entries where best_val > 0
            refine_mask = best_val > 0
            if np.any(refine_mask):
                refine_indices = np.where(refine_mask)[0]
                for ri in refine_indices:
                    gi = best_idx[ri]
                    lower = d_grid[max(0, gi - 3)]
                    upper = d_grid[min(n_grid - 1, gi + 3)]
                    dv = delta_chunk[ri]

                    # Bisection-based refinement (faster than scipy for this simple 1D problem)
                    # f(d) = p(d)*(d + dv) is unimodal in d for our parameter ranges
                    lo, hi = lower, upper
                    for _ in range(30):
                        mid = 0.5 * (lo + hi)
                        if mid <= 0:
                            lo = mid
                            continue
                        p_mid = 1.0 - np.exp(-A * np.exp(-k * mid) * dt)
                        obj_mid = p_mid * (mid + dv)

                        p_lo = 1.0 - np.exp(-A * np.exp(-k * lo) * dt)
                        obj_lo = p_lo * (lo + dv)

                        if obj_mid > obj_lo:
                            lo = mid
                        else:
                            hi = mid

                    d_opt = 0.5 * (lo + hi)
                    p_opt = 1.0 - np.exp(-A * np.exp(-k * d_opt) * dt)
                    val_opt = p_opt * (d_opt + dv)

                    if val_opt > best_val[ri]:
                        best_val[ri] = val_opt
                        best_d[ri] = d_opt

            # Assign results
            for i, ai in enumerate(idx_chunk):
                if best_val[i] > 0:
                    vals[ai] = best_val[i]
                    deps[ai] = best_d[i]
                else:
                    vals[ai] = 0.0
                    deps[ai] = np.inf

        return vals, deps

    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback: simple heuristic for remaining steps
            for nn in range(n, -1, -1):
                for qi in range(n_inv):
                    q = qi - Q
                    base = 1.0 / k
                    # Skew based on inventory and remaining time
                    remaining = (steps - nn) * dt
                    skew = gamma * sigma * sigma * remaining * q * 0.02
                    b = max(1e-6, base * (1.0 + skew))
                    a = max(1e-6, base * (1.0 - skew))
                    if q >= Q:
                        b = np.inf
                    if q <= -Q:
                        a = np.inf
                    opt_bid[nn, qi] = b
                    opt_ask[nn, qi] = a
                    V[nn, qi] = 0.0
            break

        # For this time step, compute optimal depths for all inventory levels
        # delta_b(q) = V(n+1, q+1) - V(n+1, q)  for bid (q -> q+1)
        # delta_a(q) = V(n+1, q-1) - V(n+1, q)  for ask (q -> q-1)

        # Vectorize: compute deltas for all q
        # For q in [-Q, Q-1]: delta_b is defined
        # For q in [-Q+1, Q]: delta_a is defined

        bid_deltas = np.zeros(n_inv, dtype=np.float64)
        ask_deltas = np.zeros(n_inv, dtype=np.float64)
        bid_can = np.zeros(n_inv, dtype=bool)
        ask_can = np.zeros(n_inv, dtype=bool)

        for qi in range(n_inv):
            q = qi - Q
            if q < Q:
                bid_deltas[qi] = V[n + 1, qi + 1] - V[n + 1, qi]
                bid_can[qi] = True
            if q > -Q:
                ask_deltas[qi] = V[n + 1, qi - 1] - V[n + 1, qi]
                ask_can[qi] = True

        bid_vals, bid_deps = batch_max_one_side(bid_deltas, bid_can)
        ask_vals, ask_deps = batch_max_one_side(ask_deltas, ask_can)

        V[n, :] = V[n + 1, :] - risk + bid_vals + ask_vals
        opt_bid[n, :] = bid_deps
        opt_ask[n, :] = ask_deps

    return opt_bid, opt_ask


def _get_dp_tables(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key not in _cache:
        sigma, A, k, gamma, T, steps, Q = key
        opt_bid, opt_ask = _compute_dp(sigma, A, k, gamma, T, steps, Q)
        _cache[key] = (opt_bid, opt_ask)
    return _cache[key]


def quote(t, inventory, mid, params):
    """Return optimal bid/ask depths for each path."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    opt_bid, opt_ask = _get_dp_tables(params)

    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, q_idx].astype(np.float64)
    ask_depths = opt_ask[n, q_idx].astype(np.float64)

    # Enforce boundary conditions
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)

    # Safety: ensure valid values (not below -3/k, not NaN)
    min_valid = -3.0 / k + 1e-6
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
