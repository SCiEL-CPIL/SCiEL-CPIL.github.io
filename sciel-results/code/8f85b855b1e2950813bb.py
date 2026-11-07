"""Market-making with inventory risk: DP over inventory grid with time-varying policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache DP value tables per parameter regime.
_V_TABLES = {}

# Global deadline guard (300 s hard limit).
_T0 = time.time()
_DEADLINE = 295.0  # seconds of wall clock budget for the whole run.


def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)


def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the discrete DP for the optimal policy.
    Returns a dict with optimal bid/ask depths for each step and inventory level.
    """
    if _remaining_budget() < 40.0:
        return None

    dt = T / steps
    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k

    # Inventory grid: use every integer from -Q to Q
    qs = np.arange(-Q, Q + 1, dtype=float)
    nq = len(qs)

    # V for n_remaining = 0 (terminal)
    V = np.zeros((nq,))

    # We'll store the optimal bid/ask depths for each step and inventory level.
    # bid_depths[n, i] is the optimal bid depth at step n for inventory index i.
    # ask_depths[n, i] similarly.
    bid_depths_table = np.full((steps, nq), np.inf, dtype=float)
    ask_depths_table = np.full((steps, nq), np.inf, dtype=float)

    # Precompute index mapping: for inventory q, what are the indices of q-1 and q+1?
    q_idx = np.arange(nq)
    lo = np.clip(q_idx - 1, 0, nq - 1)
    hi = np.clip(q_idx + 1, 0, nq - 1)

    # For boundary conditions: no bid fill at q = Q, no ask fill at q = -Q
    # q_idx == 0 corresponds to q = -Q, q_idx == nq-1 corresponds to q = Q

    # Search grid for depths
    # We'll search over a grid of bid and ask depths
    # Depth range: [0, max_depth]
    n_search = 30
    depth_grid = np.linspace(0.0, max_depth, n_search)

    # Precompute probabilities for each depth
    pb_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)  # (n_search,)
    pa_grid = pb_grid.copy()  # same formula for ask

    for n in range(steps, 0, -1):
        # n is the number of steps remaining (including current)
        # V is V(q, n-1) i.e. value with n-1 steps remaining
        Vprev = V.copy()

        # V(q-1) and V(q+1)
        Vq_minus = Vprev[lo]  # (nq,)
        Vq_plus = Vprev[hi]   # (nq,)

        # Risk charge: (nq,)
        risk_charge = 0.5 * gamma * sigma * sigma * qs ** 2 * dt

        # For each q, compute the total value for each (bid, ask) combination.
        # value[q, i, j] = (depth_grid[i] + depth_grid[j]) - risk_charge[q]
        #               + pb_grid[i] * Vq_minus[q] + pa_grid[j] * Vq_plus[q]
        #               + (1 - pb_grid[i] - pa_grid[j]) * Vprev[q]

        # Vectorize over q and depth combinations.
        # depth_grid: (n_search,)
        # bid_reward: (n_search,) = depth_grid
        # ask_reward: (n_search,) = depth_grid

        # term1: (nq, n_search, n_search)
        # We can factor this:
        # value = depth_grid[i] + depth_grid[j] - risk_charge[q]
        #       + pb_grid[i] * Vq_minus[q] + pa_grid[j] * Vq_plus[q]
        #       + Vprev[q] - pb_grid[i] * Vprev[q] - pa_grid[j] * Vprev[q]

        # = depth_grid[i] + pb_grid[i] * (Vq_minus[q] - Vprev[q])
        # + depth_grid[j] + pa_grid[j] * (Vq_plus[q] - Vprev[q])
        # - risk_charge[q] + Vprev[q]

        # So for each q, the optimal i and j can be found independently!
        # This is a huge simplification.

        # For bid: maximize over i: depth_grid[i] + pb_grid[i] * (Vq_minus[q] - Vprev[q])
        # For ask: maximize over j: depth_grid[j] + pa_grid[j] * (Vq_plus[q] - Vprev[q])

        # Let's define:
        # bid_val[q, i] = depth_grid[i] + pb_grid[i] * (Vq_minus[q] - Vprev[q])
        # ask_val[q, j] = depth_grid[j] + pa_grid[j] * (Vq_plus[q] - Vprev[q])

        # But we also have the constraint that bid and ask are independent in the objective
        # EXCEPT for the boundary conditions (no bid at Q, no ask at -Q) and the
        # fact that pb + pa should be <= 1 for the probability to make sense.
        # Actually, the model says bid and ask fills are independent, so the probability
        # of both happening is pb * pa, and the continuation is:
        # pb * pa * V(q-1+1) + pb*(1-pa) * V(q-1) + (1-pb)*pa * V(q+1) + (1-pb)*(1-pa) * V(q)
        # = pb * pa * V(q) + pb*(1-pa) * V(q-1) + (1-pb)*pa * V(q+1) + (1-pb)*(1-pa) * V(q)
        # = pb * pa * V(q) + pb * V(q-1) - pb*pa*V(q-1) + pa * V(q+1) - pb*pa*V(q+1)
        # + V(q) - pb*V(q) - pa*V(q) + pb*pa*V(q)
        # = V(q) + pb*(V(q-1) - V(q)) + pa*(V(q+1) - V(q))

        # So the continuation value is:
        # V(q) + pb * (V(q-1) - V(q)) + pa * (V(q+1) - V(q))

        # And the total value is:
        # depth_bid + depth_ask - risk_charge + V(q) + pb*(V(q-1)-V(q)) + pa*(V(q+1)-V(q))

        # This is separable in bid and ask! So we can optimize them independently.

        # bid_val[q, i] = depth_grid[i] + pb_grid[i] * (Vq_minus[q] - Vprev[q])
        # ask_val[q, j] = depth_grid[j] + pa_grid[j] * (Vq_plus[q] - Vprev[q])

        # Compute bid_val: (nq, n_search)
        bid_delta = Vq_minus - Vprev  # (nq,)
        bid_val = depth_grid[np.newaxis, :] + pb_grid[np.newaxis, :] * bid_delta[:, np.newaxis]

        # Compute ask_val: (nq, n_search)
        ask_delta = Vq_plus - Vprev  # (nq,)
        ask_val = depth_grid[np.newaxis, :] + pa_grid[np.newaxis, :] * ask_delta[:, np.newaxis]

        # Find optimal indices
        best_bid_idx = np.argmax(bid_val, axis=1)  # (nq,)
        best_ask_idx = np.argmax(ask_val, axis=1)  # (nq,)

        # Extract optimal depths
        opt_bid = depth_grid[best_bid_idx]  # (nq,)
        opt_ask = depth_grid[best_ask_idx]  # (nq,)

        # Apply boundary conditions:
        # No bid fill at q = Q (q_idx == nq-1)
        # No ask fill at q = -Q (q_idx == 0)
        # "No bid fill" means we set bid depth to np.inf (no quote)
        # "No ask fill" means we set ask depth to np.inf (no quote)

        # At q = Q (q_idx == nq-1): no bid fill
        opt_bid[nq - 1] = np.inf
        # At q = -Q (q_idx == 0): no ask fill
        opt_ask[0] = np.inf

        # Store in tables
        step_idx = steps - n
        bid_depths_table[step_idx, :] = opt_bid
        ask_depths_table[step_idx, :] = opt_ask

        # Update V: V_new[q] = depth_bid + depth_ask - risk_charge + Vprev[q]
        #            + pb*(Vq_minus - Vprev) + pa*(Vq_plus - Vprev)
        # = depth_bid + depth_ask - risk_charge + Vprev[q]
        #   + pb_grid[best_bid_idx] * bid_delta
        #   + pa_grid[best_ask_idx] * ask_delta

        V = (opt_bid + opt_ask - risk_charge + Vprev
             + pb_grid[best_bid_idx] * bid_delta
             + pa_grid[best_ask_idx] * ask_delta)

        # Handle np.inf in opt_bid/opt_ask: if depth is inf, probability is 0
        # pb_grid[best_bid_idx] when opt_bid is inf: pb = 1 - exp(-A*exp(-k*inf)*dt) = 1 - exp(0) = 0
        # So the formula still works since pb_grid is computed from depth_grid, not from opt_bid directly.
        # But we need to be careful: when opt_bid = np.inf, the actual pb is 0, not pb_grid[best_bid_idx].
        # Let's fix this: for q where opt_bid is inf, set the pb contribution to 0.
        # Similarly for ask.

        # Actually, let's recompute V more carefully.
        # For q where opt_bid is inf: pb = 0, so the term is 0.
        # For q where opt_ask is inf: pa = 0, so the term is 0.
        # And the depth contribution: if depth is inf, we don't earn that depth.
        # So V_new[q] = (depth_bid if finite else 0) + (depth_ask if finite else 0)
        #               - risk_charge + Vprev[q]
        #               + (pb if finite else 0) * bid_delta
        #               + (pa if finite else 0) * ask_delta

        # Let's redo:
        bid_finite = np.isfinite(opt_bid)
        ask_finite = np.isfinite(opt_ask)

        bid_depth_eff = np.where(bid_finite, opt_bid, 0.0)
        ask_depth_eff = np.where(ask_finite, opt_ask, 0.0)
        pb_eff = np.where(bid_finite, pb_grid[best_bid_idx], 0.0)
        pa_eff = np.where(ask_finite, pa_grid[best_ask_idx], 0.0)

        V = (bid_depth_eff + ask_depth_eff - risk_charge + Vprev
             + pb_eff * bid_delta + pa_eff * ask_delta)

        # Check time budget
        if n % 50 == 0 and _remaining_budget() < 25.0:
            return None

    return {
        'bid_depths': bid_depths_table,
        'ask_depths': ask_depths_table,
        'qs': qs,
        'nq': nq,
    }


def _interp_depth(q, qs, depths):
    """
    Linearly interpolate depths at continuous q values using grid qs.
    q: (N,) array of inventory values
    qs: (nq,) sorted grid
    depths: (nq,) array of depth values on the grid
    Returns: (N,) array of interpolated depths
    """
    # Handle np.inf: if either neighbor is inf, the result should be inf
    # Actually, for interpolation, if q is exactly on a grid point with inf, return inf.
    # For simplicity, let's do: if q is at boundary, return the boundary value.
    # Otherwise, interpolate. If either interpolated value is inf, return inf.

    q_min = qs[0]
    q_max = qs[-1]
    q_step = qs[1] - qs[0]

    # Clip q to valid range
    q_clipped = np.clip(q, q_min, q_max)

    idx = (q_clipped - q_min) / q_step
    i0 = np.floor(idx).astype(int)
    i1 = i0 + 1
    i0 = np.clip(i0, 0, len(qs) - 1)
    i1 = np.clip(i1, 0, len(qs) - 1)
    frac = idx - np.floor(idx)
    frac = np.clip(frac, 0.0, 1.0)

    d0 = depths[i0]
    d1 = depths[i1]

    # If either is inf, the result is inf (conservative: no quote if uncertain)
    # Actually, if one is inf and the other is finite, we should return the finite one
    # if the frac is closer to it. But for simplicity, let's just interpolate and
    # handle inf by: if both are inf, return inf. If one is inf, return the other.
    result = (1.0 - frac) * d0 + frac * d1

    # Handle inf cases
    both_inf = np.isinf(d0) & np.isinf(d1)
    d0_inf = np.isinf(d0) & ~np.isinf(d1)
    d1_inf = np.isinf(d1) & ~np.isinf(d0)

    result = np.where(both_inf, np.inf, result)
    result = np.where(d0_inf, d1, result)
    result = np.where(d1_inf, d0, result)

    # For out-of-range q, return boundary value
    out_low = q < q_min
    out_high = q > q_max
    result = np.where(out_low, depths[0], result)
    result = np.where(out_high, depths[-1], result)

    return result


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _V_TABLES:
        table = _solve_dp(sigma, A, k, gamma, T, steps, Q)
        _V_TABLES[key] = table
    table = _V_TABLES[key]

    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k

    if table is None:
        # Fall back to analytic shift
        shift = gamma * sigma * sigma * 0.5 / (A * k)
        q = inventory.astype(float)
        bid = np.clip(inv_k + q * shift, 0.0, max_depth)
        ask = np.clip(inv_k - q * shift, 0.0, max_depth)
        # Boundary conditions
        bid = np.where(q >= Q, np.inf, bid)
        ask = np.where(q <= -Q, np.inf, ask)
        bid = np.nan_to_num(bid, nan=inv_k, posinf=max_depth, neginf=0.0)
        ask = np.nan_to_num(ask, nan=inv_k, posinf=max_depth, neginf=0.0)
        bid = np.clip(bid, 0.0, max_depth)
        ask = np.clip(ask, 0.0, max_depth)
        # Re-apply boundary inf
        bid = np.where(inventory >= Q, np.inf, bid)
        ask = np.where(inventory <= -Q, np.inf, ask)
        return bid.astype(float), ask.astype(float)

    # Determine current step index
    dt = T / steps
    step_idx = int(round(t / dt))
    step_idx = np.clip(step_idx, 0, steps - 1)

    q = inventory.astype(float)

    # Interpolate the optimal depths for this step
    bid = _interp_depth(q, table['qs'], table['bid_depths'][step_idx])
    ask = _interp_depth(q, table['qs'], table['ask_depths'][step_idx])

    # Ensure valid values: clip finite values to [0, max_depth], keep inf as inf
    bid_finite = np.isfinite(bid)
    ask_finite = np.isfinite(ask)
    bid = np.where(bid_finite, np.clip(bid, 0.0, max_depth), bid)
    ask = np.where(ask_finite, np.clip(ask, 0.0, max_depth), ask)

    # Replace NaN with a safe default
    bid = np.where(np.isnan(bid), inv_k, bid)
    ask = np.where(np.isnan(ask), inv_k, ask)

    return bid.astype(float), ask.astype(float)
# EVOLVE-BLOCK-END
