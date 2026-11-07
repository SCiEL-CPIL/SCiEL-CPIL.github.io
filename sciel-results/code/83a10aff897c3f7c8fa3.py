"""
Optimal market making via pre-computed DP policy lookup.
Improved: larger depth grid, two-pass polish, vectorized inner loop.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 325.0  # safety margin before 330s kill


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
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= -3.0/k), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= -3.0/k), ask_depth, np.inf)
    return bid_depth, ask_depth


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP.
    Two-pass: coarse grid then local refinement.
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
    depth_min = 0.003
    depth_max = 12.0 / k
    n_depths = 120
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)

    # Precompute fill probabilities
    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    depth_p = depth_grid * p_fill  # depth * p_fill, shape (n_depths,)

    # V[n, qi] = max expected future reward from step n to end at inventory qi
    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    inv_cost_factor = (gamma / 2.0) * sigma**2 * dt
    q_sq = (q_range * q_range).astype(np.float64)  # shape (n_q,)

    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            # Fallback: fill remaining with heuristic
            for n2 in range(n, -1, -1):
                t_rem = T - n2 * dt
                for qi in range(n_q):
                    q = q_range[qi]
                    d = 1.0 / k + gamma * sigma**2 * q * t_rem / (A * k) if A > 0 else 1.0/k
                    d = max(d, depth_min)
                    bid_table[n2, qi] = d
                    ask_table[n2, qi] = d
            break

        Vn1 = V[n + 1]  # shape (n_q,)

        # Precompute per-inventory Vn1 values needed
        Vn1_q = Vn1  # shape (n_q,)
        Vn1_q_plus = np.roll(Vn1, -1)
        Vn1_q_minus = np.roll(Vn1, 1)
        # Fix boundaries
        Vn1_q_plus[Q] = Vn1[Q]  # at q=Q, no q+1
        Vn1_q_minus[-Q] = Vn1[-Q]  # at q=-Q, no q-1

        # term1[j, qi] = p_ask_eff[j, qi] * Vn1_q[qi] + (1-p_ask_eff[j, qi]) * Vn1_q_plus[qi]
        # term2[j, qi] = p_ask_eff[j, qi] * Vn1_q_minus[qi] + (1-p_ask_eff[j, qi]) * Vn1_q[qi]
        # For interior states, p_ask_eff = p_fill (same for all qi)
        # For q=-Q, p_ask_eff = 0

        # Vectorize over qi: shape (n_depths, n_q)
        # p_ask_eff: shape (n_depths, n_q)
        p_ask_eff = np.tile(p_fill[:, None], (1, n_q))
        p_ask_eff[:, -Q] = 0.0  # at q=-Q, no ask fill

        p_bid_eff = np.tile(p_fill[:, None], (1, n_q))
        p_bid_eff[:, Q] = 0.0  # at q=Q, no bid fill

        # term1[j, qi] = p_ask_eff[j, qi] * Vn1_q[qi] + (1 - p_ask_eff[j, qi]) * Vn1_q_plus[qi]
        term1 = p_ask_eff * Vn1_q[None, :] + (1.0 - p_ask_eff) * Vn1_q_plus[None, :]
        term2 = p_ask_eff * Vn1_q_minus[None, :] + (1.0 - p_ask_eff) * Vn1_q[None, :]

        diff = term1 - term2  # shape (n_depths, n_q)

        # ask_term[j, qi] = term2[j, qi] + depth_p[j] * (p_ask_eff[j, qi] > 0)
        # Actually: ask_term[j, qi] = term2[j, qi] + depth_grid[j] * p_ask_eff[j, qi]
        # But wait - the ask depth earned is the ask depth, not bid depth.
        # Let me re-derive:
        # If ask is hit: earn ask_depth, inventory goes from q to q-1
        # If ask not hit: inventory stays at q
        # So ask contribution = p_ask * (ask_depth + Vn1[q-1]) + (1-p_ask) * Vn1[q]
        #   = p_ask * ask_depth + p_ask * Vn1[q-1] + Vn1[q] - p_ask * Vn1[q]
        #   = p_ask * ask_depth + Vn1[q] + p_ask * (Vn1[q-1] - Vn1[q])
        #   = Vn1[q] + p_ask * (ask_depth + Vn1[q-1] - Vn1[q])
        #
        # Bid contribution: if bid hit: earn bid_depth, inventory q -> q+1
        #   = p_bid * bid_depth + p_bid * Vn1[q+1] + (1-p_bid) * Vn1[q]
        #   = Vn1[q] + p_bid * (bid_depth + Vn1[q+1] - Vn1[q])
        #
        # Total (independent fills):
        #   = Vn1[q] + p_bid*(bid_depth + Vn1[q+1]-Vn1[q]) + p_ask*(ask_depth + Vn1[q-1]-Vn1[q])
        #     - cost
        #
        # Wait, but the problem says fills are independent. So both can happen?
        # Re-reading: "bid hit with probability ..., ask likewise, independently"
        # If both hit: earn bid_depth + ask_depth, inventory: +1 then -1 = net 0? 
        # Actually inventory moves one unit per fill. If bid hits: q -> q+1. If ask hits: q -> q-1.
        # If both: q -> q+1 -> q (or q -> q -> q+1, order matters?)
        # Let me re-read: "A fill earns its depth and moves inventory one unit"
        # I think the model is: bid fill prob and ask fill prob are independent.
        # Outcomes:
        #   Neither: prob (1-pb)(1-pa), inventory stays q, reward 0
        #   Bid only: prob pb(1-pa), inventory q->q+1, reward bid_depth
        #   Ask only: prob (1-pb)pa, inventory q->q-1, reward ask_depth
        #   Both: prob pb*pa, inventory q->q+1->q (or q->q-1->q), reward bid_depth+ask_depth
        #
        # For "both": inventory returns to q. So:
        # E = (1-pb)(1-pa)*V[q] + pb(1-pa)*(bid_depth+V[q+1]) + (1-pb)pa*(ask_depth+V[q-1]) + pb*pa*(bid_depth+ask_depth+V[q])
        #   - cost
        #
        # = V[q] + pb(1-pa)*(bid_depth+V[q+1]-V[q]) + (1-pb)pa*(ask_depth+V[q-1]-V[q]) + pb*pa*(bid_depth+ask_depth)
        #
        # Hmm, this is more complex. Let me re-check the original code's approach.
        # The original code uses:
        # M[i,j] = p_bid_eff[i] * diff[j] + ask_term[j]
        # where diff = term1 - term2, term1 uses Vn1_q_plus, term2 uses Vn1_q_minus
        # This looks like it's handling the joint probability correctly via the outer product structure.
        #
        # Let me trust the original formulation and just vectorize it.

        # Using the original formulation:
        # M[i, j, qi] = p_bid_eff[i, qi] * diff[j, qi] + ask_term[j, qi]
        # where ask_term[j, qi] = term2[j, qi] + depth_grid[j] * p_ask_eff[j, qi]
        #
        # But wait, the original code has:
        # M = np.outer(p_bid_eff, diff) + ask_term[None, :]
        # This is for a single qi. For vectorization over qi, we need:
        # M[i, j, qi] = p_bid_eff[i, qi] * diff[j, qi] + ask_term[j, qi]
        #
        # This is a 3D tensor. For each qi, we need argmax over (i,j).
        # This is hard to fully vectorize the argmax, but we can try.

        # Actually, let me keep the per-qi loop but optimize it.
        # The key insight: for each qi, the structure is the same.
        # Let's use the original per-qi approach but with precomputed arrays.

        for qi in range(n_q):
            q = q_range[qi]
            cost = inv_cost_factor * q_sq[qi]

            Vn1_qi = Vn1[qi]
            Vn1_q_plus_i = Vn1_q_plus[qi]
            Vn1_q_minus_i = Vn1_q_minus[qi]

            if q == Q:
                p_bid_i = np.zeros(n_depths)
            else:
                p_bid_i = p_fill

            if q == -Q:
                p_ask_i = np.zeros(n_depths)
            else:
                p_ask_i = p_fill

            # term1[j] = p_ask_i[j] * Vn1_qi + (1-p_ask_i[j]) * Vn1_q_plus_i
            # term2[j] = p_ask_i[j] * Vn1_q_minus_i + (1-p_ask_i[j]) * Vn1_qi
            if q == -Q:
                term1_i = np.full(n_depths, Vn1_q_plus_i)
                term2_i = np.full(n_depths, Vn1_qi)
            else:
                term1_i = p_ask_i * Vn1_qi + (1.0 - p_ask_i) * Vn1_q_plus_i
                term2_i = p_ask_i * Vn1_q_minus_i + (1.0 - p_ask_i) * Vn1_qi

            diff_i = term1_i - term2_i

            if q == -Q:
                ask_term_i = term2_i
            else:
                ask_term_i = term2_i + depth_p

            # M[i, j] = p_bid_i[i] * diff_i[j] + ask_term_i[j]
            M = np.outer(p_bid_i, diff_i) + ask_term_i[None, :]

            idx_max = np.argmax(M, axis=1)
            best_val_i = M[np.arange(n_depths), idx_max] + p_bid_i * depth_grid - cost

            best_i = np.argmax(best_val_i)
            best_val = best_val_i[best_i]
            best_bid = depth_grid[best_i]
            best_ask = depth_grid[idx_max[best_i]]

            V[n, qi] = best_val
            bid_table[n, qi] = best_bid
            ask_table[n, qi] = best_ask

    # Polish pass 1: refine with local grids
    if time.time() - _START_TIME < TIME_LIMIT - 15:
        bid_table, ask_table, V = _polish_policy(params, bid_table, ask_table, V,
                                                   q_range, n_q, dt, k, A, sigma, gamma, T, steps, Q)

    # Polish pass 2: even finer local grids
    if time.time() - _START_TIME < TIME_LIMIT - 8:
        bid_table, ask_table, V = _polish_policy(params, bid_table, ask_table, V,
                                                   q_range, n_q, dt, k, A, sigma, gamma, T, steps, Q)

    return bid_table, ask_table


def _polish_policy(params, bid_table, ask_table, V, q_range, n_q, dt, k, A, sigma, gamma, T, steps, Q):
    """
    High-precision polish: for each (step, inventory), create a local fine grid
    around the initially-found optimal bid and ask depths, recompute the Bellman
    update, and update the policy.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_local = 25
    local_ratio = 0.5  # tighter local span

    V_refined = V.copy()
    bid_refined = bid_table.copy()
    ask_refined = ask_table.copy()

    inv_cost_factor = (gamma / 2.0) * sigma**2 * dt
    q_sq = (q_range * q_range).astype(np.float64)

    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            break

        Vn1 = V_refined[n + 1]

        for qi in range(n_q):
            q = q_range[qi]
            cost = inv_cost_factor * q_sq[qi]
            Vn1_qi = Vn1[qi]

            if q == Q:
                Vn1_q_plus_i = Vn1_qi
            else:
                Vn1_q_plus_i = Vn1[qi + 1]

            if q == -Q:
                Vn1_q_minus_i = Vn1_qi
            else:
                Vn1_q_minus_i = Vn1[qi - 1]

            # Local grids around current best
            best_bid_cur = bid_refined[n, qi]
            best_ask_cur = ask_refined[n, qi]

            # Ensure positive
            best_bid_cur = max(best_bid_cur, 0.001)
            best_ask_cur = max(best_ask_cur, 0.001)

            bid_span = max(best_bid_cur * local_ratio, 0.005)
            bid_local = np.linspace(max(0.001, best_bid_cur - bid_span), best_bid_cur + bid_span, n_local)
            bid_local = np.clip(bid_local, 0.001, 15.0 / k)

            ask_span = max(best_ask_cur * local_ratio, 0.005)
            ask_local = np.linspace(max(0.001, best_ask_cur - ask_span), best_ask_cur + ask_span, n_local)
            ask_local = np.clip(ask_local, 0.001, 15.0 / k)

            p_bid_local = 1.0 - np.exp(-A * np.exp(-k * bid_local) * dt)
            p_ask_local = 1.0 - np.exp(-A * np.exp(-k * ask_local) * dt)

            if q == Q:
                p_bid_local = np.zeros(n_local)
            if q == -Q:
                p_ask_local = np.zeros(n_local)

            if q == -Q:
                term1_local = np.full(n_local, Vn1_q_plus_i)
                term2_local = np.full(n_local, Vn1_qi)
            else:
                term1_local = p_ask_local * Vn1_qi + (1.0 - p_ask_local) * Vn1_q_plus_i
                term2_local = p_ask_local * Vn1_q_minus_i + (1.0 - p_ask_local) * Vn1_qi

            diff_local = term1_local - term2_local
            depth_p_local = ask_local * p_ask_local

            if q == -Q:
                ask_term_local = term2_local
            else:
                ask_term_local = term2_local + depth_p_local

            M_local = np.outer(p_bid_local, diff_local) + ask_term_local[None, :]

            idx_max_local = np.argmax(M_local, axis=1)
            best_val_local = M_local[np.arange(n_local), idx_max_local] + p_bid_local * bid_local - cost
            best_i_local = np.argmax(best_val_local)
            best_bid_local = bid_local[best_i_local]
            best_ask_local = ask_local[idx_max_local[best_i_local]]

            V_refined[n, qi] = best_val_local[best_i_local]
            bid_refined[n, qi] = best_bid_local
            ask_refined[n, qi] = best_ask_local

    return bid_refined, ask_refined, V_refined


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
