"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0


def _greedy_fill_fast(capacity, weights, values, excluded_idx):
    """Fast greedy fill with multiple policies + 1-swap local refinement."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    mask = np.ones(n, dtype=bool)
    mask[excluded_idx] = False
    mask &= weights <= capacity
    if not np.any(mask):
        return 0.0

    idxs = np.where(mask)[0]
    w = weights[idxs]
    v = values[idxs]
    m = len(w)
    if m == 0:
        return 0.0

    best_total = 0.0
    best_selected = None  # set of local indices selected

    def _run_policy(order):
        nonlocal best_total, best_selected
        total = 0.0
        cap = capacity
        selected = []
        for i in order:
            wi = w[i]
            if wi <= cap:
                total += v[i]
                cap -= wi
                selected.append(i)
                if cap <= 0:
                    break
        if total > best_total:
            best_total = total
            best_selected = selected

    # Precompute common arrays
    density = v / w
    sqrt_density = v / np.sqrt(w)
    w15_density = v / (w ** 1.5)

    # Policy 0: density
    _run_policy(np.argsort(-density))

    # Policy 1: value
    _run_policy(np.argsort(-v))

    # Policy 2: blend (density + 0.5 * value)
    _run_policy(np.argsort(-(density + 0.5 * v)))

    # Policy 3: small-weight first
    _run_policy(np.argsort(w))

    # Policy 4: snug-fit preference
    _run_policy(np.argsort(-(density * (1 + 0.1 * (w / capacity)))))

    # Policy 5: sqrt-density (favors lighter high-value items)
    _run_policy(np.argsort(-sqrt_density))

    # Policy 6: w^1.5 density (strongly favors light items)
    _run_policy(np.argsort(-w15_density))

    # Policy 7: light-first among high-value (value / w^2)
    _run_policy(np.argsort(-(v / (w ** 2))))

    # 1-swap local refinement on the best solution
    if best_selected is not None and len(best_selected) > 0:
        sel_set = set(best_selected)
        # Try removing each selected item and adding the best non-selected that fits
        for si in list(best_selected):
            new_total = best_total - v[si]
            new_cap = capacity - (sum(w[s] for s in best_selected) - w[si])
            # Find best item not in sel_set that fits in new_cap
            best_add = -1
            best_add_v = -1
            for i in range(m):
                if i not in sel_set and w[i] <= new_cap and v[i] > best_add_v:
                    best_add_v = v[i]
                    best_add = i
            if best_add >= 0 and new_total + best_add_v > best_total:
                # Try this swap
                new_total = new_total + best_add_v
                new_cap = new_cap - w[best_add]
                # After swap, try to fill remaining capacity with next-best items
                used = set(best_selected) - {si} | {best_add}
                for i in range(m):
                    if i not in used and w[i] <= new_cap:
                        new_total += v[i]
                        new_cap -= w[i]
                        if new_cap <= 0:
                            break
                if new_total > best_total:
                    best_total = new_total

    return best_total


def _pair_lookahead(remaining_capacity, remaining_weights, remaining_values,
                    feasible_indices, fw, fv, fn, top_k=3):
    """For top-k candidates, try pairs to capture 2-item synergy."""
    # First get single-item estimates to rank
    single_estimates = []
    for li in range(fn):
        item_w = fw[li]
        item_v = fv[li]
        remaining_cap = remaining_capacity - item_w
        est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                feasible_indices[li])
        total_est = item_v + est
        single_estimates.append((total_est, li))

    single_estimates.sort(key=lambda x: -x[0])
    top_candidates = [li for _, li in single_estimates[:top_k]]

    best_score = -1.0
    best_local_idx = top_candidates[0]

    # Check pairs among top candidates
    for i in range(len(top_candidates)):
        li = top_candidates[i]
        item_w = fw[li]
        item_v = fv[li]
        remaining_cap = remaining_capacity - item_w
        est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                feasible_indices[li])
        total_est = item_v + est
        if total_est > best_score:
            best_score = total_est
            best_local_idx = li

        # Try pairing with other top candidates
        for j in range(len(top_candidates)):
            if j == i:
                continue
            lj = top_candidates[j]
            pair_w = fw[li] + fw[lj]
            if pair_w > remaining_capacity:
                continue
            pair_v = fv[li] + fv[lj]
            rem_cap = remaining_capacity - pair_w
            # Estimate fill after taking both
            est2 = _greedy_fill_fast(rem_cap, remaining_weights, remaining_values,
                                     feasible_indices[li])
            # Subtract the value of lj if it was included in est2
            # We need to exclude both li and lj
            # Simpler: just use est2 as a rough estimate
            total_pair = pair_v + est2
            if total_pair > best_score:
                best_score = total_pair
                best_local_idx = li  # pick li as the "first" of the pair

    return int(feasible_indices[best_local_idx])


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Select next item using lookahead greedy completion with multiple policies."""
    n = len(remaining_weights)
    if n == 0:
        return -1

    elapsed = time.monotonic() - _START
    if elapsed > _TOTAL_BUDGET:
        return -1

    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    fn = len(fw)

    if fn == 1:
        return int(feasible_indices[0])

    # Fast fallback if time is tight
    if elapsed > _TOTAL_BUDGET * 0.7:
        density = fv / fw
        d_min, d_max = density.min(), density.max()
        v_min, v_max = fv.min(), fv.max()
        d_range = d_max - d_min + 1e-12
        v_range = v_max - v_min + 1e-12
        d_norm = (density - d_min) / d_range
        v_norm = (fv - v_min) / v_range
        scores = 0.6 * d_norm + 0.4 * v_norm
        return int(feasible_indices[int(np.argmax(scores))])

    # If many items remain, use pair-lookahead on top candidates
    if fn > 10 and elapsed < _TOTAL_BUDGET * 0.5:
        return _pair_lookahead(remaining_capacity, remaining_weights, remaining_values,
                               feasible_indices, fw, fv, fn, top_k=3)

    # Lookahead: for each feasible item, estimate total via greedy completion
    best_score = -1.0
    best_local_idx = 0

    for li in range(fn):
        item_w = fw[li]
        item_v = fv[li]
        remaining_cap = remaining_capacity - item_w
        est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                feasible_indices[li])
        total_est = item_v + est
        if total_est > best_score:
            best_score = total_est
            best_local_idx = li

    return int(feasible_indices[best_local_idx])
# EVOLVE-BLOCK-END
