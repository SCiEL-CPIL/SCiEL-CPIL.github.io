"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0


def _greedy_fill_fast(capacity, weights, values, excluded_idx):
    """Fast greedy fill with multiple policies, returns best total value."""
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

    best_total = 0.0

    def _run_policy(order):
        total = 0.0
        cap = capacity
        for i in order:
            wi = w[i]
            if wi <= cap:
                total += v[i]
                cap -= wi
                if cap <= 0:
                    break
        return total

    # Policy 0: density
    order = np.argsort(-(v / w))
    t = _run_policy(order)
    if t > best_total:
        best_total = t

    # Policy 1: value
    order = np.argsort(-v)
    t = _run_policy(order)
    if t > best_total:
        best_total = t

    # Policy 2: blend (density + 0.5 * value)
    order = np.argsort(-(v / w + 0.5 * v))
    t = _run_policy(order)
    if t > best_total:
        best_total = t

    # Policy 3: small-weight first (to pack more items)
    order = np.argsort(w)
    t = _run_policy(order)
    if t > best_total:
        best_total = t

    # Policy 4: high value-to-weight with preference for items that fit snugly
    scores = (v / w) * (1 + 0.1 * (w / capacity))
    order = np.argsort(-scores)
    t = _run_policy(order)
    if t > best_total:
        best_total = t

    # Policy 5: normalized density (value / weight^0.5) — favors lighter high-value items
    order = np.argsort(-(v / np.sqrt(w)))
    t = _run_policy(order)
    if t > best_total:
        best_total = t

    # Policy 6: value / weight^1.5 — strong preference for light items
    order = np.argsort(-(v / (w * np.sqrt(w))))
    t = _run_policy(order)
    if t > best_total:
        best_total = t

    # Policy 7: maximize value subject to weight <= 0.3*capacity first, then density
    light_mask = w <= 0.3 * capacity
    if np.any(light_mask):
        light_order = np.where(light_mask)[0]
        light_order = light_order[np.argsort(-(v[light_order] / w[light_order]))]
        heavy_order = np.where(~light_mask)[0]
        heavy_order = heavy_order[np.argsort(-(v[heavy_order] / w[heavy_order]))]
        combined = np.concatenate([light_order, heavy_order])
        t = _run_policy(combined)
        if t > best_total:
            best_total = t

    # Local swap refinement: try to improve best solution by 1-swap
    # Rebuild the best solution's item set
    best_order = np.argsort(-(v / w))  # default, we'll rebuild below
    # Actually, let's just do a simple 1-swap on the density-greedy result
    # Rebuild density greedy to get the selected set
    selected = set()
    cap = capacity
    for i in best_order:
        wi = w[i]
        if wi <= cap:
            selected.add(i)
            cap -= wi
            if cap <= 0:
                break

    if selected:
        sel_list = list(selected)
        # Try removing one selected item and adding one unselected item
        unselected = set(range(len(w))) - selected
        if unselected:
            unselected_list = list(unselected)
            for si in sel_list:
                freed = w[si]
                for ui in unselected_list:
                    if w[ui] <= freed + 1e-12:
                        # Check: removing si and adding ui
                        # Total weight change: -w[si] + w[ui]
                        # Total value change: -v[si] + v[ui]
                        if v[ui] > v[si]:
                            new_cap = capacity - (sum(w[s] for s in selected) - w[si] + w[ui])
                            if new_cap >= -1e-12:
                                new_val = sum(v[s] for s in selected) - v[si] + v[ui]
                                if new_val > best_total:
                                    best_total = new_val

    return best_total


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

    # Lookahead: for each feasible item, estimate total via greedy completion
    best_score = -1.0
    best_local_idx = 0
    scores_arr = np.zeros(fn)

    for li in range(fn):
        item_w = fw[li]
        item_v = fv[li]
        remaining_cap = remaining_capacity - item_w
        est = _greedy_fill_fast(remaining_cap, remaining_weights, remaining_values,
                                feasible_indices[li])
        total_est = item_v + est
        scores_arr[li] = total_est
        if total_est > best_score:
            best_score = total_est
            best_local_idx = li

    # Pair-lookahead: for the top-3 candidates, also try picking a second item
    # that fills the remaining gap well, to break ties or find better combos
    if fn >= 2:
        top3 = np.argsort(-scores_arr)[:min(3, fn)]
        for li in top3:
            item_w = fw[li]
            item_v = fv[li]
            remaining_cap = remaining_capacity - item_w
            if remaining_cap <= 0:
                continue
            # Find best second item that fits in remaining_cap
            second_mask = np.ones(n, dtype=bool)
            second_mask[feasible_indices[li]] = False
            second_mask &= remaining_weights <= remaining_cap
            if not np.any(second_mask):
                continue
            second_idxs = np.where(second_mask)[0]
            sw = remaining_weights[second_idxs]
            sv = remaining_values[second_idxs]
            if len(sw) > 0:
                best_second_local = int(np.argmax(sv / sw))
                best_second = second_idxs[best_second_local]
                pair_remaining_cap = remaining_cap - sw[best_second_local]
                est2 = _greedy_fill_fast(pair_remaining_cap, remaining_weights,
                                         remaining_values, best_second)
                total_pair = item_v + sv[best_second_local] + est2
                if total_pair > best_score:
                    best_score = total_pair
                    best_local_idx = li

    return int(feasible_indices[best_local_idx])
# EVOLVE-BLOCK-END
