"""Refined greedy knapsack heuristic with forward-simulation look-ahead and a polish pass."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0 - 20.0  # safety margin (s)
_T0 = time.monotonic()


def _sorted_indices(weights, values):
    """Return indices sorted by value density descending, then value descending."""
    n = len(weights)
    if n == 0:
        return np.array([], dtype=int)
    dens = values / weights
    # stable sort: primary density desc, secondary value desc
    order = np.lexsort((-values, -dens))
    return order


def _greedy_fill_from(order, used, capacity, weights, values):
    """Walk the pre-sorted density order, skipping used items, and greedily fill.
    Returns (total_value, chosen_mask). O(n) per call."""
    total = 0.0
    cap = capacity
    chosen = np.zeros(len(weights), dtype=bool)
    for i in order:
        if used[i]:
            continue
        w = weights[i]
        if w <= cap:
            total += values[i]
            cap -= w
            chosen[i] = True
            if cap <= 0:
                break
    return total, chosen


def _polish_swap(chosen, weights, values, capacity):
    """Try to improve a chosen set by swapping one chosen item for one unchosen item.
    Returns improved chosen mask (or original if no improvement)."""
    n = len(weights)
    chosen = chosen.copy()
    chosen_value = float(values[chosen].sum())
    chosen_weight = float(weights[chosen].sum())
    improved = False
    # For each chosen item, find the best unchosen item that fits after removing it
    for i in range(n):
        if not chosen[i]:
            continue
        free = capacity - (chosen_weight - weights[i])
        if free <= 0:
            continue
        # best unchosen item with weight <= free
        best_j = -1
        best_v = -1.0
        for j in range(n):
            if chosen[j]:
                continue
            if weights[j] <= free and values[j] > best_v:
                best_v = values[j]
                best_j = j
        if best_j >= 0 and best_v > values[i]:
            # swap improves value
            chosen[i] = False
            chosen[best_j] = True
            chosen_weight = chosen_weight - weights[i] + weights[best_j]
            chosen_value = chosen_value - values[i] + best_v
            improved = True
    return chosen, improved


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Pick the next item to add (index into the feasible arrays) or -1 to stop."""
    if time.monotonic() - _T0 > _DEADLINE:
        return -1

    remaining_weights = np.asarray(remaining_weights, dtype=float)
    remaining_values = np.asarray(remaining_values, dtype=float)
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    cap = float(remaining_capacity)
    feasible = remaining_weights <= cap
    if not np.any(feasible):
        return -1

    # Precompute the global density order for the current candidate set
    order = _sorted_indices(remaining_weights, remaining_values)

    # Baseline: pure greedy (no look-ahead) as a fallback
    base_total, base_chosen = _greedy_fill_from(order, np.zeros(n, dtype=bool), cap,
                                                remaining_weights, remaining_values)

    # Forward-simulation look-ahead over feasible candidates
    best_total = -1.0
    best_idx = -1
    best_chosen = None
    feasible_idxs = np.where(feasible)[0]

    for cand in feasible_idxs:
        used = np.zeros(n, dtype=bool)
        used[cand] = True
        total, chosen = _greedy_fill_from(order, used, cap - remaining_weights[cand],
                                          remaining_weights, remaining_values)
        if total > best_total:
            best_total = total
            best_idx = int(cand)
            best_chosen = chosen
        if time.monotonic() - _T0 > _DEADLINE:
            break

    if best_idx < 0:
        # Fallback to pure greedy
        return int(order[0])

    # Polish pass: try a single swap improvement on the chosen set
    if best_chosen is not None:
        polished, _ = _polish_swap(best_chosen, remaining_weights, remaining_values, cap)
        # If the polish changed the set, ensure best_idx is still in the polished set;
        # if the polish removed best_idx, fall back to the first item of the polished set.
        if not polished[best_idx]:
            # pick the first (highest-density) item in the polished set
            poly_idxs = np.where(polished)[0]
            if len(poly_idxs) == 0:
                return -1
            # choose the one with highest density among polished
            dens = remaining_values[poly_idxs] / remaining_weights[poly_idxs]
            best_idx = int(poly_idxs[np.argmax(dens)])

    return int(best_idx)
# EVOLVE-BLOCK-END
