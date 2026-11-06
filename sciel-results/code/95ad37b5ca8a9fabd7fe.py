"""0/1 knapsack heuristic: look-ahead greedy with a density-fill surrogate."""
import time
import numpy as np

# Hard wall-clock budget (evaluator kills at 150s; use a safety margin).
_DEADLINE = 140.0
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _density_fill(capacity, weights, values):
    """Greedy fill by value density (fractional-ish ordering, 0/1 picks).
    Returns the total value obtainable by repeatedly taking the highest
    value/weight item that still fits. O(n log n)."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    # Only consider items that can fit at least once
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    feasible = w <= capacity
    if not np.any(feasible):
        return 0.0
    w = w[feasible]
    v = v[feasible]
    order = np.argsort(v / w)[::-1]
    cap = capacity
    total = 0.0
    for idx in order:
        wi = w[idx]
        if wi <= cap:
            cap -= wi
            total += v[idx]
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline guard: if we're out of time, stop cleanly.
    if _elapsed() > _DEADLINE:
        return -1

    remaining_weights = np.asarray(remaining_weights, dtype=float)
    remaining_values = np.asarray(remaining_values, dtype=float)
    n = len(remaining_weights)
    if n == 0:
        return -1

    # Feasibility: item must fit in current capacity.
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_idx = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_idx]
    fv = remaining_values[feasible_idx]

    # For each feasible candidate, estimate the total value we'd get:
    #   value of this item + greedy density-fill of the leftover capacity
    # using the *other* remaining feasible items.
    # To keep this fast, we approximate the leftover fill using the full
    # feasible set (a slight overestimate, but consistent across candidates
    # since the removed item is small relative to the set).
    # We compute the full-set density fill once, then for each candidate we
    # subtract its contribution and add its own value. This is a smooth
    # surrogate for "best value if I commit to this pick now."

    # Full-set density fill of the current capacity (baseline).
    # We'll compute per-candidate: candidate_value + fill(cap - w_i) over others.
    # Approximation: fill(cap - w_i) over others ~= fill(cap - w_i) over all
    # feasible items (cheap and monotonic-ish). Good enough for ranking.

    best_score = -1.0
    best_local = 0
    for k, i in enumerate(feasible_idx):
        if _elapsed() > _DEADLINE - 1.0:
            break
        wi = fw[k]
        vi = fv[k]
        leftover_cap = remaining_capacity - wi
        if leftover_cap <= 0:
            est = vi
        else:
            est = vi + _density_fill(leftover_cap, fw, fv)
        if est > best_score:
            best_score = est
            best_local = k

    return int(feasible_idx[best_local])
