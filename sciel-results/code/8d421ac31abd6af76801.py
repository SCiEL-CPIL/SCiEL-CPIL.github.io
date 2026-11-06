"""Bounded lookahead 0/1 knapsack heuristic with Lagrangian-guided scoring."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.monotonic()
_TOTAL_BUDGET = 115.0


def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))


def _greedy_complete(w, v, cap, policy='density'):
    """Simulate greedy completion. Returns total added value."""
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0
    if policy == 'density':
        order = np.argsort(-v / np.maximum(w, 1e-12))
    elif policy == 'value':
        order = np.argsort(-v)
    elif policy == 'blend':
        order = np.argsort(-(v / np.maximum(w, 1e-12) + 0.15 * v))
    else:
        order = np.argsort(-v / np.maximum(w, 1e-12))

    rem = float(cap)
    total = 0.0
    for i in order:
        wi = float(w[i])
        if wi <= rem:
            rem -= wi
            total += float(v[i])
            if rem <= 0:
                break
    return total


def _lagrangian_lambda(w, v, cap):
    """
    Estimate the Lagrangian multiplier lambda such that the continuous
    relaxation (take all items with v_i - lambda*w_i > 0) approximately
    exhausts capacity. We use binary search on lambda.
    """
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0

    # Bounds for lambda: 0 to max(v/w)
    densities = v / np.maximum(w, 1e-12)
    lam_lo = 0.0
    lam_hi = float(np.max(densities)) + 1e-9

    # If even at lambda=0 the total weight exceeds capacity, we need positive lambda
    total_w = float(np.sum(w))
    if total_w <= cap:
        return 0.0

    # Binary search for lambda where sum of weights of items with positive reduced cost ≈ cap
    for _ in range(40):
        lam_mid = (lam_lo + lam_hi) / 2.0
        # Items with v_i - lambda * w_i > 0 are "taken" in the relaxation
        reduced_costs = v - lam_mid * w
        taken = reduced_costs > 0
        weight_taken = float(np.sum(w[taken]))

        if weight_taken > cap:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    return (lam_lo + lam_hi) / 2.0


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item to add using Lagrangian-guided scoring with lookahead.
    
    Strategy:
    1. Compute lambda from the continuous relaxation of the current state.
    2. Score each feasible item by its Lagrangian objective: v_i - lambda * w_i.
    3. For top candidates, do a full lookahead with greedy completion to verify.
    4. Return the best candidate.
    """
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)
    n = len(w)

    if n == 0 or cap <= 0:
        return -1

    # Only consider feasible items (weight <= capacity)
    feasible = w <= cap
    if not np.any(feasible):
        return -1

    feasible_indices = np.where(feasible)[0]
    fw = w[feasible]
    fv = v[feasible]
    n_feasible = len(fw)

    # Check time budget
    if _remaining_budget() < 0.5:
        # Fallback to simple density
        density = fv / np.maximum(fw, 1e-12)
        best_local = int(np.argmax(density))
        return int(feasible_indices[best_local])

    # Compute Lagrangian lambda for the current state
    lam = _lagrangian_lambda(w, v, cap)

    # Score all feasible items by Lagrangian objective
    lag_scores = fv - lam * fw

    # Select top candidates for lookahead verification
    # For small n, evaluate all; for larger n, sample top candidates
    if n_feasible <= 30:
        candidates = feasible_indices
    else:
        # Take top 30 by Lagrangian score
        top_k = min(30, n_feasible)
        top_local = np.argsort(-lag_scores)[:top_k]
        candidates = feasible_indices[top_local]

    best_score = -1.0
    best_candidate = None

    for ci in candidates:
        wi = float(w[ci])
        vi = float(v[ci])
        new_cap = cap - wi
        if new_cap < 0:
            continue

        # Create remaining arrays after taking this item
        mask = np.ones(n, dtype=bool)
        mask[ci] = False
        rem_w = w[mask]
        rem_v = v[mask]

        # Try multiple policies for completion
        for policy in ('density', 'value', 'blend'):
            completion_value = _greedy_complete(rem_w, rem_v, new_cap, policy)
            total_value = vi + completion_value
            if total_value > best_score:
                best_score = total_value
                best_candidate = ci

    if best_candidate is None:
        # Fallback to highest Lagrangian score feasible item
        best_local = int(np.argmax(lag_scores))
        return int(feasible_indices[best_local])

    return int(best_candidate)
# EVOLVE-BLOCK-END
