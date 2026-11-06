"""0/1 Knapsack heuristic using Lagrangian relaxation with dynamic lambda tuning."""
import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 110.0  # 10s safety margin before 120s limit


def _greedy_01_completion(capacity, weights, values):
    """Greedy 0/1 fill by value density, with 1-swap local refinement."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0

    # Sort by value density descending
    order = np.argsort(-values / weights)
    used = np.zeros(n, dtype=bool)
    total_val = 0.0
    total_wt = 0.0

    # Greedy fill
    for idx in order:
        if weights[idx] <= capacity - total_wt:
            used[idx] = True
            total_wt += weights[idx]
            total_val += values[idx]

    # 1-swap local refinement: try replacing one used item with two unused
    used_list = np.where(used)[0]
    unused_list = np.where(~used)[0]

    improved = True
    while improved:
        improved = False
        for ui in used_list:
            for uj in unused_list:
                for uk in unused_list:
                    if uj == uk:
                        continue
                    new_wt = total_wt - weights[ui] + weights[uj] + weights[uk]
                    new_val = total_val - values[ui] + values[uj] + values[uk]
                    if new_wt <= capacity and new_val > total_val + 1e-12:
                        used[ui] = False
                        used[uj] = True
                        used[uk] = True
                        total_wt = new_wt
                        total_val = new_val
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item using Lagrangian relaxation:
    - Tune lambda so fractional knapsack exhausts capacity
    - Score each feasible item by (value - lambda * weight) + lookahead bonus
    """
    global _DEADLINE

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter feasible items
    feasible = remaining_weights <= remaining_capacity
    if not np.any(feasible):
        return -1

    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback: best value density among feasible
        feasible_idx = np.where(feasible)[0]
        return int(feasible_idx[np.argmax(remaining_values[feasible_idx] / remaining_weights[feasible_idx])])

    # Tune lambda via bisection so that fractional knapsack total weight = capacity
    # Sort by value density
    densities = remaining_values / remaining_weights
    order = np.argsort(-densities)

    def frac_knapsack_weight(lam):
        """Total weight selected by fractional knapsack with threshold lambda."""
        # Items with (value - lambda * weight) > 0 are fully taken
        # The marginal item is partially taken
        scores = remaining_values - lam * remaining_weights
        # Sort by density (already have order)
        total_w = 0.0
        for idx in order:
            if scores[idx] > 0:
                total_w += remaining_weights[idx]
            elif scores[idx] == 0:
                # Partially take
                # This item is at the boundary
                total_w += remaining_weights[idx]  # approx
                break
            else:
                break
        return total_w

    # Bisection for lambda
    lam_lo = 0.0
    lam_hi = np.max(densities) * 2.0

    # If even at lam=0 we can't fill, lambda is 0
    if frac_knapsack_weight(0.0) < remaining_capacity:
        lam = 0.0
    else:
        # Bisection: find lam where frac weight = capacity
        for _ in range(30):
            lam_mid = (lam_lo + lam_hi) / 2.0
            w = frac_knapsack_weight(lam_mid)
            if w > remaining_capacity:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        lam = (lam_lo + lam_hi) / 2.0

    # Score feasible items: Lagrangian objective + greedy completion lookahead
    feasible_idx = np.where(feasible)[0]
    best_score = -np.inf
    best_idx = -1

    for idx in feasible_idx:
        wi = remaining_weights[idx]
        vi = remaining_values[idx]

        # Lagrangian score
        lag_score = vi - lam * wi

        # Lookahead: greedy 0/1 completion after picking this item
        cap_rem = remaining_capacity - wi
        if cap_rem < 0:
            continue

        # Build arrays excluding idx
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        w_rem = remaining_weights[mask]
        v_rem = remaining_values[mask]

        # Quick deadline check for expensive lookahead
        if time.time() > _DEADLINE - 5.0:
            lookahead = 0.0
        else:
            lookahead = _greedy_01_completion(cap_rem, w_rem, v_rem)

        total_score = vi + lookahead + 0.1 * lag_score

        if total_score > best_score:
            best_score = total_score
            best_idx = idx

    if best_idx == -1:
        # Fallback: best density
        return int(feasible_idx[np.argmax(densities[feasible_idx])])

    return int(best_idx)
