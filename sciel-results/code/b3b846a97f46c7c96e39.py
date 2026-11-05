import time
import sys
from fractions import Fraction
import bisect
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # Split into two halves for meet-in-the-middle
    nA = 23
    nB = n - nA

    # Generate all subset sums for each half, pruning sums >= 1.0
    def generate_pruned_sums(vals):
        """Generate all subset sums < 1.0 with their bitmasks."""
        sums = [0.0]
        masks = [0]
        for i, v in enumerate(vals):
            bit = 1 << i
            current_len = len(sums)
            for k in range(current_len):
                new_s = sums[k] + v
                if new_s < 1.0:
                    sums.append(new_s)
                    masks.append(masks[k] | bit)
            # Time check
            if time.time() - start_time > DEADLINE - 5:
                return None, None
        return np.array(sums, dtype=np.float64), np.array(masks, dtype=np.int64)

    valsA = [1.0 / d for d in denominators[:nA]]
    valsB = [1.0 / d for d in denominators[nA:]]

    sumsA, masksA = generate_pruned_sums(valsA)
    if sumsA is None:
        # Fallback: greedy
        return _greedy_fallback(denominators)

    sumsB, masksB = generate_pruned_sums(valsB)
    if sumsB is None:
        return _greedy_fallback(denominators)

    # Sort B sums for binary search
    orderB = np.argsort(sumsB)
    sumsB_sorted = sumsB[orderB]
    masksB_sorted = masksB[orderB]

    # For each sum in A, find the largest B sum such that sumA + sumB < 1.0
    # Use searchsorted: find position where target would be inserted (side='left')
    # Then the valid index is pos - 1
    targets = 1.0 - sumsA
    validA = targets > 0  # sumA < 1.0 (should all be true due to pruning)

    if not np.any(validA):
        return _greedy_fallback(denominators)

    sA_valid = sumsA[validA]
    mA_valid = masksA[validA]
    targets_valid = targets[validA]

    pos = np.searchsorted(sumsB_sorted, targets_valid, side='left')
    valid_pos = pos > 0

    if not np.any(valid_pos):
        return _greedy_fallback(denominators)

    pos_valid = pos[valid_pos]
    sB_indices = pos_valid - 1

    sA_cand = sA_valid[valid_pos]
    mA_cand = mA_valid[valid_pos]
    sB_cand = sumsB_sorted[sB_indices]
    mB_cand = masksB_sorted[sB_indices]

    totals = sA_cand + sB_cand
    # Double-check strict inequality (floating point safety)
    valid_total = totals < 1.0

    if not np.any(valid_total):
        return _greedy_fallback(denominators)

    totals_valid = totals[valid_total]
    sA_final = sA_cand[valid_total]
    mA_final = mA_cand[valid_total]
    sB_final = sB_cand[valid_total]
    mB_final = mB_cand[valid_total]

    deficits = 1.0 - totals_valid
    best_idx = np.argmin(deficits)

    best_maskA = int(mA_final[best_idx])
    best_maskB = int(mB_final[best_idx])

    # Reconstruct selected indices
    selected = []
    for i in range(nA):
        if best_maskA & (1 << i):
            selected.append(i)
    for i in range(nB):
        if best_maskB & (1 << i):
            selected.append(nA + i)

    if not selected:
        return _greedy_fallback(denominators)

    # Verify with exact Fraction arithmetic
    total_frac = Fraction(0)
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])

    if total_frac >= 1:
        # Floating point gave a false positive; remove the smallest contribution
        # and try local search
        return _local_search_polish(denominators, selected, start_time, DEADLINE)

    # Try to improve with local search
    if time.time() - start_time < DEADLINE - 10:
        improved = _local_search_polish(denominators, selected, start_time, DEADLINE)
        if improved:
            return improved

    return selected


def _local_search_polish(denominators, initial_selected, start_time, deadline):
    """Try to improve the solution via local search (add/remove/swap elements)."""
    n = len(denominators)
    selected = set(initial_selected)
    total = Fraction(0)
    for idx in selected:
        total += Fraction(1, denominators[idx])

    if total >= 1:
        # Need to fix: remove elements until valid
        # Remove the one with largest 1/d first
        while total >= 1 and selected:
            # Find element with largest reciprocal
            max_elem = max(selected, key=lambda i: Fraction(1, denominators[i]))
            total -= Fraction(1, denominators[max_elem])
            selected.remove(max_elem)
        if not selected:
            return _greedy_fallback(denominators)

    best_total = total
    best_selected = set(selected)
    available = set(range(n)) - selected

    # Local search: try adding one element, removing one, or swapping
    improved = True
    while improved and time.time() - start_time < deadline - 5:
        improved = False

        # Try adding each available element
        for idx in sorted(available, key=lambda i: -1.0 / denominators[i]):
            new_total = best_total + Fraction(1, denominators[idx])
            if new_total < 1 and new_total > best_total:
                best_total = new_total
                best_selected = best_selected | {idx}
                available.discard(idx)
                improved = True
                break

        # If no add helped, try removing one to make room for a better addition
        if not improved:
            # Try each removal
            for idx in sorted(best_selected, key=lambda i: 1.0 / denominators[i]):
                new_total = best_total - Fraction(1, denominators[idx])
                if new_total < 1:
                    # Now try to add the best available
                    for add_idx in sorted(available, key=lambda i: -1.0 / denominators[i]):
                        candidate_total = new_total + Fraction(1, denominators[add_idx])
                        if candidate_total < 1 and candidate_total > best_total:
                            best_total = candidate_total
                            best_selected = (best_selected - {idx}) | {add_idx}
                            available = available - {add_idx} | {idx}
                            improved = True
                            break
                if improved:
                    break

        # Also try single swaps
        if not improved:
            for rem_idx in sorted(best_selected, key=lambda i: 1.0 / denominators[i]):
                for add_idx in sorted(available, key=lambda i: -1.0 / denominators[i]):
                    new_total = best_total - Fraction(1, denominators[rem_idx]) + Fraction(1, denominators[add_idx])
                    if new_total < 1 and new_total > best_total:
                        best_total = new_total
                        best_selected = (best_selected - {rem_idx}) | {add_idx}
                        available = available - {add_idx} | {rem_idx}
                        improved = True
                        break
                if improved:
                    break

    result = sorted(best_selected)
    if not result:
        return _greedy_fallback(denominators)

    # Final verification
    final_total = Fraction(0)
    for idx in result:
        final_total += Fraction(1, denominators[idx])
    if final_total >= 1:
        return _greedy_fallback(denominators)

    return result


def _greedy_fallback(denominators):
    """Simple greedy fallback: add denominators in order if sum stays < 1."""
    selected = []
    total = Fraction(0)
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
