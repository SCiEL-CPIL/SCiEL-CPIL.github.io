"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Denominators are 2, 3, ..., 48
    # Zero-based index i corresponds to denominator d = i + 2
    # We need a nonempty subset of indices such that sum(1/(i+2)) < 1
    # and we want to MINIMIZE the deficit 1 - sum, i.e., MAXIMIZE the sum subject to sum < 1.
    
    best_sum = Fraction(0)
    best_indices = []
    
    def update_best(s, indices):
        nonlocal best_sum, best_indices
        if s < 1 and s > best_sum:
            best_sum = s
            best_indices = list(indices)
    
    def elapsed():
        return time.time() - start_time
    
    # Step 1: Greedy initial solution
    selected, total = [], Fraction()
    for i, d in enumerate(range(2, 49)):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    if selected:
        update_best(total, selected)
    
    # Step 2: Known good solution
    candidate = [0, 1, 5, 41]
    s = Fraction(1, 2) + Fraction(1, 3) + Fraction(1, 7) + Fraction(1, 43)
    update_best(s, candidate)
    
    # Step 3: Brute force over first 12 denominators (indices 0-11, denoms 2-13)
    # 2^12 = 4096 subsets, very fast
    for mask in range(1, 1 << 12):
        s = Fraction(0)
        indices = []
        for i in range(12):
            if mask & (1 << i):
                s += Fraction(1, i + 2)
                indices.append(i)
        if s < 1:
            update_best(s, indices)
        if elapsed() > DEADLINE:
            return best_indices
    
    # Step 4: Targeted search - for each subset of small denominators,
    # find the best combination of 1-2 additional fractions from larger denominators
    
    # Try all subsets of {0,1,2,3,4,5} (denoms 2-7) of size 1-3
    # combined with one fraction from {6,...,46} (denoms 8-48)
    for size in range(1, 4):
        for combo in combinations(range(6), size):
            base_sum = Fraction(0)
            for i in combo:
                base_sum += Fraction(1, i + 2)
            if base_sum >= 1:
                continue
            target = Fraction(1) - base_sum
            if target <= 0:
                continue
            best_single = None
            best_single_val = Fraction(0)
            for j in range(6, 47):
                val = Fraction(1, j + 2)
                if base_sum + val < 1:
                    if val > best_single_val:
                        best_single_val = val
                        best_single = j
            if best_single is not None:
                total_sum = base_sum + best_single_val
                indices = list(combo) + [best_single]
                update_best(total_sum, indices)
        if elapsed() > DEADLINE:
            return best_indices
    
    # Try subsets of size 2 from first 8 denominators with one additional
    for combo in combinations(range(8), 2):
        base_sum = Fraction(0)
        for i in combo:
            base_sum += Fraction(1, i + 2)
        if base_sum >= 1:
            continue
        target = Fraction(1) - base_sum
        if target <= 0:
            continue
        best_single = None
        best_single_val = Fraction(0)
        for j in range(8, 47):
            val = Fraction(1, j + 2)
            if base_sum + val < 1:
                if val > best_single_val:
                    best_single_val = val
                    best_single = j
        if best_single is not None:
            total_sum = base_sum + best_single_val
            indices = list(combo) + [best_single]
            update_best(total_sum, indices)
    
    if elapsed() > DEADLINE:
        return best_indices
    
    # Step 5: More targeted search - subsets of size 3 from first 7 with one additional
    for combo in combinations(range(7), 3):
        base_sum = Fraction(0)
        for i in combo:
            base_sum += Fraction(1, i + 2)
        if base_sum >= 1:
            continue
        target = Fraction(1) - base_sum
        if target <= 0:
            continue
        best_single = None
        best_single_val = Fraction(0)
        for j in range(7, 47):
            val = Fraction(1, j + 2)
            if base_sum + val < 1:
                if val > best_single_val:
                    best_single_val = val
                    best_single = j
        if best_single is not None:
            total_sum = base_sum + best_single_val
            indices = list(combo) + [best_single]
            update_best(total_sum, indices)
        if elapsed() > DEADLINE:
            return best_indices
    
    # Step 6: Local search - try swapping one element in best_indices with another
    if best_indices:
        current_set = set(best_indices)
        current_sum = best_sum
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            for i in list(current_set):
                # Try removing i and adding j
                for j in range(47):
                    if j in current_set:
                        continue
                    new_sum = current_sum - Fraction(1, i + 2) + Fraction(1, j + 2)
                    if new_sum < 1 and new_sum > current_sum:
                        new_set = current_set - {i} | {j}
                        new_indices = sorted(new_set)
                        update_best(new_sum, new_indices)
                        current_set = new_set
                        current_sum = new_sum
                        improved = True
                        break
                if elapsed() > DEADLINE:
                    return best_indices
    
    # Step 7: Try adding one more fraction to the current best
    if best_indices and elapsed() < DEADLINE:
        current_set = set(best_indices)
        current_sum = best_sum
        for j in range(47):
            if j in current_set:
                continue
            new_sum = current_sum + Fraction(1, j + 2)
            if new_sum < 1 and new_sum > current_sum:
                new_set = current_set | {j}
                new_indices = sorted(new_set)
                update_best(new_sum, new_indices)
    
    # Step 8: More aggressive local search - try all pairs of swaps
    if best_indices and elapsed() < DEADLINE:
        current_set = set(best_indices)
        current_sum = best_sum
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            # Try removing two elements and adding two different ones
            elems = list(current_set)
            for a in range(len(elems)):
                for b in range(a+1, len(elems)):
                    i, j = elems[a], elems[b]
                    removed_sum = Fraction(1, i + 2) + Fraction(1, j + 2)
                    for k in range(47):
                        if k in current_set:
                            continue
                        for l in range(k+1, 47):
                            if l in current_set:
                                continue
                            added_sum = Fraction(1, k + 2) + Fraction(1, l + 2)
                            new_sum = current_sum - removed_sum + added_sum
                            if new_sum < 1 and new_sum > current_sum:
                                new_set = (current_set - {i, j}) | {k, l}
                                new_indices = sorted(new_set)
                                update_best(new_sum, new_indices)
                                current_set = new_set
                                current_sum = new_sum
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if elapsed() > DEADLINE:
                return best_indices
    
    # Step 9: Final attempt - greedy with different orderings
    if elapsed() < DEADLINE:
        # Try greedy from largest denominators first, then add small ones
        for start_idx in range(47):
            selected, total = [], Fraction()
            for i in range(47):
                if i == start_idx:
                    continue
                if total + Fraction(1, i + 2) < 1:
                    selected.append(i)
                    total += Fraction(1, i + 2)
            if selected:
                update_best(total, selected)
            if elapsed() > DEADLINE:
                return best_indices
    
    return best_indices
# EVOLVE-BLOCK-END
