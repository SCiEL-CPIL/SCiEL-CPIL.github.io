"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill
    
    # Denominators are 2, 3, ..., 48
    # Zero-based index i corresponds to denominator d = i + 2
    # We need a nonempty subset of indices such that sum(1/(i+2)) < 1
    # and we want to MINIMIZE the deficit 1 - sum, i.e., MAXIMIZE the sum subject to sum < 1.
    
    # Strategy:
    # 1. Greedy initial solution (from inspiration 1): take fractions in order until we'd exceed 1.
    #    This gives {0, 1} -> sum = 5/6, deficit = 1/6.
    # 2. Known good solution: [0, 1, 5, 41] -> 1/2 + 1/3 + 1/7 + 1/43 = 1805/1806, deficit = 1/1806.
    # 3. Brute force over first 10 denominators (indices 0-9, denoms 2-11).
    # 4. Targeted search: for each subset of "large" fractions (small denominators), 
    #    find the best single fraction from the remaining that gets closest to 1.
    # 5. Local search: try swapping one element in the best solution with another.
    
    best_sum = Fraction(0)
    best_indices = []
    
    def update_best(s, indices):
        nonlocal best_sum, best_indices
        if s < 1 and s > best_sum:
            best_sum = s
            best_indices = indices
    
    # Step 1: Greedy (from inspiration 1)
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
    
    # Step 3: Brute force over first 10 denominators
    for mask in range(1, 1 << 10):
        s = Fraction(0)
        indices = []
        for i in range(10):
            if mask & (1 << i):
                s += Fraction(1, i + 2)
                indices.append(i)
        if s < 1:
            update_best(s, indices)
        if time.time() - start_time > DEADLINE:
            return best_indices
    
    # Step 4: Targeted search
    # For each subset of the first 6 denominators (indices 0-5, denoms 2-7),
    # find the best single fraction from indices 6-46 that gets closest to 1.
    # Also try subsets of size 2-3 from first 8 denominators with one additional fraction.
    
    # First, try all subsets of {0,1,2,3,4,5} (denoms 2-7) of size 1-3
    # combined with one fraction from {6,...,46} (denoms 8-48)
    
    for size in range(1, 4):
        for combo in combinations(range(6), size):
            base_sum = Fraction(0)
            for i in combo:
                base_sum += Fraction(1, i + 2)
            if base_sum >= 1:
                continue
            # Find best single fraction from indices 6-46
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
        if time.time() - start_time > DEADLINE:
            return best_indices
    
    # Also try subsets of size 2 from first 8 denominators with one additional
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
    
    if time.time() - start_time > DEADLINE:
        return best_indices
    
    # Step 5: Local search - try swapping one element in best_indices with another
    if best_indices:
        current_set = set(best_indices)
        current_sum = best_sum
        improved = True
        while improved and time.time() - start_time < DEADLINE:
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
                if time.time() - start_time > DEADLINE:
                    return best_indices
    
    # Step 6: Try adding one more fraction to the current best
    if best_indices and time.time() - start_time < DEADLINE:
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
    
    return best_indices
# EVOLVE-BLOCK-END
