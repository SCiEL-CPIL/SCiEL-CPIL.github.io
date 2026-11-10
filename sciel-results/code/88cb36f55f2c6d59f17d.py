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
    
    # Precompute reciprocal values
    recip = [Fraction(1, i + 2) for i in range(47)]
    
    # Known best solutions from inspirations
    # Deficit 1/4070: {1/3, 1/4, 1/5, 1/6, 1/37, 1/44}
    c1 = [1, 2, 3, 4, 35, 42]
    s1 = sum(recip[i] for i in c1)
    update_best(s1, c1)
    
    # Deficit 1/4173: {1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/12, 1/41}
    c2 = [1, 2, 3, 4, 5, 6, 10, 39]
    s2 = sum(recip[i] for i in c2)
    update_best(s2, c2)
    
    # Step 1: Greedy initial solution
    selected, total = [], Fraction(0)
    for i, d in enumerate(range(2, 49)):
        if total + recip[i] < 1:
            selected.append(i)
            total += recip[i]
    if selected:
        update_best(total, selected)
    
    # Step 2: Brute force over first 12 denominators (indices 0-11, denoms 2-13)
    for mask in range(1, 1 << 12):
        s = Fraction(0)
        indices = []
        for i in range(12):
            if mask & (1 << i):
                s += recip[i]
                indices.append(i)
        if s < 1:
            update_best(s, indices)
        if elapsed() > DEADLINE - 5:
            return best_indices
    
    # Step 3: Meet-in-the-middle approach
    # Split indices into two halves: A (0-23) and B (24-46)
    # For each half, enumerate all subsets and their sums
    # Then combine subsets from A and B to find best total < 1
    
    half_a = list(range(0, 24))
    half_b = list(range(24, 47))
    
    # Enumerate all subsets of half_a with their sums
    # 2^24 = 16M which is too many, so we'll use a smarter approach
    # Instead, enumerate subsets of a smaller core and use greedy for the rest
    
    # Step 3a: Enumerate subsets of first 15 denominators (indices 0-14)
    # and for each, greedily add from indices 15-46
    for mask in range(1, 1 << 15):
        if elapsed() > DEADLINE - 10:
            break
        
        base_sum = Fraction(0)
        base_set = set()
        for i in range(15):
            if mask & (1 << i):
                base_sum += recip[i]
                base_set.add(i)
        
        if base_sum >= 1:
            continue
        
        # Greedily add from indices 15-46
        current_sum = base_sum
        current_set = set(base_set)
        for i in range(15, 47):
            if i not in current_set:
                if current_sum + recip[i] < 1:
                    current_set.add(i)
                    current_sum += recip[i]
        
        if current_set:
            update_best(current_sum, sorted(current_set))
    
    # Step 3b: Also try subsets of indices 10-24 (middle range) with greedy fill
    if elapsed() < DEADLINE - 10:
        mid_indices = list(range(10, 25))
        for mask in range(1, 1 << len(mid_indices)):
            if elapsed() > DEADLINE - 10:
                break
            
            base_sum = Fraction(0)
            base_set = set()
            for k, i in enumerate(mid_indices):
                if mask & (1 << k):
                    base_sum += recip[i]
                    base_set.add(i)
            
            if base_sum >= 1:
                continue
            
            # Greedily add from all other indices
            current_sum = base_sum
            current_set = set(base_set)
            for i in range(47):
                if i not in current_set:
                    if current_sum + recip[i] < 1:
                        current_set.add(i)
                        current_sum += recip[i]
            
            if current_set:
                update_best(current_sum, sorted(current_set))
    
    # Step 4: Targeted search - for each subset of small denominators,
    # find the best combination of 1-2 additional fractions from larger denominators
    for size in range(1, 4):
        if elapsed() > DEADLINE - 15:
            break
        for combo in combinations(range(8), size):
            base_sum = sum(recip[i] for i in combo)
            if base_sum >= 1:
                continue
            target = Fraction(1) - base_sum
            if target <= 0:
                continue
            
            # Try adding one fraction from larger denominators
            best_single = None
            best_single_val = Fraction(0)
            for j in range(8, 47):
                val = recip[j]
                if base_sum + val < 1:
                    if val > best_single_val:
                        best_single_val = val
                        best_single = j
            if best_single is not None:
                total_sum = base_sum + best_single_val
                indices = list(combo) + [best_single]
                update_best(total_sum, indices)
            
            # Try adding two fractions from larger denominators
            if elapsed() > DEADLINE - 15:
                break
            for j in range(8, 47):
                if elapsed() > DEADLINE - 15:
                    break
                s1 = base_sum + recip[j]
                if s1 >= 1:
                    continue
                for k in range(j + 1, 47):
                    s2 = s1 + recip[k]
                    if s2 < 1:
                        total_sum = s2
                        indices = list(combo) + [j, k]
                        update_best(total_sum, indices)
                    else:
                        break  # Since recip[k] decreases, no need to continue
    
    # Step 5: More targeted search - subsets of size 3 from first 7 with one additional
    if elapsed() < DEADLINE - 15:
        for combo in combinations(range(7), 3):
            base_sum = sum(recip[i] for i in combo)
            if base_sum >= 1:
                continue
            target = Fraction(1) - base_sum
            if target <= 0:
                continue
            best_single = None
            best_single_val = Fraction(0)
            for j in range(7, 47):
                val = recip[j]
                if base_sum + val < 1:
                    if val > best_single_val:
                        best_single_val = val
                        best_single = j
            if best_single is not None:
                total_sum = base_sum + best_single_val
                indices = list(combo) + [best_single]
                update_best(total_sum, indices)
            if elapsed() > DEADLINE - 15:
                break
    
    # Step 6: Local search - try swapping one element in best_indices with another
    if best_indices and elapsed() < DEADLINE - 5:
        current_set = set(best_indices)
        current_sum = best_sum
        improved = True
        while improved and elapsed() < DEADLINE - 3:
            improved = False
            for i in list(current_set):
                # Try removing i and adding j
                for j in range(47):
                    if j in current_set:
                        continue
                    new_sum = current_sum - recip[i] + recip[j]
                    if new_sum < 1 and new_sum > current_sum:
                        new_set = current_set - {i} | {j}
                        new_indices = sorted(new_set)
                        update_best(new_sum, new_indices)
                        current_set = new_set
                        current_sum = new_sum
                        improved = True
                        break
                if elapsed() > DEADLINE - 3:
                    break
    
    # Step 7: Try adding one more fraction to the current best
    if best_indices and elapsed() < DEADLINE - 5:
        current_set = set(best_indices)
        current_sum = best_sum
        for j in range(47):
            if j in current_set:
                continue
            new_sum = current_sum + recip[j]
            if new_sum < 1 and new_sum > current_sum:
                new_set = current_set | {j}
                new_indices = sorted(new_set)
                update_best(new_sum, new_indices)
    
    # Step 8: More aggressive local search - try all pairs of swaps
    if best_indices and elapsed() < DEADLINE - 5:
        current_set = set(best_indices)
        current_sum = best_sum
        improved = True
        while improved and elapsed() < DEADLINE - 3:
            improved = False
            # Try removing two elements and adding two different ones
            elems = list(current_set)
            for a in range(len(elems)):
                if elapsed() > DEADLINE - 3:
                    break
                for b in range(a + 1, len(elems)):
                    if elapsed() > DEADLINE - 3:
                        break
                    i, j = elems[a], elems[b]
                    removed_sum = recip[i] + recip[j]
                    for k in range(47):
                        if elapsed() > DEADLINE - 3:
                            break
                        if k in current_set:
                            continue
                        for l in range(k + 1, 47):
                            if l in current_set:
                                continue
                            added_sum = recip[k] + recip[l]
                            new_sum = current_sum - removed_sum + added_sum
                            if new_sum < 1 and new_sum > current_sum:
                                new_set = (current_set - {i, j}) | {k, l}
                                new_indices = sorted(new_set)
                                update_best(new_sum, new_indices)
                                current_set = new_set
                                current_sum = new_sum
                                improved = True
                                break
                        if improved or elapsed() > DEADLINE - 3:
                            break
                    if improved or elapsed() > DEADLINE - 3:
                        break
                if improved or elapsed() > DEADLINE - 3:
                    break
    
    # Step 9: Final attempt - greedy with different orderings
    if elapsed() < DEADLINE - 3:
        # Try greedy from largest denominators first, then add small ones
        for start_idx in range(47):
            if elapsed() > DEADLINE - 3:
                break
            selected, total = [], Fraction(0)
            for i in range(47):
                if i == start_idx:
                    continue
                if total + recip[i] < 1:
                    selected.append(i)
                    total += recip[i]
            if selected:
                update_best(total, selected)
    
    return best_indices
# EVOLVE-BLOCK-END
