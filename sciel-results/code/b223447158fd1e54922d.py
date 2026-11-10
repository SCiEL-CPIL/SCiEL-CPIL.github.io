"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    best_sum = Fraction(0)
    best_indices = []
    
    def update_best(s, indices):
        nonlocal best_sum, best_indices
        if s < 1 and s > best_sum:
            best_sum = s
            best_indices = list(indices)
            
    def elapsed():
        return time.time() - start_time
    
    # Known best solutions
    # Deficit 1/4070: {1/3, 1/4, 1/5, 1/6, 1/37, 1/44}
    c1 = [1, 2, 3, 4, 35, 42]
    s1 = Fraction(1, 3) + Fraction(1, 4) + Fraction(1, 5) + Fraction(1, 6) + Fraction(1, 37) + Fraction(1, 44)
    update_best(s1, c1)
    
    # Deficit 1/4173: {1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/12, 1/41}
    c2 = [1, 2, 3, 4, 5, 6, 10, 39]
    s2 = Fraction(1, 3) + Fraction(1, 4) + Fraction(1, 5) + Fraction(1, 6) + Fraction(1, 7) + Fraction(1, 8) + Fraction(1, 12) + Fraction(1, 41)
    update_best(s2, c2)
    
    # Greedy: pick largest fractions first
    selected, total = [], Fraction(0)
    for i, d in enumerate(range(2, 49)):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    if selected:
        update_best(total, selected)
    
    # Key structural idea: enumerate subsets of LARGE denominators (small fractions),
    # then greedily fill in small denominators (large fractions) that fit.
    # Large denominators: indices 15-46 (denoms 17-48)
    # We enumerate subsets of size up to 4 from these, then greedily add small denoms.
    
    large_indices = list(range(15, 47))  # indices 15 to 46, denoms 17 to 48
    small_indices = list(range(0, 15))   # indices 0 to 14, denoms 2 to 16
    
    # Precompute reciprocal values
    recip = [Fraction(1, i + 2) for i in range(47)]
    
    # Enumerate subsets of large denominators of size 1 to 4
    for size in range(1, 5):
        if elapsed() > DEADLINE - 2:
            break
        for subset in combinations(large_indices, size):
            if elapsed() > DEADLINE - 2:
                break
            
            subset_sum = Fraction(0)
            subset_set = set()
            for idx in subset:
                subset_sum += recip[idx]
                subset_set.add(idx)
            
            if subset_sum >= 1:
                continue
            
            # Greedily add small denominators that fit
            current_set = set(subset_set)
            current_sum = subset_sum
            
            for idx in small_indices:
                if idx not in current_set:
                    if current_sum + recip[idx] < 1:
                        current_set.add(idx)
                        current_sum += recip[idx]
            
            if current_set:
                update_best(current_sum, sorted(current_set))
    
    # Also try subsets of size 5 from large denominators (but limit to those with sum < 1)
    if elapsed() < DEADLINE - 5:
        for subset in combinations(large_indices, 5):
            if elapsed() > DEADLINE - 2:
                break
            
            subset_sum = Fraction(0)
            subset_set = set()
            for idx in subset:
                subset_sum += recip[idx]
                subset_set.add(idx)
            
            if subset_sum >= 1:
                continue
            
            # Greedily add small denominators
            current_set = set(subset_set)
            current_sum = subset_sum
            
            for idx in small_indices:
                if idx not in current_set:
                    if current_sum + recip[idx] < 1:
                        current_set.add(idx)
                        current_sum += recip[idx]
            
            if current_set:
                update_best(current_sum, sorted(current_set))
    
    # Local search refinement from the best solution found
    if best_indices and elapsed() < DEADLINE - 3:
        current_set = set(best_indices)
        current_sum = best_sum
        
        improved = True
        while improved and elapsed() < DEADLINE - 1:
            improved = False
            
            # 1-swap moves
            for i in list(current_set):
                if elapsed() > DEADLINE - 1:
                    break
                for j in range(47):
                    if j in current_set:
                        continue
                    new_sum = current_sum - recip[i] + recip[j]
                    if new_sum < 1 and new_sum > current_sum:
                        current_set = current_set - {i} | {j}
                        current_sum = new_sum
                        update_best(new_sum, sorted(current_set))
                        improved = True
                        break
            
            if elapsed() > DEADLINE - 1:
                break
            
            # 2-swap moves (remove 2, add 2)
            items = list(current_set)
            for a in range(len(items)):
                if elapsed() > DEADLINE - 1:
                    break
                for b in range(a + 1, len(items)):
                    if elapsed() > DEADLINE - 1:
                        break
                    i, j = items[a], items[b]
                    base_sum = current_sum - recip[i] - recip[j]
                    base_set = current_set - {i, j}
                    for k in range(47):
                        if elapsed() > DEADLINE - 1:
                            break
                        if k in base_set:
                            continue
                        if base_sum + recip[k] >= 1:
                            continue
                        for l in range(k + 1, 47):
                            if l in base_set or l == k:
                                continue
                            if base_sum + recip[k] + recip[l] < 1:
                                new_sum = base_sum + recip[k] + recip[l]
                                if new_sum > current_sum:
                                    current_set = base_set | {k, l}
                                    current_sum = new_sum
                                    update_best(new_sum, sorted(current_set))
                                    improved = True
                                    break
                            if improved or elapsed() > DEADLINE - 1:
                                break
                    if improved or elapsed() > DEADLINE - 1:
                        break
                if improved or elapsed() > DEADLINE - 1:
                    break
    
    return best_indices
# EVOLVE-BLOCK-END
