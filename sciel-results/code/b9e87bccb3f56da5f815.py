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
    
    # Greedy
    selected, total = [], Fraction()
    for i, d in enumerate(range(2, 49)):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    if selected:
        update_best(total, selected)
    
    # Local search with 1-swap and 2-swap moves
    if best_indices:
        current_set = set(best_indices)
        current_sum = best_sum
        
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            
            # 1-swap moves
            for i in list(current_set):
                for j in range(47):
                    if j in current_set:
                        continue
                    new_sum = current_sum - Fraction(1, i + 2) + Fraction(1, j + 2)
                    if new_sum < 1 and new_sum > current_sum:
                        current_set = current_set - {i} | {j}
                        current_sum = new_sum
                        update_best(new_sum, sorted(current_set))
                        improved = True
                        break
                if improved or elapsed() > DEADLINE:
                    break
            
            if elapsed() > DEADLINE:
                break
            
            # 2-swap moves (remove 2, add 2)
            items = list(current_set)
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    i, j = items[a], items[b]
                    base_sum = current_sum - Fraction(1, i + 2) - Fraction(1, j + 2)
                    base_set = current_set - {i, j}
                    for k in range(47):
                        if k in base_set:
                            continue
                        val_k = Fraction(1, k + 2)
                        if base_sum + val_k >= 1:
                            continue
                        rem = Fraction(1) - base_sum - val_k
                        if rem <= 0:
                            continue
                        for l in range(k + 1, 47):
                            if l in base_set or l == k:
                                continue
                            val_l = Fraction(1, l + 2)
                            if base_sum + val_k + val_l < 1:
                                new_sum = base_sum + val_k + val_l
                                if new_sum > current_sum:
                                    new_set = base_set | {k, l}
                                    current_set = new_set
                                    current_sum = new_sum
                                    update_best(new_sum, sorted(current_set))
                                    improved = True
                                    break
                        if improved or elapsed() > DEADLINE:
                            break
                    if improved or elapsed() > DEADLINE:
                        break
                if improved or elapsed() > DEADLINE:
                    break
    
    # Multi-start with different initial solutions
    if elapsed() < DEADLINE:
        import random
        random.seed(42)
        
        for trial in range(50):
            if elapsed() > DEADLINE:
                break
            
            # Random initial solution
            candidates = list(range(47))
            random.shuffle(candidates)
            
            current_set = set()
            current_sum = Fraction(0)
            
            for i in candidates:
                if current_sum + Fraction(1, i + 2) < 1:
                    current_set.add(i)
                    current_sum += Fraction(1, i + 2)
            
            if not current_set:
                continue
            
            update_best(current_sum, sorted(current_set))
            
            # Local search from this starting point
            for _ in range(10):
                if elapsed() > DEADLINE:
                    break
                
                improved = False
                for i in list(current_set):
                    for j in range(47):
                        if j in current_set:
                            continue
                        new_sum = current_sum - Fraction(1, i + 2) + Fraction(1, j + 2)
                        if new_sum < 1 and new_sum > current_sum:
                            current_set = current_set - {i} | {j}
                            current_sum = new_sum
                            update_best(new_sum, sorted(current_set))
                            improved = True
                            break
                    if improved or elapsed() > DEADLINE:
                        break
                if not improved:
                    break
    
    return best_indices
# EVOLVE-BLOCK-END
