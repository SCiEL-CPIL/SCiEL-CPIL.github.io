import time
import math
import random
import numpy as np
from itertools import combinations, product

def score_set(U):
    """Compute the score for a set U."""
    if not U or 0 not in U:
        return -1
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return -1
    
    max_u = max(U)
    if max_u == 0:
        return -1
        
    # Compute sumset and difference set using numpy for speed
    U_arr = np.array(U)
    sumset = set((U_arr[:, None] + U_arr[None, :]).flatten().tolist())
    diffset = set((U_arr[:, None] - U_arr[None, :]).flatten().tolist())
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
        
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
        
    return 1.0 + math.log(ratio) / denom

def build_mixed_radix_set(radices, digit_sets):
    """Build a set from mixed-radix representation.
    U = {sum d_i * r_0 * r_1 * ... * r_{i-1} : d_i in D_i}
    """
    if not radices:
        return [0]
    
    # Compute positional weights
    weights = [1]
    for r in radices[:-1]:
        weights.append(weights[-1] * r)
    
    # Build set via Cartesian product
    U = {0}
    for i, (weight, digits) in enumerate(zip(weights, digit_sets)):
        new_U = set()
        for u in U:
            for d in digits:
                new_U.add(u + d * weight)
        U = new_U
    
    return sorted(U)

def optimize_mixed_radix(deadline):
    """Greedy hill-climbing optimization of mixed-radix parameters."""
    best_set = [0, 1, 2, 4, 8, 16, 32, 64]
    best_score = score_set(best_set)
    
    # Start with a reasonable mixed-radix structure
    # Try different numbers of blocks
    for n_blocks in range(2, 6):
        if time.time() > deadline - 10:
            break
        
        # Initial radices: try geometric-like growth
        for base_growth in [1.5, 2.0, 2.5, 3.0]:
            if time.time() > deadline - 10:
                break
            
            radices = [int(base_growth ** i) + 1 for i in range(n_blocks)]
            # Ensure radices are at least 2
            radices = [max(2, r) for r in radices]
            
            # Digit sets: start with full ranges
            digit_sets = [list(range(r)) for r in radices]
            
            current_set = build_mixed_radix_set(radices, digit_sets)
            if not current_set or 0 not in current_set:
                continue
            current_score = score_set(current_set)
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
            
            # Greedy hill-climbing: try modifying one radix or digit set at a time
            improved = True
            max_iter = 50
            for _ in range(max_iter):
                if time.time() > deadline - 10:
                    break
                if not improved:
                    break
                improved = False
                
                # Try modifying a single radix
                for i in range(len(radices)):
                    if time.time() > deadline - 10:
                        break
                    for new_r in range(max(2, radices[i] - 3), min(radices[i] + 4, 50)):
                        if new_r == radices[i]:
                            continue
                        new_radices = list(radices)
                        new_radices[i] = new_r
                        new_digit_sets = [list(range(r)) for r in new_radices]
                        
                        test_set = build_mixed_radix_set(new_radices, new_digit_sets)
                        if not test_set or 0 not in test_set:
                            continue
                        test_score = score_set(test_set)
                        
                        if test_score > current_score + 1e-10:
                            radices = new_radices
                            digit_sets = new_digit_sets
                            current_set = test_set
                            current_score = test_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current_set)
                            break
                
                # Try modifying a single digit set (subset of full range)
                for i in range(len(digit_sets)):
                    if time.time() > deadline - 10:
                        break
                    # Try removing one element from digit set
                    if len(digit_sets[i]) > 1:
                        for d_to_remove in digit_sets[i][:]:
                            new_digit_sets = [list(ds) for ds in digit_sets]
                            new_digit_sets[i] = [d for d in new_digit_sets[i] if d != d_to_remove]
                            
                            test_set = build_mixed_radix_set(radices, new_digit_sets)
                            if not test_set or 0 not in test_set:
                                continue
                            test_score = score_set(test_set)
                            
                            if test_score > current_score + 1e-10:
                                digit_sets = new_digit_sets
                                current_set = test_set
                                current_score = test_score
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_set = list(current_set)
                                break
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_set = [0, 1, 3, 7]
    best_score = score_set(best_set)
    
    # Phase 1: Mixed-radix structural search
    mixed_set, mixed_score = optimize_mixed_radix(deadline)
    if mixed_score > best_score:
        best_score = mixed_score
        best_set = mixed_set
    
    # Phase 2: Exhaustive search for small sets
    M_global = 20
    range_vals = range(1, M_global + 1)
    
    for n in range(3, 9):
        if time.time() > deadline - 15:
            break
        for comb in combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
            if time.time() > deadline - 15:
                break
                
    # Phase 3: Local search from best_set
    random.seed(42)
    current_set = list(best_set)
    current_score = best_score
    
    max_iter = 20000
    for i in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        new_set = set(current_set)
        mutation_type = random.choice(['add', 'remove', 'replace'])
        
        if mutation_type == 'add' and len(new_set) < 20:
            max_elem = max(new_set)
            new_elem = random.randint(1, max_elem + 15)
            if new_elem not in new_set:
                new_set.add(new_elem)
        elif mutation_type == 'remove' and len(new_set) > 2:
            elems = [x for x in new_set if x != 0]
            if elems:
                new_set.remove(random.choice(elems))
        elif mutation_type == 'replace' and len(new_set) > 2:
            elems = [x for x in new_set if x != 0]
            if elems:
                old = random.choice(elems)
                new_set.remove(old)
                max_elem = max(new_set) if new_set else 1
                new_elem = random.randint(1, max_elem + 15)
                if new_elem not in new_set:
                    new_set.add(new_elem)
        
        if len(new_set) < 2:
            continue
        
        new_score = score_set(list(new_set))
        if new_score > current_score:
            current_set = list(new_set)
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(new_set)
                
    # Phase 4: Final polish - try to add/remove single elements for marginal gains
    if time.time() < deadline - 3:
        current_set = list(best_set)
        current_score = best_score
        for _ in range(500):
            if time.time() > deadline - 3:
                break
            
            # Try adding one element
            max_elem = max(current_set)
            for new_elem in range(1, max_elem + 8):
                if new_elem in current_set:
                    continue
                test_set = current_set + [new_elem]
                test_score = score_set(test_set)
                if test_score > current_score + 1e-10:
                    current_set = test_set
                    current_score = test_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                    break
            
            # Try removing one element
            if len(current_set) > 3:
                for elem in current_set:
                    if elem == 0:
                        continue
                    test_set = [x for x in current_set if x != elem]
                    test_score = score_set(test_set)
                    if test_score > current_score + 1e-10:
                        current_set = test_set
                        current_score = test_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set)
                        break
                        
    return best_set
