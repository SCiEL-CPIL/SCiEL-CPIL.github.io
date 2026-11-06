import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    base = 51975
    
    # Find all divisors of base
    divisors = []
    for i in range(1, int(base**0.5) + 1):
        if base % i == 0:
            divisors.append(i)
            if i != base // i:
                divisors.append(base // i)
    
    # Filter: m > 1, odd
    divisors = [m for m in divisors if m > 1 and m % 2 == 1]
    divisors.sort(reverse=True)
    
    # Precompute indices for each modulus and residue
    indices_by_m = {}
    for m in divisors:
        res_indices = []
        for r in range(m):
            res_indices.append(np.arange(r, base, m))
        indices_by_m[m] = res_indices
    
    best_uncovered = base
    best_classes = []
    
    # Multi-start with different random seeds for diversity
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        if time.time() > deadline - 15:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(base, dtype=np.uint8)
        available_moduli = set(divisors)
        classes = []
        
        # Greedy phase with random tie-breaking
        while available_moduli:
            if time.time() > deadline - 20:
                break
                
            best_gain = -1
            best_m = None
            best_r = None
            best_indices = None
            best_tie = None
            
            for m in available_moduli:
                res_indices_list = indices_by_m[m]
                for r in range(m):
                    idx = res_indices_list[r]
                    gain = np.sum(covered[idx] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
                        best_indices = idx
                        best_tie = random.random()
                    elif gain == best_gain and gain > 0:
                        # Random tie-breaking for diversity
                        tie = random.random()
                        if best_tie is None or tie > best_tie:
                            best_tie = tie
                            best_m = m
                            best_r = r
                            best_indices = idx
                            
            if best_gain <= 0:
                break
                
            classes.append([best_m, best_r])
            covered[best_indices] = 1
            available_moduli.remove(best_m)
            
            current_uncovered = np.sum(covered == 0)
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in classes]
                
            if best_uncovered == 0:
                break
    
    # Local search phase on the best solution found
    if best_classes and time.time() < deadline - 10:
        random.seed(42)
        classes = [c[:] for c in best_classes]
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[indices_by_m[m][r]] = 1
        
        available_moduli = set(divisors) - set(c[0] for c in classes)
        
        improved = True
        max_rounds = 20
        for round_num in range(max_rounds):
            if time.time() > deadline - 5:
                break
            if not improved:
                break
            improved = False
            
            current_uncovered = np.sum(covered == 0)
            
            # Try removing one class and adding a different one
            for i, (m1, r1) in enumerate(classes):
                if time.time() > deadline - 5:
                    break
                    
                idx1 = indices_by_m[m1][r1]
                
                # Try all available moduli
                for m2 in list(available_moduli):
                    if time.time() > deadline - 5:
                        break
                    for r2 in range(m2):
                        if time.time() > deadline - 5:
                            break
                        idx2 = indices_by_m[m2][r2]
                        
                        # Compute new uncovered count if we swap class i with (m2, r2)
                        # Need to recompute coverage without class i but with (m2, r2)
                        new_covered = np.zeros(base, dtype=np.uint8)
                        for j, (mj, rj) in enumerate(classes):
                            if j != i:
                                new_covered[indices_by_m[mj][rj]] = 1
                        new_covered[idx2] = 1
                        
                        uncovered_after = np.sum(new_covered == 0)
                        
                        if uncovered_after < current_uncovered:
                            # Accept the swap
                            classes[i] = [m2, r2]
                            covered = new_covered
                            available_moduli.discard(m2)
                            available_moduli.add(m1)
                            current_uncovered = uncovered_after
                            improved = True
                            if uncovered_after < best_uncovered:
                                best_uncovered = uncovered_after
                                best_classes = [c[:] for c in classes]
    
    return best_classes
