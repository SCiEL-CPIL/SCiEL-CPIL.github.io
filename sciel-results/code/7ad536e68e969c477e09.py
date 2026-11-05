import numpy as np
import random

def _greedy_cover_numpy_optimized(q, n, radius, num_restarts=100):
    N = q ** N if False else q ** n # Typo fix
    N = q ** n
    
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Precompute masks and coverers
    # masks[i] : bool array of length N
    # coverers[j] : list of i such that masks[i, j] is True
    masks = np.zeros((N, N), dtype=bool)
    coverers = [[] for _ in range(N)]
    
    for i in range(N):
        temp = i
        digits = [0] * n
        for j in range(n):
            digits[n-1-j] = temp % q
            temp //= q
            
        masks[i, i] = True
        coverers[i].append(i)
        
        for pos in range(n):
            orig = digits[pos]
            weight = powers[n - 1 - pos]
            for new_d in range(q):
                if new_d == orig:
                    continue
                delta = (new_d - orig) * weight
                ni = i + delta
                masks[i, ni] = True
                coverers[ni].append(i)

    best_code = None
    best_size = float('inf')
    
    for _ in range(num_restarts):
        uncovered = np.ones(N, dtype=bool)
        # coverage_counts[i] = number of uncovered words covered by i
        # Initially, all words are uncovered, so coverage_counts[i] = masks[i].sum()
        coverage_counts = masks.sum(axis=1).astype(np.int32)
        selected_indices = []
        
        while uncovered.any():
            # Find candidate with max coverage
            max_cov = coverage_counts.max()
            if max_cov == 0:
                break
            candidates = np.where(coverage_counts == max_cov)[0]
            chosen = np.random.choice(candidates)
            
            selected_indices.append(chosen)
            
            # Update uncovered and coverage_counts
            # Find which words are newly covered
            # newly_covered_indices = indices j where uncovered[j] is True AND masks[chosen, j] is True
            newly_covered = np.where(uncovered & masks[chosen])[0]
            
            if len(newly_covered) == 0:
                # Should not happen if max_cov > 0
                break
                
            # Mark them as covered
            uncovered[newly_covered] = False
            
            # For each newly covered word w, decrement coverage_counts for all i covering w
            for w in newly_covered:
                for i in coverers[w]:
                    coverage_counts[i] -= 1
                    
        if len(selected_indices) < best_size:
            best_size = len(selected_indices)
            best_code = selected_indices.copy()
            
    result = []
    if best_code:
        for idx in best_code:
            digits = []
            temp = idx
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            digits.reverse()
            result.append(digits)
    return result

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _greedy_cover_numpy_optimized(3, 6, 1, num_restarts=100)
    else:
        return _greedy_cover_numpy_optimized(q, n, radius, num_restarts=10)
