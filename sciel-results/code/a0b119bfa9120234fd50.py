"""Optimized odd covering system search using randomized greedy set cover."""
import numpy as np
import random

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11

def _get_divisors(n):
    """Get all divisors of n greater than 1."""
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

def construct():
    base = BASE
    # Precompute all divisors > 1 of 51975
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divs.append(d)
    
    # Precompute indices for all candidate classes
    # candidates: list of (m, r, indices_array)
    candidates = []
    for m in divisors:
        for r in range(m):
            indices = np.arange(r, base, m)
            candidates.append((m, r, indices))
    
    # Sort candidates by potential coverage (base // m) descending
    # This ensures we consider high-impact moduli first, but tie-breaking will be randomized
    candidates.sort(key=lambda x: -base // x[0])
    
    best_classes = None
    best_uncovered = base
    num_restarts = 10
    
    for _ in range(num_restarts):
        used_m = set()
        selected_indices = []
        remaining = np.ones(base, dtype=bool)  # True = uncovered
        
        for _step in range(len(divisors)):
            best_gain = -1
            best_candidates = []
            
            for i, (m, r, indices) in enumerate(candidates):
                if m in used_m:
                    continue
                gain = remaining[indices].sum()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates or best_gain <= 0:
                break
            
            # Randomly pick one from the best candidates
            idx = random.choice(best_candidates)
            m, r, indices = candidates[idx]
            used_m.add(m)
            selected_indices.append(idx)
            remaining[indices] = False
        
        current_uncovered = remaining.sum()
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_classes = [[candidates[i][0], candidates[i][1]] for i in selected_indices]
            
        # Early stop if we found a perfect covering
        if best_uncovered == 0:
            break
            
    return best_classes
