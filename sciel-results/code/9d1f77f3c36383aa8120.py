import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    candidates = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
        
    N = len(candidates)
    
    # Precompute pairwise intersections as a numpy array
    # pair_int[i][j] = intersection of candidates[i] and candidates[j]
    pair_int = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        ci = candidates[i]
        for j in range(i + 1, N):
            val = ci & candidates[j]
            pair_int[i, j] = val
            pair_int[j, i] = val

    def can_add(family, k):
        """
        Check if candidate k can be added to family without forming a sunflower.
        Uses precomputed pair_int for speed.
        """
        n_fam = len(family)
        ck = candidates[k]
        
        # For each pair (a, b) in family, check if |A ∩ B| == |A ∩ K| == |B ∩ K|
        for i in range(n_fam):
            idx_a = family[i]
            ca = candidates[idx_a]
            aik = ca & ck
            
            for j in range(i + 1, n_fam):
                idx_b = family[j]
                cb = candidates[idx_b]
                
                ab = pair_int[idx_a, idx_b]
                if ab == aik:
                    bk = cb & ck
                    if ab == bk:
                        return False
        return True

    # Initialize best_family with a valid single set
    best_family = [0] 
    best_size = 1
    
    # Run multiple randomized greedy attempts
    while time.time() < deadline:
        order = list(range(N))
        random.shuffle(order)
        
        family = []
        
        # Phase 1: Greedy Construction
        for idx in order:
            if can_add(family, idx):
                family.append(idx)
                
        if len(family) > best_size:
            best_size = len(family)
            best_family = family.copy()
            
        # Phase 2: Local search - try to add more elements
        if time.time() < deadline - 5.0:
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                # Try to add a random element not in family
                for _ in range(200):
                    idx = random.randint(0, N - 1)
                    if idx not in family:
                        if can_add(family, idx):
                            family.append(idx)
                            improved = True
                            if len(family) > best_size:
                                best_size = len(family)
                                best_family = family.copy()
                            break

    # Convert best_family indices back to sorted lists of integers
    result = []
    for idx in best_family:
        m = candidates[idx]
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))
        
    return result
