import time
import random
from itertools import combinations

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
    
    # Precompute pairwise intersections
    # Using a list of lists for pair_int
    pair_int = [[0] * N for _ in range(N)]
    for i in range(N):
        ci = candidates[i]
        for j in range(i + 1, N):
            val = ci & candidates[j]
            pair_int[i][j] = val
            pair_int[j][i] = val

    def can_add(family, k):
        """
        Check if candidate k can be added to family without forming a sunflower.
        """
        n_fam = len(family)
        ck = candidates[k]
        
        # Iterate over pairs in family
        for i in range(n_fam):
            idx_a = family[i]
            ca = candidates[idx_a]
            aik = ca & ck
            
            for j in range(i + 1, n_fam):
                idx_b = family[j]
                cb = candidates[idx_b]
                
                ab = pair_int[idx_a][idx_b]
                if ab == aik:
                    bk = cb & ck
                    if ab == bk:
                        return False
        return True

    # Initialize best_family with a valid single set to ensure non-empty return
    # [1, 2, 3, 4] is a valid 4-subset
    initial_mask = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
    initial_idx = 0 # Find index of this mask? Or just use 0 and ensure it's valid?
    # Actually, let's just pick the first candidate as the seed for best_family
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
            
        # Optional: Local search if time permits
        if time.time() < deadline - 5.0 and len(family) < 250:
             # Simple swap local search
             improved = True
             while improved and time.time() < deadline - 5.0:
                 improved = False
                 # Try to add a random element not in family
                 # Or swap?
                 # Let's just try adding randoms first
                 for _ in range(100):
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
