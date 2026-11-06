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
    
    # Precompute pairwise intersections for faster sunflower checks
    # pair_int[i][j] = candidates[i] & candidates[j]
    # Using a flat list for better cache locality
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
        family: list of indices
        k: index
        """
        n_fam = len(family)
        ck = candidates[k]
        
        for i in range(n_fam):
            idx_a = family[i]
            ca = candidates[idx_a]
            aik = ca & ck
            
            for j in range(i + 1, n_fam):
                idx_b = family[j]
                cb = candidates[idx_b]
                
                ab = pair_int[idx_a][idx_b]
                if ab != aik:
                    continue
                
                bk = cb & ck
                if ab == bk:
                    return False
        return True

    def find_blocker(family, k):
        """
        Find a blocking pair (a, b) in family that prevents adding k.
        Returns (a, b) indices or None if k can be added.
        """
        n_fam = len(family)
        ck = candidates[k]
        
        for i in range(n_fam):
            idx_a = family[i]
            ca = candidates[idx_a]
            aik = ca & ck
            
            for j in range(i + 1, n_fam):
                idx_b = family[j]
                cb = candidates[idx_b]
                
                ab = pair_int[idx_a][idx_b]
                if ab != aik:
                    continue
                
                bk = cb & ck
                if ab == bk:
                    return (idx_a, idx_b)
        return None

    best_family = []
    best_size = 0
    
    # Run multiple randomized greedy + local search attempts
    while time.time() < deadline:
        random.shuffle(candidates)
        order = list(range(N))
        random.shuffle(order)
        
        family = []
        family_set = set()
        
        # Phase 1: Greedy Construction
        for idx in order:
            if can_add(family, idx):
                family.append(idx)
                family_set.add(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family.copy()
            
        # Phase 2: Local Search (Swapping)
        steps_without_improvement = 0
        local_best_size = len(family)
        
        while time.time() < deadline - 2.0:
            # If we have run out of time or are stuck, break
            if steps_without_improvement > 500:
                break
                
            # Pick a random candidate from the complement
            idx = random.randint(0, N - 1)
            c = candidates[idx]
            
            if idx in family_set:
                continue
            
            blocker = find_blocker(family, idx)
            
            if blocker is None:
                # Valid addition
                family.append(idx)
                family_set.add(idx)
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = family.copy()
                steps_without_improvement = 0
                local_best_size = len(family)
            else:
                # Conflict found
                a, b = blocker
                if random.random() < 0.5:
                    victim = a
                else:
                    victim = b
                
                # Remove victim
                family.remove(victim)
                family_set.remove(victim)
                
                # Try to add c
                # We must re-check validity because there might be other blockers
                if can_add(family, idx):
                    family.append(idx)
                    family_set.add(idx)
                    # Size is same (removed 1, added 1)
                    # This is a swap. It doesn't increase size.
                    steps_without_improvement += 1
                else:
                    # Swap failed. Restore victim.
                    family.append(victim)
                    family_set.add(victim)
                    steps_without_improvement += 1

    # Convert best_family indices back to sorted lists of integers
    result = []
    for idx in best_family:
        m = candidates[idx]
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))
        
    return result
