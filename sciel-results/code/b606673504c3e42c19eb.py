import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 88.0  # safety margin before 90s wall-clock

    # 255 = 3 * 5 * 17
    # Any element x in the set must have gcd(x, 255) > 1, so x divisible by 3, 5, or 17.
    
    # Build candidate list: all numbers in [1,255] with gcd(x,255)>1
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency using bitmasks
    # adj[i] is a bitmask where bit j is set if gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # We need max clique containing idx_255
    # The clique must be a subset of {255} ∪ {v : v adjacent to 255}
    # i.e., {idx_255} ∪ (adj[idx_255] vertices)
    
    def popcount(x):
        return x.bit_count()
    
    best_size = [0]
    best_clique = [set()]
    
    # Bron-Kerbosch with pivot for maximum clique
    def bron_kerbosch(clique, candidates_mask):
        if time.time() > DEADLINE:
            return
        
        if popcount(candidates_mask) + len(clique) <= best_size[0]:
            return
        
        if candidates_mask == 0:
            if len(clique) > best_size[0]:
                best_size[0] = len(clique)
                best_clique[0] = set(clique)
            return
        
        # Choose pivot: vertex with max neighbors in candidates_mask
        pivot = -1
        max_deg = -1
        temp = candidates_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            deg = popcount(adj[v] & candidates_mask)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        # Branch on candidates not adjacent to pivot
        if pivot >= 0:
            branch_mask = candidates_mask & ~(adj[pivot])
        else:
            branch_mask = candidates_mask
        
        temp = branch_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            
            new_clique = clique | {v}
            new_candidates = candidates_mask & adj[v]
            
            bron_kerbosch(new_clique, new_candidates)
            
            candidates_mask &= ~(1 << v)
            
            if time.time() > DEADLINE:
                return
    
    # Start: clique = {idx_255}, candidates = all neighbors of 255
    init_clique = {idx_255}
    init_candidates = adj[idx_255]
    
    bron_kerbosch(init_clique, init_candidates)
    
    # Greedy local search polish: try to add vertices one by one
    if time.time() < DEADLINE:
        current_clique = best_clique[0].copy()
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if i in current_clique:
                    continue
                # Check if i is adjacent to all in current_clique
                can_add = True
                for j in current_clique:
                    if not (adj[i] & (1 << j)):
                        can_add = False
                        break
                if can_add:
                    current_clique.add(i)
                    improved = True
        if len(current_clique) > best_size[0]:
            best_size[0] = len(current_clique)
            best_clique[0] = current_clique
    
    # Try more aggressive local search: swap-based
    if time.time() < DEADLINE:
        current_clique = best_clique[0].copy()
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for i in range(n):
                if i in current_clique:
                    continue
                # Try to add i by removing some elements
                # First check if we can just add
                can_add = all((adj[i] & (1 << j)) for j in current_clique)
                if can_add:
                    current_clique.add(i)
                    improved = True
                    break
                # Try swap: remove one element j, add i
                for j in list(current_clique):
                    if i == j:
                        continue
                    new_clique = current_clique - {j}
                    if all((adj[i] & (1 << k)) for k in new_clique):
                        # Check all pairs in new_clique still work (they did before)
                        current_clique = new_clique | {i}
                        improved = True
                        break
                if improved:
                    break
    
    # Convert back to values
    result = sorted([candidates[i] for i in best_clique[0]])
    
    # Verify 255 is in result
    if 255 not in result:
        result.append(255)
    
    return result
