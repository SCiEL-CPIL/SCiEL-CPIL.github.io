import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s
    
    # 255 = 3 * 5 * 17
    # Any element in the set must be adjacent to 255, i.e., gcd(x, 255) > 1
    # So x must be divisible by 3, 5, or 17
    
    # Build candidate list: all numbers in [1,255] with gcd(x,255)>1
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    # 255 must be in the set
    # Find index of 255
    idx_255 = candidates.index(255)
    
    n = len(candidates)
    
    # Build adjacency: adj[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    # Use bitmasks for efficiency
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # We need max clique containing idx_255
    # Start with the set of all vertices adjacent to 255 (plus 255 itself)
    # The clique must be a subset of {255} ∪ {v : v adjacent to 255}
    
    # Initialize: all multiples of 3 form a clique
    # Let's find a good initial clique
    init_clique = set()
    for i in range(n):
        if candidates[i] % 3 == 0:
            init_clique.add(i)
    
    # Verify it's a clique
    clique_list = list(init_clique)
    for i in range(len(clique_list)):
        for j in range(i+1, len(clique_list)):
            if not (adj[clique_list[i]] & (1 << clique_list[j])):
                break
    
    best_clique = init_clique.copy()
    best_size = len(best_clique)
    
    # Branch and bound for max clique containing idx_255
    # Use Bron-Kerbosch with pivot
    
    def popcount(x):
        return bin(x).count('1')
    
    # For max clique search, we work with a candidate set
    # The clique must contain idx_255, so all other vertices must be adjacent to idx_255
    # and to each other
    
    # Let's use a recursive branch and bound
    # State: current clique (as set), candidates (vertices that can be added)
    
    best = [best_size]
    best_set = [best_clique.copy()]
    
    def bron_kerbosch(clique, candidates_mask, excluded_mask):
        if time.time() > DEADLINE:
            return
        
        if popcount(candidates_mask) + len(clique) <= best[0]:
            return
        
        if candidates_mask == 0:
            if len(clique) > best[0]:
                best[0] = len(clique)
                best_set[0] = set(clique)
            return
        
        # Choose pivot: vertex with max neighbors in candidates
        # For simplicity, pick first vertex
        # Better: pick vertex with max degree in candidates subgraph
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
        branch_mask = candidates_mask & ~(adj[pivot] if pivot >= 0 else 0)
        
        temp = branch_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            
            new_clique = clique | {v}
            new_candidates = candidates_mask & adj[v]
            new_excluded = excluded_mask | candidates_mask  # not needed in this formulation
            
            bron_kerbosch(new_clique, new_candidates, new_excluded)
            
            candidates_mask &= ~(1 << v)
            excluded_mask |= (1 << v)
            
            if time.time() > DEADLINE:
                return
    
    # Start: clique = {idx_255}, candidates = all neighbors of 255
    init_clique_set = {idx_255}
    init_candidates = adj[idx_255]
    
    bron_kerbosch(init_clique_set, init_candidates, 0)
    
    # Convert back to values
    result = [candidates[i] for i in best_set[0]]
    result.sort()
    
    # Verify 255 is in result
    if 255 not in result:
        result.append(255)
    
    return result
