import time
import random

# Precompute all 6-cycles for n=17
# A 6-cycle is a set of 6 vertices and a cyclic ordering.
# We represent a cycle as a tuple of 6 vertex indices in order.
# The edges of the cycle are (v0,v1), (v1,v2), ..., (v4,v5), (v5,v0).
# We only need to store the list of edge indices for each cycle.

def get_edge_index(i, j, n):
    # Lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
    if i > j:
        i, j = j, i
    # Number of edges before (i, j) is:
    # For each k < i, there are (n - 1 - k) edges.
    # Plus (j - i - 1) edges from (i, i+1) to (i, j-1).
    return sum(n - 1 - k for k in range(i)) + (j - i - 1)

def precompute_cycles(n):
    cycles = []
    from itertools import combinations
    from math import factorial
    
    # Number of distinct cycles on 6 labeled vertices is (6-1)! / 2 = 60.
    # We can generate them by fixing v0=0 and permuting v1..v5, then dividing by 2 for direction.
    # Or simpler: iterate over all permutations of 5 elements, fix the first, and only take half.
    
    # Let's just generate all 60 cycles for a fixed set of 6 vertices.
    # Let the vertices be 0, 1, 2, 3, 4, 5.
    # A cycle is (0, a, b, c, d, e) where a,b,c,d,e is a permutation of 1,2,3,4,5.
    # Total permutations = 120.
    # Reverse of (0, a, b, c, d, e) is (0, e, d, c, b, a).
    # We can just take all 120 and deduplicate by sorting the tuple? No, cycles are undirected.
    # Standard way: Fix 0. Permute 1,2,3,4,5. Take only those where a < e? No, that's not quite right for all cases.
    # Actually, (0, a, b, c, d, e) is the same as (0, e, d, c, b, a).
    # So we can just generate all 120, convert to a canonical form (e.g. min of tuple and reversed tuple), and unique them.
    
    base_cycles = set()
    from itertools import permutations
    for perm in permutations([1, 2, 3, 4, 5]):
        cycle = (0, perm[0], perm[1], perm[2], perm[3], perm[4])
        rev = (0, perm[4], perm[3], perm[2], perm[1], perm[0])
        canon = min(cycle, rev)
        base_cycles.add(canon)
        
    # There should be 60 of these.
    for comb in combinations(range(n), 6):
        # Map 0..5 to the vertices in comb
        mapping = {k: v for k, v in enumerate(comb)}
        for b_cycle in base_cycles:
            mapped_cycle = tuple(mapping[v] for v in b_cycle)
            # Store edge indices
            edges = []
            for k in range(6):
                u = mapped_cycle[k]
                v = mapped_cycle[(k + 1) % 6]
                idx = get_edge_index(u, v, n)
                edges.append(idx)
            cycles.append(tuple(edges))
            
    return cycles

def construct():
    start_time = time.time()
    deadline = 110.0  # Keep safety margin
    
    # Try n=17 first
    n = 17
    num_edges = n * (n - 1) // 2
    cycles = precompute_cycles(n)
    
    # Penalty function: count of monochromatic 6-cycles
    # colors: list of length num_edges, each in {0,1,2,3}
    
    def penalty(colors):
        p = 0
        for cycle in cycles:
            c0 = colors[cycle[0]]
            # Check if all 6 are same color
            if all(colors[idx] == c0 for idx in cycle[1:]):
                p += 1
        return p
    
    # Local search
    best_penalty = float('inf')
    best_colors = None
    
    for attempt in range(50):  # Max 50 restarts
        if time.time() > deadline:
            break
            
        # Random initialization
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        cur_penalty = penalty(colors)
        
        if cur_penalty < best_penalty:
            best_penalty = cur_penalty
            best_colors = colors[:]
            
        if cur_penalty == 0:
            break
            
        # Hill climbing
        max_steps = 200
        stuck_count = 0
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if stuck_count > 100:
                break
                
            # Pick a random edge to change
            idx = random.randint(0, num_edges - 1)
            old_color = colors[idx]
            
            # Find the color that minimizes penalty
            best_c = old_color
            min_p = cur_penalty
            
            for c in range(4):
                if c == old_color:
                    continue
                colors[idx] = c
                new_p = penalty(colors)
                if new_p < min_p:
                    min_p = new_p
                    best_c = c
            colors[idx] = best_c
            
            if min_p < cur_penalty:
                cur_penalty = min_p
                stuck_count = 0
                if cur_penalty < best_penalty:
                    best_penalty = cur_penalty
                    best_colors = colors[:]
                if cur_penalty == 0:
                    break
            else:
                # Revert if no improvement (simple hill climbing)
                colors[idx] = old_color
                stuck_count += 1
                
        if cur_penalty == 0:
            break
            
    if best_penalty == 0:
        return {'n': n, 'colors': best_colors}
        
    # Fallback to n=9 if 17 failed (though it shouldn't)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
