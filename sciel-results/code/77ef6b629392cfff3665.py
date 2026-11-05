import time
from math import gcd
import numpy as np

N = 512
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    # 1. Precompute LCMs and build conflict graph
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            if (a * b) // g > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_size = 0
    best_set = []
    
    # Helper to get max degree vertex in R
    def get_max_degree_vertex(R_mask):
        max_deg = -1
        best_v = -1
        temp_mask = R_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            deg = (adj[v] & R_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_mask ^= lsb
        return best_v

    # Helper to compute greedy coloring bound for R_mask
    # Returns the number of colors used, which is an upper bound on alpha(R)
    def greedy_color_bound(R_mask):
        # We want to color the graph induced by R_mask.
        # Greedy coloring: assign colors in some order.
        # To get a tight bound, we should order vertices by degree (descending) or just natural order?
        # Standard greedy coloring for MIS bound: 
        # We partition R into independent sets (color classes).
        # The number of classes is the upper bound.
        # To minimize the number of classes (tighter bound), we should pick vertices that have high degree?
        # Actually, the standard "coloring bound" in Tomita's algorithm uses a specific ordering.
        # Let's use a simple greedy: 
        # While R is not empty:
        #   Find a maximal independent set in R? No, that's hard.
        #   Greedy coloring: 
        #   colors = {}
        #   available = R_mask
        #   color_idx = 0
        #   while available:
        #      # Pick a vertex v from available
        #      # Remove v and its neighbors from available? No, that's for finding one independent set.
        #      # For coloring: 
        #      # We assign color color_idx to a set of non-adjacent vertices.
        #      # This is equivalent to finding a maximal independent set in the remaining graph, 
        #      # but we want to color ALL vertices.
        #      
        #      # A simpler greedy coloring algorithm:
        #      # Iterate vertices in some order. Assign smallest color not used by neighbors.
        #      # The number of colors is the bound.
        #      
        #      # To make it fast and effective for pruning, we can use the "saturation" heuristic or just simple degree ordering.
        #      # Let's try: Sort vertices in R by degree in R (descending).
        #      # Then assign colors greedily.
        
        vertices = []
        temp = R_mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            vertices.append(v)
            temp ^= lsb
            
        # Sort by degree in R (descending)
        # This helps produce fewer colors
        vertices.sort(key=lambda v: (adj[v] & R_mask).bit_count(), reverse=True)
        
        color_of = {}
        max_color = 0
        for v in vertices:
            # Find smallest color not used by neighbors
            used_colors = set()
            # Check neighbors of v in R
            neighbors_mask = adj[v] & R_mask
            t = neighbors_mask
            while t:
                lsb = t & -t
                u = lsb.bit_length() - 1
                if u in color_of:
                    used_colors.add(color_of[u])
                t ^= lsb
            
            # Find smallest non-negative integer not in used_colors
            c = 0
            while c in used_colors:
                c += 1
            
            color_of[v] = c
            if c > max_color:
                max_color = c
                
        return max_color + 1

    def solve(R_mask, current_size, current_set):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > DEADLINE:
            return
            
        if R_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Compute coloring bound
        # Optimization: If current_size + bit_count(R_mask) <= best_size, prune
        # But coloring bound is tighter.
        # However, coloring is O(N^2). If R is small, just use popcount?
        # For N=512, let's always try coloring if R is large enough, or just always.
        
        rem = R_mask.bit_count()
        if current_size + rem <= best_size:
            return
            
        # If R is small, coloring might be overhead. 
        # Let's use coloring bound if rem > 10, else popcount.
        # Actually, let's just use coloring bound, it's the key to Tomita's speed.
        
        # To avoid overhead of sorting and set operations in Python for every node,
        # we can approximate or use a faster bitset-based coloring?
        # Python sets are slow. Let's see if we can optimize.
        
        # Alternative: Use a precomputed list of vertices in R?
        # The `get_max_degree_vertex` also iterates.
        
        # Let's implement a faster coloring bound using bitsets if possible?
        # Or just stick to the list-based one but ensure it's not called too often.
        # The pruning is what reduces the call count.
        
        bound = greedy_color_bound(R_mask)
        
        if current_size + bound <= best_size:
            return
            
        # Select vertex with max degree in R
        v = get_max_degree_vertex(R_mask)
        
        if v == -1:
            return
            
        # Branch 1: Exclude v
        R_excl = R_mask & ~(1 << v)
        solve(R_excl, current_size, current_set)
        
        # Branch 2: Include v
        R_incl = R_mask & ~(adj[v] | (1 << v))
        current_set.append(v + 1)
        solve(R_incl, current_size + 1, current_set)
        current_set.pop()

    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    if best_size == 0:
        return [1]
        
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
