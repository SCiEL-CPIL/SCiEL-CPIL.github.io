import time
from math import gcd

def construct():
    """
    Finds a large subset of [1, 512] such that lcm(a, b) <= 512 for all pairs.
    Uses optimized Tomita algorithm for Maximum Clique with bitset optimization
    and a greedy initialization to establish a strong lower bound.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 512
    nodes = list(range(1, N + 1))
    
    # 1. Build adjacency bitmasks
    # adj[i] is a bitmask where bit j is 1 if gcd(i, j) * (i*j)//gcd(i,j) <= 512? No, lcm(i,j) <= 512.
    # lcm(i, j) = i * j // gcd(i, j)
    
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            l = (i * j) // g
            if l <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Greedy Initial Clique
    # We want to find a large clique quickly to set best_size.
    # Strategy: Greedily pick vertices with high degree in the current candidate set P.
    
    def get_greedy_clique(P_mask):
        clique = []
        current_P = P_mask
        while current_P:
            # Find vertex in current_P with max degree within current_P
            best_v = -1
            best_deg = -1
            m = current_P
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = (adj[v] & current_P).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
            
            if best_v == -1:
                break
                
            clique.append(best_v)
            # Update P: intersect with neighbors of best_v
            current_P &= adj[best_v]
            
        return clique

    initial_P = 0
    for i in range(1, N + 1):
        initial_P |= (1 << i)
        
    # Get a good initial clique
    greedy_clique = get_greedy_clique(initial_P)
    best_clique = greedy_clique[:]
    best_size = len(greedy_clique)
    
    # Helper to get list of vertices from bitmask
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        return verts

    # Helper to greedily color vertices in P
    # Returns list of (vertex, color) sorted by color descending
    def color_vertices(P_mask):
        verts = get_vertices(P_mask)
        if not verts:
            return []
            
        # Sort by degree in P descending
        degs = []
        for v in verts:
            degs.append((v, (adj[v] & P_mask).bit_count()))
        degs.sort(key=lambda x: x[1], reverse=True)
        
        colors = [] # list of (vertex, color)
        uncolored = [v for v, _ in degs]
        current_color = 0
        
        # Precompute neighbor masks for speed? adj is already bitmasks.
        # For each color round, find an independent set in the remaining graph.
        # A vertex v can be colored c if it is not adjacent to any vertex already colored c in this round.
        
        while uncolored:
            current_color += 1
            colored_in_round = []
            new_uncolored = []
            # We can optimize this by keeping a mask of "forbidden" vertices for the current color?
            # No, simple loop is fine for N=512.
            for v in uncolored:
                is_adjacent = False
                # Check if v is adjacent to any in colored_in_round
                # Instead of loop, we can maintain a bitmask of colored_in_round?
                # Let's just use the bitmask check.
                # But we need to check against the list.
                # Optimization: maintain a mask `round_mask` of vertices colored in this round.
                # If adj[v] & round_mask != 0, then adjacent.
                
                # Let's refactor the loop to use a mask for the current round
                pass # Placeholder for logic below
                
            # Re-implementing the loop with mask optimization
            round_mask = 0
            for v in uncolored:
                if adj[v] & round_mask:
                    new_uncolored.append(v)
                else:
                    colors.append((v, current_color))
                    round_mask |= (1 << v)
            uncolored = new_uncolored
            
        colors.sort(key=lambda x: x[1], reverse=True)
        return colors

    def expand(clique, P_mask):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
            
        # Pruning: if even taking all candidates doesn't beat best
        # We need a bound. Coloring gives a better bound.
        
        colored = color_vertices(P_mask)
        
        # Iterate through vertices in the order determined by coloring
        for v, c in colored:
            # Pruning bound:
            # If len(clique) + c <= best_size, we can break.
            if len(clique) + c <= best_size:
                break
                
            # Include v
            new_P = P_mask & adj[v]
            
            # Recurse
            expand(clique + [v], new_P)
            
            # Remove v from P_mask for the next iteration
            P_mask ^= (1 << v)
            
            # Check time
            if time.time() > deadline:
                return

    # Initial call
    # We can start with the greedy clique to skip some work?
    # No, standard Tomita starts with empty clique and full P.
    # But having a high best_size helps pruning.
    
    # To further speed up, we can order the initial P vertices by degree.
    # The coloring function does this internally.
    
    expand([], initial_P)
    
    if not best_clique:
        return [1]
        
    return sorted(best_clique)

# EVOLVE-BLOCK-END
