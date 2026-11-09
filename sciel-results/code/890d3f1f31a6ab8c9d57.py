import time
from math import gcd

def construct():
    """
    Finds a large subset of [1, 512] such that lcm(a, b) <= 512 for all pairs.
    Uses the Tomita algorithm for Maximum Clique with bitset optimization.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 512
    nodes = list(range(1, N + 1))
    
    # 1. Build adjacency bitmasks
    # adj[i] will be an integer where bit j is 1 if i and j are connected.
    # We index 1..512, so we need 513 bits.
    # Bit 0 is unused or for node 0 (not in range).
    # Node i corresponds to bit (i-1) or bit i? Let's use bit (i-1) for node i to keep indices 0-based for bits.
    # Or simpler: bit i for node i. We need up to bit 512.
    # Python ints handle arbitrary size, so this is fine.
    
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # lcm(i, j) = i * j // gcd(i, j)
            g = gcd(i, j)
            l = (i * j) // g
            if l <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Also, a node is always a "neighbor" of itself in the context of clique expansion?
    # No, usually we define P as candidates. If we pick v, new P is (P \ {v}) & adj[v].
    # This requires v to be connected to all in new P.
    # It does not require v to be connected to itself.
    # So adj[i] should NOT have bit i set.
    
    # 2. Tomita Algorithm
    best_clique = []
    best_size = 0
    
    # Initial candidate set: all nodes
    # We represent P as a bitmask.
    initial_P = 0
    for i in range(1, N + 1):
        initial_P |= (1 << i)
        
    # Helper to get list of vertices from bitmask
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        return verts

    # Helper to greedily color vertices in P
    # Returns list of (vertex, color) sorted by color descending
    def color_vertices(P_mask):
        # We need to assign colors such that adjacent vertices have different colors.
        # Greedy approach:
        # 1. Order vertices by degree in P (descending) to get better bound? 
        #    Or just iterate.
        # 2. Assign smallest available color.
        
        verts = get_vertices(P_mask)
        if not verts:
            return []
            
        # Precompute degrees within P for sorting?
        # For speed, let's just sort by global degree or index.
        # Actually, the order of coloring affects the tightness of the bound.
        # A common heuristic is to process vertices with higher degree first.
        
        # Let's calculate degree in P for each vertex
        degs = []
        for v in verts:
            # Count bits in (adj[v] & P_mask)
            degs.append((v, (adj[v] & P_mask).bit_count()))
            
        # Sort by degree descending
        degs.sort(key=lambda x: x[1], reverse=True)
        
        colors = [] # list of (vertex, color)
        uncolored = [v for v, _ in degs]
        current_color = 0
        
        while uncolored:
            current_color += 1
            colored_in_round = []
            new_uncolored = []
            for v in uncolored:
                # Check if v is adjacent to any in colored_in_round
                # If not, we can assign current_color
                is_adjacent = False
                for u in colored_in_round:
                    if adj[v] & (1 << u):
                        is_adjacent = True
                        break
                if not is_adjacent:
                    colors.append((v, current_color))
                    colored_in_round.append(v)
                else:
                    new_uncolored.append(v)
            uncolored = new_uncolored
            
        # Sort colors by color descending, then by vertex index (or whatever)
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
        # We need a bound. The number of vertices in P is a weak bound.
        # Coloring gives a better bound.
        
        colored = color_vertices(P_mask)
        
        # If max color + current size <= best, we can return immediately?
        # The max color is the last color assigned in the greedy process?
        # No, the bound for the entire set is the number of colors used.
        # But in Tomita, we iterate through vertices.
        
        # Iterate through vertices in the order determined by coloring
        for v, c in colored:
            # Pruning bound:
            # The maximum size of a clique we can form starting with 'clique' and including 'v'
            # is bounded by len(clique) + c.
            # Why? Because in the coloring, v has color c. This means there are at most c-1 vertices 
            # with colors < c that could be added? 
            # Actually, the standard Tomita bound is:
            # If we process vertices in order of decreasing color, 
            # and for a vertex v with color k, we know that any clique containing v 
            # can have at most k-1 other vertices from the remaining candidates 
            # that have colors < k? 
            # No, the bound is: len(clique) + c <= best_size -> prune.
            
            if len(clique) + c <= best_size:
                # Since we are processing in decreasing order of c, 
                # all subsequent vertices will have c' <= c.
                # So len(clique) + c' <= len(clique) + c <= best_size.
                # We can break the loop entirely.
                break
                
            # Include v
            new_P = P_mask & adj[v]
            # Remove v from new_P? 
            # adj[v] does not contain v (bit v is 0 in adj[v]).
            # So new_P = P_mask & adj[v] already excludes v.
            
            # Recurse
            expand(clique + [v], new_P)
            
            # Remove v from P_mask for the next iteration
            P_mask ^= (1 << v)
            
            # Optional: Early stop if best_size == N? Not possible here.
            # Optional: Check time
            
    # Initial call
    # Sort initial vertices? 
    # The coloring function sorts them.
    # We can start with the full set.
    expand([], initial_P)
    
    # Fallback: If B&B didn't find anything (shouldn't happen), return single largest element?
    # Or just return [1]
    if not best_clique:
        return [1]
        
    return sorted(best_clique)

# EVOLVE-BLOCK-END
