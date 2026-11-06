import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    N = 256
    # Vertices are 0..255 representing numbers 1..256
    # adj_mask[i] is a bitmask of neighbors of vertex i
    
    # 1. Build Graph
    # Square numbers up to 256+256 = 512. Squares: 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484.
    # Max square is 484 (22^2). 23^2 = 529 > 512.
    squares = [k*k for k in range(2, 23)] # 4 to 484
    
    adj_mask = [0] * N
    
    for i in range(N):
        val_i = i + 1
        for sq in squares:
            val_j = sq - val_i
            if 1 <= val_j <= N:
                j = val_j - 1
                if i != j: # Self loops are not edges in standard graph, but condition a+b square for a=b means 2a is square.
                    # The problem says "a+b is never a perfect square for a<=b in the set, INCLUDING a=b".
                    # This means if 2*a is a square, a cannot be in the set.
                    # In MIS formulation, if 2*a is square, vertex a is "forbidden" or we must ensure it's not picked.
                    # Actually, if 2*a is a square, then a+a is a square, so a cannot be in any valid set.
                    # So we can just remove such vertices from the candidate pool entirely?
                    # Yes. If 2*a is a square, a is incompatible with itself.
                    # So, pre-filter vertices where 2*(i+1) is a square.
                    pass
            # Note: The loop above adds edge (i,j). If i==j, we handle separately.
            # Let's refine:
            # If val_j == val_i (i.e., 2*val_i == sq), then vertex i is invalid on its own.
            # We can mask it out from the start.
            
    # Re-do graph construction properly
    valid_vertices = []
    for i in range(N):
        val = i + 1
        if 2 * val in squares:
            continue # This vertex can never be in the set
        valid_vertices.append(i)
        
    # Map original indices to compact indices? 
    # Using original indices 0..255 with masks is fine.
    
    adj_mask = [0] * N
    for i in range(N):
        val_i = i + 1
        if 2 * val_i in squares:
            continue # Skip invalid vertices
        for sq in squares:
            val_j = sq - val_i
            if 1 <= val_j <= N:
                j = val_j - 1
                if i != j:
                    # Check if j is valid
                    if 2 * (j+1) not in squares:
                        adj_mask[i] |= (1 << j)
                    # Note: We don't need to add edge i->j if j is invalid, but mask will handle it if we filter candidates.
                    # It's cleaner to only consider valid vertices in candidates.
                    
    # Initialize candidates mask with all valid vertices
    candidates = 0
    for i in valid_vertices:
        candidates |= (1 << i)
        
    # Greedy Initial Solution
    def greedy_mis(cand_mask):
        selected = 0
        mask = cand_mask
        while mask:
            # Pick vertex with lowest degree in remaining graph?
            # Or just pick first?
            # Let's try picking vertex with min degree to maximize remaining options.
            best_v = -1
            min_deg = N+1
            # Iterate through bits
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                # Degree of v in mask
                deg = (adj_mask[v] & mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                m ^= lsb
                
            if best_v == -1: break
            
            selected |= (1 << best_v)
            # Remove best_v and its neighbors
            mask &= ~(adj_mask[best_v] | (1 << best_v))
            
        return selected.bit_count()

    best_size = greedy_mis(candidates)
    best_set = 0 # Store bitmask of best set
    
    # To store best set, we need to track it.
    # Let's modify greedy to return the mask
    def greedy_mis_mask(cand_mask):
        selected = 0
        mask = cand_mask
        while mask:
            best_v = -1
            min_deg = N+1
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = (adj_mask[v] & mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                m ^= lsb
            if best_v == -1: break
            selected |= (1 << best_v)
            mask &= ~(adj_mask[best_v] | (1 << best_v))
        return selected

    best_set = greedy_mis_mask(candidates)
    best_size = best_set.bit_count()

    # Branch and Bound
    # State: current_mask, current_size, current_set_mask
    # Upper bound: coloring bound
    
    def coloring_bound(mask):
        # Greedy coloring of the subgraph induced by mask
        # Returns number of colors used
        uncolored = mask
        colors = 0
        while uncolored:
            colors += 1
            # Find a maximal independent set in uncolored to assign this color
            # Greedy MIS on uncolored
            color_set = 0
            m = uncolored
            while m:
                # Pick vertex with min degree in m?
                # For coloring, picking any vertex works, but we want a large independent set to reduce colors?
                # No, we want to assign this color to as many vertices as possible to minimize total colors.
                # So we want a Max Independent Set in the subgraph? That's hard.
                # Standard greedy coloring: pick a vertex, assign color, remove it and neighbors? No, that's for graph coloring.
                # For upper bound on MIS, we use the fact that $\alpha(G) \le \chi(G)$.
                # So we compute a valid coloring.
                # Standard greedy coloring:
                # While uncolored not empty:
                #   Pick a vertex v.
                #   Assign current color to v? No, we need to assign same color to non-adjacent vertices.
                #   Let's use the standard "coloring for MIS bound" algorithm:
                #   It constructs a sequence of colors.
                #   Actually, a simpler bound is just bit_count(mask).
                #   A better bound is the greedy coloring number.
                
                # Let's implement a simple greedy coloring that returns the number of colors.
                # This is used as an upper bound for the size of the MIS.
                
                pass # Placeholder, will implement below

    # Simpler Upper Bound: 
    # The size of the maximum independent set is at most the number of vertices in the clique cover?
    # Or just use a simple heuristic: 
    # Sort vertices by degree, pick one, remove neighbors, count. This is a lower bound on MIS, not upper.
    # Upper bound: 
    # $\alpha(G) \le \sum_{v \in V} \frac{1}{d(v)+1}$? No.
    # Let's use the coloring bound properly.
    
    def get_coloring_bound(mask):
        # Returns an upper bound on the size of the MIS in the subgraph induced by mask
        # Using greedy coloring
        uncolored = mask
        colors = 0
        while uncolored:
            colors += 1
            # Assign current color to a maximal independent set in uncolored
            # To make the bound tight, we want to use few colors, so we want large independent sets per color.
            # Greedy Max Independent Set heuristic:
            # Pick vertex with min degree?
            m = uncolored
            while m:
                # Pick vertex v with min degree in m
                best_v = -1
                min_deg = N+1
                mm = m
                while mm:
                    lsb = mm & -mm
                    v = lsb.bit_length() - 1
                    deg = (adj_mask[v] & m).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                    mm ^= lsb
                if best_v == -1: break
                # Add best_v to color set
                # Remove best_v and its neighbors from uncolored?
                # No, in coloring, we assign color to v, then we must remove v and its neighbors from the "available for this color" set?
                # Yes. The vertices that get this color form an independent set.
                # So we pick v, add to color_set, remove v and N(v) from m (the pool for this color).
                # Then when m is empty, we move to next color.
                # Wait, this is just finding a Max Independent Set?
                # If we find a Max Independent Set, we assign it color 1.
                # Then remove it from graph, find next Max Independent Set, assign color 2.
                # The number of colors is the bound.
                # This is exactly the greedy coloring algorithm that produces a valid coloring.
                
                uncolored &= ~(adj_mask[best_v] | (1 << best_v))
                m &= ~(adj_mask[best_v] | (1 << best_v))
            
        return colors

    # This coloring_bound is expensive.
    # Let's try a simpler bound first: bit_count(mask).
    # And use the coloring bound only when bit_count(mask) + current_size > best_size.
    
    def dfs(mask, current_size, current_set):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Upper bound
        ub = mask.bit_count()
        if current_size + ub <= best_size:
            return
            
        # Tighter bound: coloring
        # Only compute if simple bound doesn't prune
        # Coloring bound is expensive, so maybe use it selectively?
        # For now, let's rely on good vertex ordering and simple bound, or a fast coloring bound.
        
        # Vertex ordering: pick vertex with max degree?
        # In B&B for MIS, picking a vertex with high degree often prunes more.
        # Let's pick vertex with max degree in mask.
        best_v = -1
        max_deg = -1
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (adj_mask[v] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            m ^= lsb
            
        if best_v == -1: return
        
        # Branch 1: Include best_v
        new_mask_1 = mask & ~(adj_mask[best_v] | (1 << best_v))
        dfs(new_mask_1, current_size + 1, current_set | (1 << best_v))
        
        # Branch 2: Exclude best_v
        new_mask_2 = mask & ~(1 << best_v)
        dfs(new_mask_2, current_size, current_set)
        
    # Start DFS
    dfs(candidates, 0, 0)
    
    # Convert best_set bitmask to list of numbers
    result = []
    m = best_set
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb
        
    return result
