import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build allowed masks
    # allowed_mask[i] is a bitmask where bit j is 1 if i and j can be in the same set
    # i.e., (i-j) % m is NOT in forbidden
    allowed_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 not in forbidden and d2 not in forbidden:
                mask |= (1 << j)
        allowed_mask[i] = mask
        
    # Initial greedy solution to set a good lower bound
    def greedy_solve():
        best = []
        candidates = (1 << m) - 1
        while candidates:
            # Pick vertex with max degree in candidate subgraph
            # Approximation: pick vertex with max popcount(allowed_mask[v] & candidates)
            max_deg = -1
            v = -1
            temp = candidates
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                deg = (allowed_mask[idx] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                temp &= temp - 1
            if v == -1:
                break
            best.append(v)
            candidates &= allowed_mask[v]
        return best, len(best)
        
    init_sol, init_size = greedy_solve()
    best_size = init_size
    best_clique = init_sol[:]
    
    # Exact solver
    # State: clique (list), candidates (int mask)
    # Time limit
    start_time = time.time()
    time_limit = 590  # seconds
    
    # We use a global variable to track best
    # To avoid deep recursion limits, we might need to increase recursion limit or use iterative deepening?
    # N=205, max depth 205. Default limit 1000 is fine.
    
    import sys
    sys.setrecursionlimit(3000)
    
    def expand(clique, candidates):
        nonlocal best_size, best_clique, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False # Signal timeout
            
        if candidates == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return True
            
        # Upper bound: size of max clique in candidates
        # Use greedy coloring to get upper bound
        # The number of colors in a greedy coloring is an upper bound on clique size
        
        # We compute an ordering of vertices in candidates to help coloring?
        # Standard Tomita uses a specific ordering.
        # For simplicity, we use a simple greedy coloring on the subgraph.
        
        # To speed up coloring, we can order vertices by degree descending?
        # Let's just do a simple greedy coloring.
        
        # Note: We need to ensure we don't do too much work in the bound calculation.
        
        def color_bound(mask):
            # Returns upper bound on clique size in mask
            colors = 0
            uncovered = mask
            while uncovered:
                colors += 1
                # Greedy independent set
                indep = 0
                temp = uncovered
                # Order vertices? Let's pick lowest index first for determinism and speed
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    # Add idx to independent set
                    indep |= lsb
                    # Remove idx and its neighbors from temp
                    # Neighbors in the Allowed graph are allowed_mask[idx]
                    # We want to remove vertices that are NOT compatible?
                    # No, in Coloring for Clique, we color the graph such that no two adjacent vertices have the same color.
                    # So if we pick v, we cannot pick any neighbor of v in the SAME color.
                    # So we remove v and its neighbors from 'temp' (the set of vertices available for this color).
                    temp &= ~allowed_mask[idx]
                    temp &= ~lsb
                uncovered &= ~indep
            return colors

        ub = color_bound(candidates)
        
        if len(clique) + ub <= best_size:
            return True
            
        # Branching
        # Select a vertex to branch on.
        # Heuristic: Pick vertex with highest degree in the candidate subgraph?
        # Or simply the first one?
        # Tomita suggests picking a vertex from the last color class?
        # Let's pick the vertex with the highest degree in the subgraph induced by candidates.
        
        max_deg = -1
        v = -1
        temp = candidates
        # Optimization: if ub is small, we might not need to scan all
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = (allowed_mask[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp &= temp - 1
            
        if v == -1:
            return True
            
        # Branch 1: Include v
        new_clique = clique + [v]
        new_candidates = candidates & allowed_mask[v]
        
        if expand(new_clique, new_candidates):
            return True # Timeout
            
        # Branch 2: Exclude v
        candidates &= ~(1 << v)
        
        if expand(clique, candidates):
            return True # Timeout
            
        return True

    expand([], (1 << m) - 1)
    
    # Return the best clique found
    return np.array(best_clique, dtype=np.int64)
