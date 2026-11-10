import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # 1. Compute forbidden differences (nonzero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency list
    # neighbors[v] is a set of vertices u such that (u-v)%m is in forbidden
    neighbors = [set() for _ in range(m)]
    for u in range(m):
        for v in range(m):
            if u == v:
                continue
            if (u - v) % m in forbidden:
                neighbors[u].add(v)
    
    # Convert to lists for faster iteration in Tomita
    adj_list = [list(neighbors[i]) for i in range(m)]
    
    # 3. Tomita Algorithm for Maximum Independent Set
    # We seek the largest subset of vertices such that no two are adjacent in this graph.
    # This is equivalent to finding the Maximum Independent Set in the "forbidden difference" graph.
    
    best_set = []
    best_size = 0
    
    # Use a simple coloring heuristic to bound the search.
    # In Tomita's algorithm, we order vertices and use a greedy coloring of the subgraph induced by candidates.
    # The color number gives an upper bound on the size of the independent set that can be added.
    
    def tomita(R, C):
        """
        R: current independent set (list)
        C: candidate set (list)
        """
        nonlocal best_set, best_size
        
        if not C:
            if len(R) > best_size:
                best_size = len(R)
                best_set = R.copy()
            return
        
        # If the upper bound (len(C) + len(R)) is <= best_size, we can prune
        # But we need a tighter bound from coloring.
        
        # Greedy coloring of C to get upper bound
        # We color vertices in C such that no two adjacent vertices have the same color.
        # The number of colors used is an upper bound on the size of a maximal independent set in C.
        colors = [0] * len(C)
        color_map = {} # vertex -> color index
        max_color = 0
        
        # Simple greedy coloring
        for i, v in enumerate(C):
            # Find smallest color not used by neighbors of v in C
            used_colors = set()
            for u in adj_list[v]:
                if u in color_map:
                    used_colors.add(color_map[u])
            
            c = 0
            while c in used_colors:
                c += 1
            colors[i] = c
            color_map[v] = c
            if c > max_color:
                max_color = c
        
        # Upper bound: len(R) + max_color
        if len(R) + max_color <= best_size:
            return
        
        # Order C by color (highest color first) to improve pruning
        # Actually, standard Tomita orders by color in the reverse order of assignment 
        # or uses a specific expansion. 
        # A common variant: Sort C by color descending.
        # Let's sort indices by color descending.
        c_indices = list(range(len(C)))
        c_indices.sort(key=lambda i: colors[i], reverse=True)
        
        # We can also prune C if the color of the last vertex in sorted order + len(R) <= best_size
        # But let's just iterate.
        
        # To ensure we find the *set*, we need to backtrack.
        # Tomita is typically used for size. To get the set, we need to keep track of choices.
        
        # Since m=205 is small, a recursive backtracking with good pruning should work.
        # However, implementing full Tomita with set recovery is complex.
        # Given the small size, a simpler branch and bound might suffice if optimized.
        
        # Let's use a standard recursive MIS solver with bitmasks or sets.
        # With 205 nodes, sets are okay but bitmasks are faster.
        # Let's stick to sets for clarity unless it's too slow. 600s is plenty.
        
        pass # This function is a stub, I will implement a simpler robust solver below.

    # Alternative: Use a standard backtracking MIS solver with greedy coloring bound.
    # Since 205 is small, even a slightly suboptimal bound will finish quickly.
    
    def solve_mis():
        nonlocal best_set, best_size
        # Order vertices by degree (descending) to help pruning? 
        # Or just use 0..204.
        
        def expand(R, C):
            nonlocal best_set, best_size
            if not C:
                if len(R) > best_size:
                    best_size = len(R)
                    best_set = R.copy()
                return
            
            # Coloring bound
            # Greedy coloring of C
            colors = {}
            max_color = 0
            # Process C in some order. Let's say natural order of C.
            for v in C:
                used = set()
                for u in adj_list[v]:
                    if u in colors:
                        used.add(colors[u])
                c = 0
                while c in used:
                    c += 1
                colors[v] = c
                if c > max_color:
                    max_color = c
            
            if len(R) + max_color <= best_size:
                return
            
            # Sort C by color descending
            sorted_C = sorted(C, key=lambda x: colors[x], reverse=True)
            
            # Branch on the first vertex
            # We can choose any vertex. Choosing one with high color is often good.
            # But we must ensure we explore all possibilities? No, we just need *a* maximum set.
            # Wait, branch and bound explores all branches that *could* beat the current best.
            # So we pick a vertex, include it, and exclude its neighbors.
            
            # To maximize pruning, we should pick a vertex that is "critical".
            # Let's just pick the first one in sorted_C.
            v = sorted_C[0]
            
            # Branch 1: Include v
            new_R = R + [v]
            # New C is C without v and its neighbors
            new_C = [u for u in C if u != v and u not in neighbors[v]]
            expand(new_R, new_C)
            
            # Branch 2: Exclude v
            # We don't strictly need to branch on exclude if we just want *a* max set?
            # No, MIS is NP-hard. We must explore both.
            # However, if we just want the size, Tomita is efficient.
            # For the set, we need to be careful.
            # Actually, if we always include the first vertex in the sorted list, we might miss the max set
            # if the max set doesn't contain that vertex.
            # So we MUST branch on exclude.
            
            new_C2 = [u for u in C if u != v]
            expand(R, new_C2)

        expand([], list(range(m)))

    start_time = time.time()
    deadline = start_time + 590.0
    
    # Run the solver
    solve_mis()
    
    # If for some reason the solver didn't finish (unlikely for 205), return best found
    if not best_set:
        # Fallback to greedy if MIS fails or times out (though it shouldn't)
        available = set(range(m))
        selected = []
        while available:
            min_deg = float('inf')
            candidates = []
            for v in available:
                deg = len(neighbors[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            v = candidates[0]
            selected.append(v)
            available.discard(v)
            for n in neighbors[v]:
                available.discard(n)
        best_set = selected

    return np.array(best_set, dtype=np.int64)
