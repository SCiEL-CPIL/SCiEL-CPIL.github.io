import time
import random
import sys

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e not in edges and degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, adj):
        m = len(edges)
        # Map vertex to list of edge indices incident to it
        vert_to_edges = [[] for _ in range(max(adj.keys()) + 1 if adj else 1)]
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].append(i)
            vert_to_edges[v].append(i)
            
        # Initialize conflict bitmasks
        conflict_masks = [0] * m
        
        # 1. Shared vertices
        for u in range(len(vert_to_edges)):
            e_list = vert_to_edges[u]
            for i in range(len(e_list)):
                ei = e_list[i]
                mask = 0
                for j in range(i + 1, len(e_list)):
                    ej = e_list[j]
                    mask |= (1 << ej)
                    conflict_masks[ei] |= (1 << ej)
                    conflict_masks[ej] |= (1 << ei)
        
        # 2. Induced matching violations (joined by edge)
        edge_set = set(edges)
        for i, (u, v) in enumerate(edges):
            neighbors_u = adj.get(u, set())
            neighbors_v = adj.get(v, set())
            
            # Any edge (w, x) disjoint from (u,v) such that w or x is neighbor of u or v
            # We iterate j > i to avoid duplicates
            for j in range(i + 1, m):
                if (conflict_masks[i] >> j) & 1:
                    continue # Already conflict via shared vertex
                
                w, x = edges[j]
                if w in neighbors_u or x in neighbors_u or w in neighbors_v or x in neighbors_v:
                    conflict_masks[i] |= (1 << j)
                    conflict_masks[j] |= (1 << i)
        
        return conflict_masks

    def greedy_coloring(m, conflict_masks, seed=0):
        """
        Fast greedy coloring to get an initial upper bound.
        Returns list of colors.
        """
        if m == 0:
            return []
        
        # Order vertices by degree in conflict graph (descending)
        degrees = [bin(conflict_masks[i]).count('1') for i in range(m)]
        order = list(range(m))
        order.sort(key=lambda x: -degrees[x])
        
        assign = [-1] * m
        color_masks = [] # List of bitmasks for each color class
        
        for i in order:
            # Find first color c such that (conflict_masks[i] & color_masks[c]) == 0
            c = 0
            while c < len(color_masks):
                if (conflict_masks[i] & color_masks[c]) == 0:
                    break
                c += 1
            
            if c == len(color_masks):
                color_masks.append(0)
            
            color_masks[c] |= (1 << i)
            assign[i] = c
        
        return assign

    def solve_instance(n, d, seed, time_limit=30):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        conflict_masks = build_conflict_bitmasks(edges, adj)
        
        # Get initial upper bound
        init_assign = greedy_coloring(m, conflict_masks, seed)
        best_k = max(init_assign) + 1 if init_assign else 0
        
        # Sort vertices for B&B by degree descending
        degrees = [bin(conflict_masks[i]).count('1') for i in range(m)]
        order = list(range(m))
        order.sort(key=lambda x: -degrees[x])
        
        # Map original index to position in order
        pos_in_order = [0] * m
        for pos, orig_idx in enumerate(order):
            pos_in_order[orig_idx] = pos
        
        # B&B to minimize k
        # We try k from best_k - 1 down to 1
        # For each k, we run a DFS to see if a valid coloring exists.
        
        def can_color_with_k(k):
            """
            Returns True if graph can be colored with k colors.
            Uses B&B with symmetry breaking.
            """
            if k == 0:
                return m == 0
            
            color_masks = [0] * k
            assign = [-1] * m
            
            # Precompute sorted order of vertices to process
            # We process vertices in 'order' list (index 0 to m-1 in 'order')
            
            # To speed up, we can use a recursive function that takes the current position in 'order'
            # and the current color masks.
            
            sys.setrecursionlimit(10000)
            
            def dfs(pos):
                if pos == m:
                    return True
                
                v = order[pos]
                vm = conflict_masks[v]
                
                # Try colors 0 to k-1
                # Symmetry breaking: 
                # If this is the first vertex, it must be color 0.
                # In general, we can restrict colors to <= max_color_used_so_far + 1?
                # Simpler symmetry breaking: 
                # If vertex v has no neighbors colored with color c, and we haven't used color c+1 yet,
                # we can restrict?
                # Let's use: The first vertex (pos=0) must be color 0.
                # The second vertex (pos=1) can be color 0 or 1.
                # Generally, vertex at pos can use colors 0..min(pos, k-1).
                
                max_c = min(pos, k - 1)
                
                # Optimization: Check if any color is available
                # If no color is available, return False
                available = False
                for c in range(max_c + 1):
                    if (vm & color_masks[c]) == 0:
                        available = True
                        break
                
                if not available:
                    return False
                
                for c in range(max_c + 1):
                    if (vm & color_masks[c]) == 0:
                        color_masks[c] |= (1 << v)
                        assign[v] = c
                        if dfs(pos + 1):
                            return True
                        color_masks[c] ^= (1 << v)
                        assign[v] = -1
                return False
            
            return dfs(0)

        # Try to reduce k
        current_time = time.time()
        start_time = current_time
        
        # We want to find the minimum k.
        # We know it's <= best_k.
        # Try best_k - 1, best_k - 2, ...
        
        # To handle time, we set a per-k time limit or global.
        # If we run out of time, we keep the current best_k.
        
        # Strategy: Binary search? No, linear decrease is fine if k is small.
        # But if best_k is 45, and optimal is 20, linear is 25 steps.
        # Each step is a B&B.
        
        # Better: Try k = best_k - 1. If success, best_k = k. Repeat.
        # If fail, stop.
        
        while best_k > 1:
            if time.time() - start_time > time_limit:
                break
            
            k_to_try = best_k - 1
            
            # Optional: Lower bound check?
            # If max degree + 1 > k_to_try, impossible.
            max_deg = max(degrees)
            if max_deg + 1 > k_to_try:
                break
            
            # Run B&B
            # To avoid hanging on a single hard k, we can't easily interrupt recursion in Python without signals.
            # We rely on the time_limit check outside? No, recursion is blocking.
            # We need a time check inside dfs?
            # Let's add a simple counter or time check in dfs.
            
            # To be safe, we'll wrap the B&B in a way that we can abort if it takes too long.
            # Since Python doesn't have a built-in timeout for functions, we'll just hope it's fast.
            # For m < 300 and k < 50, it should be fast.
            
            # Let's add a global time check inside dfs by passing a deadline.
            deadline = time.time() + 5 # 5s per k attempt
            
            def dfs_timed(pos):
                if time.time() > deadline:
                    raise TimeoutError()
                if pos == m:
                    return True
                
                v = order[pos]
                vm = conflict_masks[v]
                max_c = min(pos, k_to_try - 1)
                
                available = False
                for c in range(max_c + 1):
                    if (vm & color_masks[c]) == 0:
                        available = True
                        break
                if not available:
                    return False
                
                for c in range(max_c + 1):
                    if (vm & color_masks[c]) == 0:
                        color_masks[c] |= (1 << v)
                        if dfs_timed(pos + 1):
                            return True
                        color_masks[c] ^= (1 << v)
                return False
            
            color_masks = [0] * k_to_try
            try:
                if dfs_timed(0):
                    best_k = k_to_try
            except TimeoutError:
                # Abort this k attempt, keep current best_k
                pass
            except RecursionError:
                pass

        # Construct final assignment using best_k
        # We need to actually produce the colors.
        # We can run a greedy coloring constrained to best_k?
        # Or run the B&B one last time for best_k and record the assignment?
        
        # Let's just use the greedy coloring result if it used <= best_k colors.
        # If greedy used more, we need to find a valid coloring with best_k.
        # Since we proved existence (or assumed it if we didn't disprove best_k-1),
        # we can run a greedy that tries to fit into best_k.
        
        # Greedy with fixed K
        def greedy_fixed_k(k):
            assign = [-1] * m
            color_masks = [0] * k
            
            for v in order: # Use the same order
                vm = conflict_masks[v]
                found = False
                for c in range(k):
                    if (vm & color_masks[c]) == 0:
                        color_masks[c] |= (1 << v)
                        assign[v] = c
                        found = True
                        break
                if not found:
                    # Should not happen if k is valid
                    # Assign arbitrary? No, this means k is not valid.
                    # But we assumed k is valid.
                    # For safety, add new color? No, we want exactly k.
                    # If this happens, the B&B was wrong or timeout hid a failure.
                    # Let's just return a potentially invalid coloring if we get stuck, 
                    # but the evaluator checks validity.
                    # So we should ensure validity.
                    # If greedy fails, we might need to backtrack?
                    # Let's just use the greedy result from the initial step if it was valid.
                    pass
            
            return assign

        # If best_k < initial greedy colors, we need a valid coloring with best_k.
        # The B&B proved existence. We can extract the coloring from the successful B&B run?
        # We didn't store it.
        # Let's re-run B&B for best_k and store the assignment.
        
        final_assign = [-1] * m
        color_masks_final = [0] * best_k
        
        def dfs_extract(pos):
            if pos == m:
                return True
            
            v = order[pos]
            vm = conflict_masks[v]
            max_c = min(pos, best_k - 1)
            
            for c in range(max_c + 1):
                if (vm & color_masks_final[c]) == 0:
                    color_masks_final[c] |= (1 << v)
                    final_assign[v] = c
                    if dfs_extract(pos + 1):
                        return True
                    color_masks_final[c] ^= (1 << v)
                    final_assign[v] = -1
            return False
        
        # Try to extract
        if not dfs_extract(0):
            # Fallback to initial greedy
            final_assign = init_assign
        
        return final_assign

    out = []
    deadline = time.time() + 115
    
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = deadline - time.time()
        if remaining < 10:
            # Fallback: Fast Greedy
            n, d, seed = spec
            edges = instance(n, d, seed)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            conflict_masks = build_conflict_bitmasks(edges, adj)
            out.append(greedy_coloring(m, conflict_masks, seed))
        else:
            t_limit = min(remaining - 5, 35)
            out.append(solve_instance(*spec, time_limit=t_limit))
    
    return out

if __name__ == "__main__":
    # Quick test
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(set(r)) if r else 0} colors")
