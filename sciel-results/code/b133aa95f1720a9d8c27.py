import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    time_limit = 110  # leave buffer
    
    # 1. Generate support-size-3 candidates
    candidates3 = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates3.append(v)
    candidates3 = np.array(candidates3, dtype=np.int64)
    n3 = len(candidates3)  # 1320
    
    # 2. Compute dot products and build conflict/compatibility graph
    # Compatibility: dot <= 1
    dots3 = candidates3 @ candidates3.T
    # We want Max Independent Set in Conflict Graph (dot >= 2)
    # Which is Max Clique in Compatibility Graph (dot <= 1)
    compat_matrix = (dots3 <= 1)
    np.fill_diagonal(compat_matrix, False) # Self not compatible with self for clique definition (distinct vertices)
    
    # 3. Tomita Maximum Clique Algorithm
    # adj_list[i] contains neighbors of i in compat graph
    # Using list of sets for flexibility, or boolean matrix for speed?
    # Boolean matrix is faster for checking adjacency.
    
    best_clique = []
    best_size = 0
    
    # Precompute degrees for initial ordering
    degrees = compat_matrix.sum(axis=1)
    
    # Tomita's expand function
    # R: current clique (list of indices)
    # P: candidate set (list of indices)
    # depth: current depth
    
    def expand(R, P, depth):
        nonlocal best_size, best_clique
        if time.time() - start_time > time_limit:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R.copy()
            return
            
        # If upper bound is not better, prune
        # Upper bound: len(R) + len(P) is weak. Use coloring.
        
        # Greedy Coloring of P
        # Order P by degree in P (descending)
        # To speed up, we can just use the global degree or compute local.
        # For N=1320, computing local degree in P for each step is expensive if done naively.
        # Tomita uses a specific ordering.
        
        # Simplified Tomita:
        # 1. Sort P by degree in the induced subgraph (approx by global degree or recompute)
        # 2. Color the sorted P.
        # 3. Process vertices in reverse color order.
        
        # Optimization: If len(R) + len(P) <= best_size, return
        if len(R) + len(P) <= best_size:
            return
            
        # Greedy coloring
        # We need to color the graph induced by P.
        # Sort P by degree in P.
        # deg_in_P = [sum(compat_matrix[p, q] for q in P) for p in P]
        # This is O(|P|^2).
        
        # To make it faster, we can use a heuristic order.
        # Let's try sorting by global degree first.
        P_sorted = sorted(P, key=lambda x: -degrees[x])
        
        colors = [0] * len(P_sorted)
        color_map = {v: 0 for v in P_sorted} # Map vertex to color index (1-based)
        
        # Greedy coloring
        # For each vertex v in P_sorted:
        # Assign smallest color not used by neighbors in P_sorted that have been colored.
        
        # To do this efficiently, we track used colors per vertex?
        # Or just check neighbors.
        
        # Standard greedy coloring:
        # for i, v in enumerate(P_sorted):
        #    used_colors = set()
        #    for u in P_sorted:
        #        if u == v: continue
        #        if compat_matrix[v, u]:
        #             # u is neighbor
        #             # If u has been colored (index < i), add its color
        #             # This requires knowing which ones are colored.
        #             # Since we iterate i, only j < i are colored.
        #             pass
        #    # This is O(N^2) per vertex -> O(N^3) total. Too slow for 1320?
        # 1320^3 is ~ 2e9. Too slow in Python.
        
        # We need a faster coloring or a different bound.
        # Alternative: Use a simple bound len(R) + len(P) and hope the search space is small?
        # Or use a better coloring algorithm.
        
        # Tomita's algorithm uses a specific coloring that is linear in the number of edges if implemented with bitsets, but in Python...
        
        # Let's try a simpler approach first:
        # Use the bound len(R) + len(P).
        # And use pivot selection to reduce branching.
        
        # Pivot selection:
        # Choose u in P with max degree in P.
        # Partition P into P1 = P \ N(u) and P2 = P \cap N(u).
        # Branch on P1 first? Or P2?
        # Tomita branches on P \ N(u) then P \cap N(u)?
        # Actually, Tomita's MCQ algorithm:
        # 1. Color P.
        # 2. For each vertex v in P in reverse color order:
        #    if len(R) + color[v] <= best_size: break
        #    R.append(v)
        #    P_new = P \ {v} intersect N(v)
        #    expand(R, P_new, depth+1)
        #    R.pop()
        #    P.remove(v)
        
        # The coloring is the key.
        # Let's implement a fast enough greedy coloring.
        # Since N=1320, maybe we can just use the simple bound and rely on pivots?
        # No, coloring is crucial for Tomita.
        
        # Let's try to implement coloring with NumPy vectorization?
        # Hard for sequential greedy coloring.
        
        # Alternative: Use a library? No, standard lib only.
        
        # Let's try a simpler bound:
        # If we don't use coloring, we can still use pivots.
        # But without coloring, the bound is loose.
        
        # Let's try to optimize the coloring step.
        # For each v in P_sorted:
        #   neighbors_in_P = [u for u in P if compat_matrix[v,u]]
        #   colors_used = {color[u] for u in neighbors_in_P if u in color_map}
        #   assign smallest color not in colors_used.
        
        # We can precompute neighbors for each v in P?
        
        # Given time constraints, let's try the simple bound first with pivots.
        # If it's too slow, we might timeout.
        
        # Let's implement the Pivot + Simple Bound version.
        
        # Pivot selection
        # Find u in P with max degree in P
        # deg_in_P = np.array([np.sum(compat_matrix[p, P]) for p in P])
        # This is O(|P| * |P|) if done naively.
        
        # Let's just use global degrees for pivot selection as an approximation.
        # u = P[argmax(degrees[P])]
        
        # Partition P into P1 (not connected to u) and P2 (connected to u)
        # P1 = [v for v in P if not compat_matrix[u, v]]
        # P2 = [v for v in P if compat_matrix[u, v]]
        
        # Branch on P1 first (usually smaller or leads to faster pruning?)
        # Tomita usually processes the whole P with coloring.
        
        # Let's try to implement a basic coloring.
        # To speed up, we'll use a list of sets for neighbors?
        
        # Let's switch strategy:
        # Use `networkx`? No, not allowed.
        
        # Let's try a recursive search with:
        # 1. Bound: len(R) + len(P)
        # 2. Pivot: Max degree in P (approx by global)
        # 3. Order: Descending global degree
        
        # This might be slow.
        
        # Let's try to improve the coloring.
        # We can store `P` as a set.
        
        # Revised Plan:
        # Implement Tomita with a simplified coloring that is O(N^2) per node.
        # With N=1320, and pruning, hopefully it's fast enough.
        
        # Coloring:
        P_list = list(P)
        # Sort by degree in P
        # To avoid O(N^2) degree calc, use global degree as proxy.
        P_list.sort(key=lambda x: -degrees[x])
        
        color_of = {}
        max_color = 0
        
        # Greedy coloring
        for v in P_list:
            # Find colors of neighbors in P_list that are already colored
            # Since we iterate in order, neighbors "already colored" are those earlier in P_list?
            # No, greedy coloring doesn't require order, but sequential is standard.
            # If we process v, we check neighbors u that have been assigned a color.
            # In a sequential pass, only previously visited vertices have colors.
            
            used_colors = set()
            for u in P_list:
                if u == v:
                    continue
                if u in color_of: # Only consider colored neighbors
                    if compat_matrix[v, u]:
                        used_colors.add(color_of[u])
                        
            # Assign smallest available color
            c = 1
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
                
        # Upper bound is max_color
        if len(R) + max_color <= best_size:
            return
            
        # Process vertices in reverse order of color (highest color first)
        # This is a heuristic to find large cliques early.
        # We need to process P such that we check vertices with high color first.
        
        # Create a list of (vertex, color)
        P_with_color = [(v, color_of[v]) for v in P_list]
        P_with_color.sort(key=lambda x: -x[1])
        
        # We need to update P as we go.
        # Tomita's algorithm removes v from P after branching.
        
        # To do this correctly, we iterate through the sorted list.
        # But P changes.
        # Standard Tomita:
        # for v in P (ordered by color):
        #    if bound fails: break
        #    expand(R + [v], P intersect N(v))
        #    P.remove(v)
        
        # We'll simulate this.
        # We need to be careful with modifying P while iterating.
        # We'll use a list `P_current` that we shrink.
        
        # However, the coloring was done on the initial P.
        # As we remove vertices, the bound decreases.
        
        # Let's implement the loop.
        # Note: P is a set.
        
        # We need to iterate in the order determined by color.
        # But we remove from P.
        
        # Let's create a list of vertices to process in order.
        # process_order = [v for v, c in P_with_color]
        
        # But we must skip vertices already removed.
        
        # Actually, Tomita's algorithm is usually implemented with a specific data structure.
        # In Python, we can just use a list.
        
        # Let's try this:
        # 1. Sort P by color descending.
        # 2. Iterate through this sorted list.
        # 3. If v is not in P, skip.
        # 4. Check bound: len(R) + color[v] <= best_size? (Color is upper bound for remaining P? No, color[v] is the color assigned to v. The bound for the subproblem starting with v is len(R) + color[v]. Wait.
        # The coloring bound says: The clique in P has size at most max_color.
        # More specifically, if we process vertices in order of decreasing color, and we are at vertex v with color c, then the remaining vertices (including v) have colors <= c? No, they have colors <= max_color, but the vertices processed later have lower colors?
        # Actually, the property is: If we color P, and process vertices in decreasing color order, then when we consider vertex v with color c, the number of vertices remaining in P that can form a clique with v is bounded by c?
        # Yes, Tomita's proof relies on this.
        
        # So:
        for v, c in P_with_color:
            if v not in P:
                continue
            if len(R) + c <= best_size:
                break # Since colors are decreasing, further vertices have lower c, so bound will be even smaller.
            
            # Branch
            R.append(v)
            # P_new = P \ {v} intersect N(v)
            # N(v) = {u in P if compat_matrix[v, u]}
            # We can compute this by iterating P
            P_new = set()
            for u in P:
                if u != v and compat_matrix[v, u]:
                    P_new.add(u)
            
            expand(R, P_new, depth + 1)
            
            R.pop()
            P.remove(v)
            
            # Update bound? The max_color of the new P is <= c-1?
            # Not necessarily, but the bound check at the start of expand will handle it.
            # However, we can break if len(R) + (c-1) <= best_size?
            # The next vertex has color <= c-1.
            # So if len(R) + c - 1 <= best_size, we can break?
            # Actually, the loop condition `if len(R) + c <= best_size` uses the current c.
            # After removing v, the next vertex has color c' <= c-1.
            # So the check at the top of the next iteration will handle it.
            
    # Initialize
    # Start with all vertices
    P_initial = set(range(n3))
    expand([], P_initial, 0)
    
    # Check if we found a better solution than baseline (220)
    if best_size > 220:
        result = candidates3[best_clique]
        return result
    else:
        # Return support-2 vectors
        candidates2 = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    candidates2.append(v)
        return np.array(candidates2, dtype=np.int64)
