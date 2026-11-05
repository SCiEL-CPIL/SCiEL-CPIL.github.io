import time
from random import Random

def construct():
    start_time = time.time()
    time_limit = 100.0  # Leave 20s margin for 120s limit

    def generate_graph(n, p, seed):
        rng = Random(seed)
        # adj[u] is a bitmask of neighbors
        adj = [0] * n
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.add((u, v))
        return adj, edges

    def max_clique(adj, n, time_limit, start_time):
        """
        Finds a maximum clique in the graph defined by adj (list of bitmasks).
        Uses Tomita's algorithm with coloring.
        Returns the clique as a list of vertices.
        """
        best_clique = []
        best_size = 0
        
        # Greedy coloring of a set of vertices P (bitmask)
        # Returns a list of colors assigned to vertices in P, and the max color used
        # We need to order vertices by color to prune.
        # This function returns (ordered_vertices, colors) where ordered_vertices is a list of vertices in P
        # sorted by color ascending, and colors[i] is the color of ordered_vertices[i].
        
        def color_sort(P):
            if P == 0:
                return [], []
            
            colors = []
            ordered = []
            remaining = P
            color = 0
            
            while remaining:
                color += 1
                # Find independent set in remaining for this color class
                # Greedy: pick vertex, remove it and its neighbors
                # Actually, standard coloring for max clique bound:
                # We want to assign the smallest possible color to each vertex.
                # Let's do a simple greedy coloring.
                
                # We need to process vertices in some order.
                # Let's just iterate through bits.
                # To make it efficient, we can use a list of vertices in P.
                verts = []
                temp = P
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    temp ^= lsb
                
                # Greedy coloring
                assigned = {}
                current_color = 0
                # We want to minimize colors? No, we want a valid coloring to get an upper bound.
                # Any valid coloring gives an upper bound. Greedy is fast.
                # To get a tighter bound, we might want to order vertices by degree in P.
                # But for simplicity, let's just do sequential greedy.
                
                # Actually, the standard Tomita implementation colors the candidates P.
                # It processes vertices one by one.
                
                # Let's restart with a simpler approach:
                # We will maintain a list of (vertex, color).
                # We want to sort P by color.
                
                # Let's use a standard greedy coloring routine.
                # P is a bitmask.
                uncolored = P
                color_map = {}
                c = 0
                while uncolored:
                    c += 1
                    # Find a vertex v in uncolored that is not adjacent to any vertex already colored c in this pass?
                    # No, that's independent set.
                    # We are coloring the graph induced by P.
                    # We want to assign colors such that adjacent vertices have different colors.
                    # Greedy:
                    # Pick v from uncolored. Assign color c.
                    # Remove v and its neighbors in uncolored from consideration for color c.
                    # Wait, this finds a maximal independent set for color c.
                    
                    # Let's implement:
                    # 1. Pick a vertex v from uncolored.
                    # 2. Assign color c to v.
                    # 3. Remove v and all its neighbors in uncolored from uncolored.
                    # 4. Repeat until uncolored is empty for this color c.
                    # 5. Increment c.
                    
                    # To make it deterministic and potentially better, pick vertex with highest degree in uncolored?
                    # Let's just pick the lowest index for speed, or highest degree for tighter bound.
                    # Highest degree is better for pruning.
                    
                    # Calculate degrees in uncolored
                    # This is O(n^2) per color step. n=64 is small.
                    
                    # Let's just iterate.
                    temp_uncolored = uncolored
                    while temp_uncolored:
                        # Find vertex with max degree in uncolored
                        max_deg = -1
                        best_v = -1
                        t = uncolored
                        while t:
                            lsb = t & -t
                            v = lsb.bit_length() - 1
                            # Degree of v in uncolored
                            deg = (adj[v] & uncolored).bit_count()
                            if deg > max_deg:
                                max_deg = deg
                                best_v = v
                            t ^= lsb
                        
                        if best_v == -1: break # Should not happen if uncolored != 0
                        
                        # Assign color c to best_v
                        color_map[best_v] = c
                        # Remove best_v and its neighbors from uncolored
                        neighbors = adj[best_v]
                        uncolored &= ~(1 << best_v)
                        uncolored &= ~neighbors
                        temp_uncolored = uncolored # update loop var? No, we are in a while loop based on uncolored
                        # Actually, the inner loop condition was temp_uncolored, but we modified uncolored.
                        # Let's restructure.
                        
                        break # Pick one, remove it and neighbors, then restart the "pick max degree" for the new uncolored?
                        # No, the standard greedy coloring picks a vertex, assigns color, removes it and neighbors, then repeats for the same color.
                        # So the loop `while temp_uncolored` is correct if we update temp_uncolored = uncolored.
                        
                    # Wait, the logic above is a bit messy. Let's write a clean helper.
                    
                    break # Placeholder, let's rewrite color_sort properly.
            
            # Let's rewrite color_sort cleanly
            def color_sort_clean(P):
                if P == 0:
                    return [], []
                
                verts = []
                temp = P
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    temp ^= lsb
                
                # Sort vertices by degree in P descending (greedy heuristic for tighter coloring)
                verts.sort(key=lambda v: (adj[v] & P).bit_count(), reverse=True)
                
                colors = [0] * len(verts)
                color_assign = {}
                current_color = 0
                available = set(verts)
                
                while available:
                    current_color += 1
                    # Find maximal independent set in available
                    # Greedy: pick vertex with max degree in available, add to independent set, remove it and neighbors
                    # Wait, independent set means no edges between them.
                    # So if we pick v, we cannot pick any neighbor of v.
                    
                    # To get a valid coloring, we assign the same color to an independent set.
                    # We want to cover all vertices with minimum colors? No, we just need *a* coloring.
                    # Greedy coloring:
                    # For each vertex, assign the smallest available color.
                    # This is O(V^2).
                    
                    # Let's do:
                    # 1. Iterate vertices in sorted order.
                    # 2. For each v, find the smallest color c such that no neighbor of v in already colored vertices has color c.
                    
                    colored = {}
                    # Map vertex to color
                    # We need to check neighbors that are already colored.
                    
                    # Since we sorted by degree, let's just do this.
                    for v in verts:
                        # Find used colors by neighbors
                        used_colors = set()
                        # Neighbors of v in P
                        neighbors = adj[v] & P
                        # Check which neighbors are already colored
                        for u in verts:
                            if u == v: continue
                            if (1 << u) & neighbors:
                                if u in colored:
                                    used_colors.add(colored[u])
                        
                        # Find smallest color not in used_colors
                        c = 1
                        while c in used_colors:
                            c += 1
                        colored[v] = c
                        colors[verts.index(v)] = c # This is O(N) lookup, slow. Use dict.
                        
                # This is getting complicated and slow in Python.
                # Let's use a simpler bound: just return the number of vertices in P as the bound?
                # That's weak.
                
                # Alternative: Use a precomputed maximum clique size bound? No.
                
                # Let's stick to a simpler heuristic for Max Clique:
                # 1. Pick vertex with highest degree.
                # 2. Recurse on its neighbors.
                # 3. Compare with best found.
                # This is essentially Branch and Bound without coloring, but with degree ordering.
                # For n=64, this might be too slow if not pruned well.
                
                # However, for dense graphs, the max clique is large, so the branching factor is high but the depth is small?
                # Actually, if the graph is dense, the neighborhood is large.
                
                # Let's try a different approach:
                # Since n is small, we can use a bitset-based Bron-Kerbosch to find ALL maximal cliques,
                # then pick the largest one.
                # Number of maximal cliques in dense random graphs is typically small.
                # Let's check if the number of maximal cliques is manageable.
                # For n=32, p=0.6, it's likely fine.
                # For n=64, p=0.55, it might be larger but probably still manageable for a few iterations.
                # But we do this iteratively.
                
                # Let's go back to the "Find all maximal cliques, pick largest" strategy but optimized.
                # We can stop the Bron-Kerbosch search if we find a clique of size equal to the theoretical max?
                # We don't know the theoretical max.
                
                # Let's use the iterative extraction of the largest maximal clique found by a *single* run of BK?
                # No, BK finds all.
                
                # Let's use a heuristic:
                # 1. Find all maximal cliques.
                # 2. Pick the largest.
                # 3. Remove its edges.
                # 4. Repeat.
                
                # To speed up BK, use bitsets.
                
                return [], [] # Placeholder
                
            # Let's implement Max Clique using Bitset BK that returns the LARGEST clique found.
            # We can modify BK to track the largest clique and prune if current size + |P| <= best_size.
            
            best_clique = []
            best_size = 0
            
            def bk(R, P, X):
                nonlocal best_clique, best_size
                if time.time() - start_time > time_limit:
                    return
                
                if not P and not X:
                    if len(R) > best_size:
                        best_size = len(R)
                        best_clique = list(R)
                    return
                
                # Pruning: if len(R) + len(P) <= best_size, no clique in this branch can be better
                if len(R) + len(P) <= best_size:
                    return
                
                # Pivot selection
                P_list = list(P)
                X_list = list(X)
                union = P_list + X_list
                if not union:
                    return
                
                best_u = None
                max_inter = -1
                for u in union:
                    inter = (adj[u] & P).bit_count() if isinstance(adj[u], int) else 0
                    # adj is list of ints
                    inter = (adj[u] & P).bit_count()
                    if inter > max_inter:
                        max_inter = inter
                        best_u = u
                
                if best_u is None:
                    return
                
                # Candidates are vertices in P not adjacent to best_u
                # P is a bitmask
                candidates_mask = P & ~(adj[best_u])
                
                # Iterate through candidates
                temp = candidates_mask
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    
                    R.add(v)
                    P_new = P & adj[v]
                    X_new = X & adj[v]
                    
                    bk(R, P_new, X_new)
                    
                    P &= ~(1 << v)
                    X |= (1 << v)
                    R.remove(v)
                    
                    # Early stop if we found a clique of size n? No.
                    # But if best_size is large, pruning will handle it.
                    
            bk(set(), (1 << n) - 1, 0)
            
            return best_clique

    def solve_instance(n, p, seed):
        adj, edges = generate_graph(n, p, seed)
        
        if time.time() - start_time > time_limit:
            return [[u, v] for u, v in edges]

        partition = []
        remaining_edges = set(edges)
        
        # Use a dynamic adjacency mask for the remaining edges
        # adj_mask[u] is bitmask of neighbors v such that (u,v) is in remaining_edges
        adj_mask = [0] * n
        for u, v in remaining_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        while remaining_edges:
            if time.time() - start_time > time_limit:
                break
            
            # Find max clique in graph defined by adj_mask
            clique = max_clique(adj_mask, n, time_limit, start_time)
            
            if not clique or len(clique) < 2:
                # No clique found, add remaining edges individually
                for u, v in remaining_edges:
                    partition.append([u, v])
                break
            
            partition.append(clique)
            
            # Remove edges in clique from remaining_edges and adj_mask
            clique_set = set(clique)
            # Edges in clique are all pairs (u,v) with u,v in clique
            # We need to remove them from remaining_edges
            # And clear bits in adj_mask
            
            # To efficiently remove, iterate through pairs in clique
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i+1, len(clique)):
                    v = clique[j]
                    if (u, v) in remaining_edges:
                        remaining_edges.remove((u, v))
                    if (v, u) in remaining_edges:
                        remaining_edges.remove((v, u))
                    # Clear bits
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
            
            # Optimization: If the number of remaining edges is small, just add them as 2-cliques?
            # The loop will handle it, but max_clique might be slow if graph is sparse.
            # If remaining edges < threshold, fallback to individual edges?
            # No, we want to minimize cliques.
            
        return partition

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
