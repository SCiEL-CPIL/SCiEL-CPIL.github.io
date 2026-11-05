import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(edges_set, n):
        # Adjacency matrix for fast O(1) lookup
        # adj[u][v] is True if edge (u,v) exists in the REMAINING graph
        adj = [[False] * n for _ in range(n)]
        
        # Initialize adjacency from edges_set
        for u, v in edges_set:
            adj[u][v] = True
            adj[v][u] = True
            
        # Track remaining edges to know when to stop
        # We can just check if any edge exists, or count them.
        # For efficiency, we maintain a list of active vertices or just check adj.
        # Actually, we need to remove edges. 
        # A clique partition means we remove all edges of the clique from the graph.
        
        cliques = []
        
        # Helper to get neighbors of u in the remaining graph
        def get_neighbors(u):
            return [v for v in range(n) if adj[u][v]]
            
        # Tomita's algorithm for Maximum Clique
        # Uses greedy coloring to bound the search.
        # Returns the maximum clique found.
        
        def max_clique(vertices):
            if not vertices:
                return []
            
            # If time is up, return a trivial clique (edge) to ensure termination
            if time.time() > DEADLINE:
                return vertices[:2]
                
            # We use a recursive search.
            # To avoid deep recursion issues or stack overflow on large n (though n=64 is small),
            # we can implement it carefully.
            
            # The standard Tomita algorithm operates on a set of candidate vertices.
            # It colors the subgraph induced by vertices to get an upper bound on clique size.
            
            # Greedy coloring of the subgraph induced by 'vertices'
            # Returns a list of colors assigned to vertices (indices in vertices)
            # and the max color used.
            def color_sort(vertices):
                # Greedy coloring
                # We want to order vertices such that we can bound the clique size.
                # Standard approach: 
                # 1. Color the graph.
                # 2. Sort vertices by color.
                # 3. The color of a vertex gives an upper bound on the clique size 
                #    in the subgraph induced by that vertex and all subsequent vertices in the sorted order.
                
                uncolored = list(vertices)
                colors = {}
                color = 0
                ordered_vertices = []
                
                while uncolored:
                    color += 1
                    independent_set = []
                    remaining = []
                    for v in uncolored:
                        # v can be added to independent_set if it's not connected to any in independent_set
                        # In the context of coloring for Max Clique (on the graph itself),
                        # we color the graph G. A clique in G is an independent set in G_bar.
                        # But Tomita colors G directly? 
                        # Wait. The bound is: if we color G, a clique can have at most 1 vertex of each color? No.
                        # A clique in G is a set of vertices where all are connected.
                        # If we color G such that no two adjacent vertices have the same color,
                        # then a clique can contain at most one vertex of any given color? 
                        # No, that's for Independent Set.
                        # For Max Clique, we usually color the complement graph? 
                        # Or we use the fact that if we color G, the clique number is bounded by the chromatic number of G?
                        # No, chi(G) >= omega(G). So the number of colors is an upper bound on the clique size.
                        # So we color G.
                        
                        can_add = True
                        for u in independent_set:
                            if adj[v][u]:
                                can_add = False
                                break
                        if can_add:
                            independent_set.append(v)
                            colors[v] = color
                        else:
                            remaining.append(v)
                    uncolored = remaining
                    # Add vertices of this color to the ordered list (reversed? or just append?)
                    # In Tomita, we process vertices in order of increasing color? 
                    # Actually, the standard implementation sorts by color descending or ascending?
                    # Let's look at a reference.
                    # "Color sort" usually produces a sequence where the color of vertex v is an upper bound 
                    # on the size of the clique in {v, v+1, ...}.
                    # So we should sort by color.
                    ordered_vertices.extend(independent_set)
                    
                # Now we have ordered_vertices and colors.
                # We return the ordered list and the colors dict.
                return ordered_vertices, colors

            best_clique = []
            
            # Recursive search
            # P: current set of candidates
            # We maintain a list of vertices in P, sorted by color.
            
            def search(P, R):
                nonlocal best_clique
                if time.time() > DEADLINE:
                    return
                
                if not P:
                    if len(R) > len(best_clique):
                        best_clique = list(R)
                    return
                    
                # Color sort P
                # We need to color the subgraph induced by P
                # Note: The coloring must be valid for the subgraph.
                # Greedy coloring on P.
                
                # To speed up, we can use the previous coloring if we had one, 
                # but for simplicity and correctness, we re-color or use a simple heuristic.
                # Since n is small (<=64), re-coloring is fast.
                
                # However, re-coloring from scratch in every node is O(n^2) which might be slow if the tree is deep.
                # But the depth is bounded by the clique size (max ~15-20 for n=64, p=0.55).
                # So it's acceptable.
                
                # Let's implement a specific greedy coloring for the subgraph P.
                # We want to order P such that if we process v, the max color in P\{v} gives a bound.
                
                # Simple greedy coloring of G[P]
                uncolored = list(P)
                colors = {}
                color = 0
                ordered = []
                
                while uncolored:
                    color += 1
                    indep = []
                    rem = []
                    for v in uncolored:
                        ok = True
                        for u in indep:
                            if adj[v][u]:
                                ok = False
                                break
                        if ok:
                            indep.append(v)
                            colors[v] = color
                        else:
                            rem.append(v)
                    uncolored = rem
                    # Add to ordered list
                    # We want to process vertices with lower color first? 
                    # The bound for vertex v (at position i) is colors[v].
                    # If len(R) + colors[v] <= len(best_clique), we can prune.
                    # So we should iterate through ordered.
                    ordered.extend(indep)
                    
                # Now iterate through ordered vertices
                # We need to maintain the "remaining" set P as we go.
                # Standard Tomita:
                # for i from len(ordered) down to 1:
                #    v = ordered[i]
                #    if len(R) + colors[v] <= len(best_clique): break
                #    R.add(v)
                #    P_new = P intersect N(v)
                #    search(P_new, R)
                #    R.remove(v)
                #    P.remove(v)
                
                # Note: P is the set of candidates.
                # We need to be careful with the "P" passed to search.
                # The coloring was done on the initial P.
                # As we remove vertices from P, the coloring bound might be loose, but still valid (upper bound).
                
                current_P = set(P)
                
                # We iterate in reverse order of the sorted list (highest color first? or lowest?)
                # If colors are 1..k, and v has color c, the max clique in {v} U (vertices with color >= c) is <= c?
                # No, the property is: if vertices are ordered v1, v2, ..., vk such that colors(v1) <= colors(v2) <= ...
                # Then the clique in {vi, vi+1, ..., vk} is bounded by colors(vk)?
                # Actually, the standard bound is:
                # If we order by increasing color, then for vertex vi, the clique in {vi, vi+1, ...} is bounded by colors(vi)?
                # No. 
                # Let's use the property: 
                # If we have a valid coloring, the size of the maximum clique in a set of vertices is at most the maximum color assigned to those vertices.
                # If we process vertices in order of decreasing color, and we have processed all vertices with color > c, 
                # then the remaining candidates have max color c.
                # So if len(R) + c <= len(best_clique), we can stop.
                
                # So we should sort by color DESCENDING? 
                # Or sort by color ASCENDING and iterate from end?
                # Let's sort by color ascending.
                # ordered[0] has lowest color, ordered[-1] has highest.
                # Iterate from end to start.
                
                # Wait, if I take v with high color, the bound is high.
                # If I take v with low color, the bound is low.
                # If I process low color first, I might prune early?
                # No, if I process low color first, I add it to R.
                # The bound check is: if len(R) + max_color_in_remaining_P <= len(best_clique), prune.
                
                # Let's stick to a simpler, robust implementation for small N.
                # Since N is small, we can just do a recursive branch and bound without complex coloring updates, 
                # just re-coloring the subgraph at each step.
                
                # Actually, for N=64, a simple "pick vertex with max degree in P, branch on it" with pruning 
                # based on a simple upper bound (like max degree + 1) might be enough?
                # No, coloring is much tighter.
                
                # Let's use the ordering derived from the greedy coloring.
                # We will iterate through the ordered list.
                # We maintain a set `P_set` which is the current candidates.
                
                P_set = set(P)
                
                # To allow pruning, we need to know the max color in the *remaining* part of the ordered list?
                # The standard Tomita algorithm does this:
                # It maintains a list of vertices.
                # It colors them.
                # It iterates.
                
                # Let's implement the loop:
                # We need to be able to remove vertices from P_set.
                
                # Iterate from the last element (highest color) to the first?
                # If we pick a vertex with high color, the bound is high.
                # If we pick a vertex with low color, the bound is low.
                # Usually, we want to find large cliques fast.
                # Let's iterate from highest color to lowest.
                
                # Actually, the pruning condition is:
                # if len(R) + colors[v] <= len(best_clique): 
                #     continue (or break if we are iterating in a specific way)
                
                # If we iterate in order of DECREASING color:
                # The first vertex has the highest color.
                # If len(R) + max_color <= len(best_clique), we can break entirely?
                # No, because later vertices have lower colors, so the bound is even smaller.
                # So if len(R) + current_max_color <= len(best_clique), we can break.
                
                # So we should iterate from highest color to lowest.
                
                # Re-sort ordered by color descending
                ordered_desc = sorted(ordered, key=lambda x: colors[x], reverse=True)
                
                for v in ordered_desc:
                    if v not in P_set:
                        continue # Should not happen if we manage P_set correctly, but safe
                    
                    # Pruning check
                    if len(R) + colors[v] <= len(best_clique):
                        break # Since colors are descending, all subsequent v' have colors[v'] <= colors[v]
                        
                    # Branch on v
                    R.add(v)
                    
                    # New P is neighbors of v in current P_set
                    new_P = set()
                    for u in P_set:
                        if u != v and adj[v][u]:
                            new_P.add(u)
                            
                    search(new_P, R)
                    
                    R.remove(v)
                    P_set.remove(v)
                    
                    if time.time() > DEADLINE:
                        return

            # Initial call
            # We need to pick a starting set. 
            # To find a max clique in the whole graph, we start with P = all vertices.
            # But we want to find a max clique in the REMAINING graph.
            # So P = all vertices with degree > 0? Or just all vertices.
            # If a vertex has degree 0, it can't be in a clique of size > 1.
            
            active_vertices = [v for v in range(n) if any(adj[v])]
            if not active_vertices:
                return []
                
            search(set(active_vertices), set())
            
            if not best_clique:
                # Fallback if no clique found (should not happen if active_vertices exist)
                return active_vertices[:2]
                
            return best_clique

        # Main loop
        while True:
            if time.time() > DEADLINE:
                # Fallback: add all remaining edges as pairs
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u][v]:
                            cliques.append([u, v])
                break

            # Find max clique in remaining graph
            # To speed up, we can restrict to vertices with degree > 0
            active_vertices = [v for v in range(n) if any(adj[v])]
            if not active_vertices:
                break
                
            max_c = max_clique(active_vertices)
            
            if len(max_c) < 2:
                # Should not happen if active_vertices exist and graph has edges
                # But if max_clique fails or returns 1 vertex, we force an edge
                # Find any edge
                found = False
                for u in active_vertices:
                    for v in active_vertices:
                        if u < v and adj[u][v]:
                            max_c = [u, v]
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break

            cliques.append(max_c)
            
            # Remove edges of max_c from adj
            for i in range(len(max_c)):
                for j in range(i+1, len(max_c)):
                    u, v = max_c[i], max_c[j]
                    adj[u][v] = False
                    adj[v][u] = False
                    
        return cliques

    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    e1 = random_graph(n1, p1, s1)
    c1 = solve_instance(e1, n1)
    
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    e2 = random_graph(n2, p2, s2)
    c2 = solve_instance(e2, n2)
    
    return [c1, c2]
