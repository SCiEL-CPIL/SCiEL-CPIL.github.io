import time
import random

_START_TIME = time.time()
DEADLINE = 120.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _check_c4(u, v, neighbors, neighbor_sets):
    # Check if adding (u,v) creates a C4
    # Path u-x-y-v where x in N(u), y in N(x), y in N(v)
    for x in neighbors[u]:
        for y in neighbors[x]:
            if y == u:
                continue
            if y in neighbor_sets[v]:
                return True
    return False

def construct():
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    best_edges = 0
    best_graph = []
    
    # --- Phase 1: Bipartite Construction (Inspiration 1 & 3) ---
    # We try to construct a subgraph of K_{10,10} with a valid coloring.
    # Vertices 0..9 are A, 10..19 are B.
    # A 4-cycle in K_{10,10} is u1-v1-u2-v2-u1.
    # Edges: (u1,v1), (u2,v1), (u2,v2), (u1,v2).
    # Colors must be distinct.
    
    A = list(range(10))
    B = list(range(10, 20))
    
    # Strategy: Try various linear colorings c(u,v) = (a*u + b*v + c) % 20
    # And also try to find a maximum set of edges that can be colored.
    # Since we can choose colors, we can be more flexible.
    # However, the constraint is "edge-coloured with 20 colours so every 4-cycle is rainbow".
    # This implies we need to assign colors to the edges we include.
    
    # Let's try to include as many edges of K_{10,10} as possible.
    # If we include all 100 edges, is it colorable?
    # This is equivalent to finding a graph homomorphism or specific algebraic structure.
    # A known result: K_{n,n} can be edge-colored with n colors such that every C4 is rainbow?
    # Actually, for K_{n,n}, if we color (u,v) with (u+v) % n, a C4 (u1,v1,u2,v2) has colors:
    # c1=u1+v1, c2=u2+v1, c3=u2+v2, c4=u1+v2.
    # c1=c3 iff u1+v1 = u2+v2.
    # c2=c4 iff u2+v1 = u1+v2.
    # For n=10, there are many collisions.
    
    # Let's use the "greedy removal" approach from Inspiration 1.
    # Start with all 100 edges of K_{10,10}.
    # Assign colors using a function.
    # Identify all bad C4s (where colors repeat).
    # Remove edges to break all bad C4s.
    
    # We will try multiple color functions and pick the one that leaves the most edges.
    
    def try_bipartite_coloring(color_func):
        # edge_colors[(u,v)] = c
        edge_colors = {}
        for u in A:
            for v in B:
                edge_colors[(u,v)] = color_func(u,v)
                edge_colors[(v,u)] = color_func(u,v)
        
        # Find all bad cycles
        bad_cycles = []
        for i in range(10):
            for j in range(i+1, 10):
                u1, u2 = A[i], A[j]
                for k in range(10):
                    for l in range(k+1, 10):
                        v1, v2 = B[k], B[l]
                        
                        c1 = edge_colors[(u1,v1)]
                        c2 = edge_colors[(u2,v1)]
                        c3 = edge_colors[(u2,v2)]
                        c4 = edge_colors[(u1,v2)]
                        
                        colors = {c1, c2, c3, c4}
                        if len(colors) < 4:
                            # Bad cycle
                            # Edges involved: (u1,v1), (u2,v1), (u2,v2), (u1,v2)
                            bad_cycles.append([(u1,v1), (u2,v1), (u2,v2), (u1,v2)])
        
        if not bad_cycles:
            return 100, edge_colors
        
        # Hitting set problem: remove edges to hit all bad_cycles
        # Greedy set cover
        edges_to_remove = set()
        remaining_cycles = bad_cycles[:]
        
        # Pre-compute incidence for speed? Or just greedy.
        # 100 edges, ~2000 cycles. Greedy is O(E * C^2) worst case, which is fine.
        
        while remaining_cycles and _time_left() > 10.0:
            best_edge = None
            max_hits = 0
            
            # Collect candidate edges
            candidates = set()
            for cyc in remaining_cycles:
                for e in cyc:
                    if e not in edges_to_remove:
                        candidates.add(e)
            
            if not candidates:
                break
                
            for e in candidates:
                hits = 0
                for cyc in remaining_cycles:
                    if e in cyc:
                        hits += 1
                if hits > max_hits:
                    max_hits = hits
                    best_edge = e
            
            if best_edge is None:
                break
                
            edges_to_remove.add(best_edge)
            # Filter cycles
            remaining_cycles = [cyc for cyc in remaining_cycles if best_edge not in cyc]
            
        valid_edges = 100 - len(edges_to_remove)
        
        # Reconstruct valid edge list with colors
        final_edges = []
        for u in A:
            for v in B:
                if (u,v) not in edges_to_remove:
                    final_edges.append((u,v, edge_colors[(u,v)]))
                    
        return valid_edges, final_edges

    # Try a few deterministic colorings
    strategies = [
        lambda u,v: (u + v) % 20,
        lambda u,v: (u + 2*v) % 20,
        lambda u,v: (2*u + v) % 20,
        lambda u,v: (u + v + 1) % 20,
        lambda u,v: (u * 3 + v) % 20,
        lambda u,v: (u * 2 + v * 3) % 20,
    ]
    
    # Also try random linear combinations
    for _ in range(20):
        a = random.randint(0, 19)
        b = random.randint(0, 19)
        c = random.randint(0, 19)
        strategies.append(lambda u,v,a=a,b=b,c=c: (a*u + b*v + c) % 20)
        
    best_bip_edges = 0
    best_bip_graph = []
    
    for func in strategies:
        if _time_left() < 20.0:
            break
        count, edges = try_bipartite_coloring(func)
        if count > best_bip_edges:
            best_bip_edges = count
            best_bip_graph = edges
    
    # --- Phase 2: General Graph Construction (Current Program Logic) ---
    # Now, can we add edges within A or within B?
    # Or just run the general greedy on the full graph, starting from the bipartite core?
    
    # Let's initialize the graph with the best bipartite solution found.
    current_edges = set()
    adj = [[False] * n for _ in range(n)]
    neighbors = [[] for _ in range(n)]
    neighbor_sets = [set() for _ in range(n)]
    edge_colors = {}
    
    for u, v, c in best_bip_graph:
        current_edges.add((u,v))
        adj[u][v] = True
        adj[v][u] = True
        neighbors[u].append(v)
        neighbors[v].append(u)
        neighbor_sets[u].add(v)
        neighbor_sets[v].add(u)
        edge_colors[(u,v)] = c
        edge_colors[(v,u)] = c
        
    # Now try to add more edges (within A, within B, or missing bipartite ones)
    # We need to assign a color to any new edge such that no C4 becomes non-rainbow.
    # This is harder. The current program just adds edges and assigns color 0 (which is invalid if there are C4s).
    # Wait, the current program's output `[[u, v, 0] ...]` implies it doesn't actually check color validity in the general phase,
    # it just ensures the graph is C4-free?
    # "Every 4-cycle ... must receive four distinct colours".
    # If the graph is C4-free, the condition is vacuously true.
    # So the current program constructs a C4-free graph.
    # But the bipartite construction allows C4s as long as they are rainbow.
    # So we can potentially have MORE edges by allowing C4s if we color them correctly.
    
    # Can we extend the bipartite graph by adding intra-part edges?
    # Adding an edge (u1, u2) in A creates C4s of the form u1-u2-v1-?-u1? No.
    # A C4 involving (u1, u2) must be u1-u2-v1-w-u1.
    # If w is in A, we have a triangle u1,u2,w.
    # If w is in B, we have u1-u2-v1-w-u1.
    # Edges: (u1,u2), (u2,v1), (v1,w), (w,u1).
    # We need distinct colors.
    
    # Let's try to greedily add intra-part edges to the bipartite core.
    # For each candidate edge (u,v) not in current_edges:
    #   Check if adding it creates any C4.
    #   For each new C4 created, check if we can assign a color to (u,v)
    #   such that all 4 edges have distinct colors.
    #   Note: existing edges in the C4 already have colors.
    #   If the existing 3 edges have distinct colors, we can pick the 4th color
    #   from the 20 available, as long as it's not one of the 3.
    #   If the existing 3 edges have a collision, the C4 is already invalid?
    #   No, the C4 is formed by adding (u,v). The other 3 edges existed before.
    #   If the other 3 edges had a color collision, that C4 would have been invalid before?
    #   No, because the C4 didn't exist before.
    #   However, if the other 3 edges are part of another C4 that IS valid, their colors are fixed.
    #   So we just need to ensure the new C4 is rainbow.
    #   This requires that the 3 existing edges in the new C4 have distinct colors.
    #   AND we can find a color for (u,v) distinct from them.
    
    # Let's implement this extension.
    
    candidate_edges = [e for e in all_edges if e not in current_edges]
    random.shuffle(candidate_edges)
    
    added_count = 0
    for u, v in candidate_edges:
        if _time_left() < 5.0:
            break
        if adj[u][v]:
            continue
            
        # Find all C4s created by adding (u,v)
        # A C4 is u-x-y-v-u.
        # x in N(u), y in N(v), x-y is an edge.
        # Or x in N(v), y in N(u), x-y is an edge.
        # Basically, any path of length 3 between u and v.
        
        new_c4s = []
        
        # Iterate neighbors of u
        for x in neighbors[u]:
            # Iterate neighbors of x
            for y in neighbors[x]:
                if y == u:
                    continue
                # If y is in N(v), then u-x-y-v is a path.
                # And (u,v) closes the cycle.
                # We need x != v and y != u (handled) and x != y (simple graph).
                if y in neighbor_sets[v]:
                    # Cycle: u-x-y-v-u
                    # Edges: (u,x), (x,y), (y,v), (u,v)
                    # Check colors of (u,x), (x,y), (y,v)
                    c1 = edge_colors.get((u,x))
                    c2 = edge_colors.get((x,y))
                    c3 = edge_colors.get((y,v))
                    
                    # If any color is missing, it's a problem?
                    # In our bipartite core, all edges have colors.
                    # If we add intra-part edges, they will have colors.
                    # So c1, c2, c3 should exist.
                    
                    if c1 is None or c2 is None or c3 is None:
                        # Should not happen if we maintain invariant
                        continue
                        
                    if len({c1, c2, c3}) == 3:
                        # We can assign a color to (u,v)
                        # Store this constraint
                        new_c4s.append({c1, c2, c3})
                    else:
                        # Existing 3 edges have a collision.
                        # This C4 cannot be made rainbow regardless of (u,v)'s color.
                        # So we cannot add (u,v).
                        new_c4s = None
                        break
            if new_c4s is None:
                break
                
        if new_c4s is None:
            continue
            
        # If we got here, all new C4s have 3 distinct existing colors.
        # We need to find a color for (u,v) that is distinct from all colors in all these C4s.
        # Forbidden colors: union of all colors in new_c4s.
        
        forbidden = set()
        for c_set in new_c4s:
            forbidden.update(c_set)
            
        # Also, (u,v) must not create a C4 with *existing* edges that were already C4s?
        # No, adding an edge only creates NEW C4s involving that edge.
        # Existing C4s are unaffected by the color of a new edge not in them.
        
        # Wait, what if (u,v) is part of a C4 that was already present?
        # Impossible, (u,v) was not present.
        
        # So we just need a color not in `forbidden`.
        # There are 20 colors. `forbidden` size is at most 3 * (#C4s).
        # If #C4s is small, we have plenty of choices.
        
        valid_color = None
        for c in range(20):
            if c not in forbidden:
                valid_color = c
                break
                
        if valid_color is None:
            continue
            
        # Add edge (u,v) with color valid_color
        current_edges.add((u,v))
        adj[u][v] = True
        adj[v][u] = True
        neighbors[u].append(v)
        neighbors[v].append(u)
        neighbor_sets[u].add(v)
        neighbor_sets[v].add(u)
        edge_colors[(u,v)] = valid_color
        edge_colors[(v,u)] = valid_color
        added_count += 1
        
    # Combine results
    total_edges = len(current_edges)
    
    # If the bipartite + extension is better than the pure C4-free greedy, use it.
    # We should also run the pure C4-free greedy to compare.
    
    # Run pure C4-free greedy (from current program)
    best_c4_free_edges = 0
    best_c4_free_graph = []
    
    attempts = 0
    while _time_left() > 5.0 and attempts < 5:
        current_edges_cf = set()
        adj_cf = [[False] * n for _ in range(n)]
        neighbors_cf = [[] for _ in range(n)]
        neighbor_sets_cf = [set() for _ in range(n)]
        
        random.seed(attempts)
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        graph_edges = []
        for u, v in shuffled_edges:
            if adj_cf[u][v]:
                continue
            if not _check_c4(u, v, neighbors_cf, neighbor_sets_cf):
                adj_cf[u][v] = True
                adj_cf[v][u] = True
                neighbors_cf[u].append(v)
                neighbors_cf[v].append(u)
                neighbor_sets_cf[u].add(v)
                neighbor_sets_cf[v].add(u)
                graph_edges.append((u, v))
                current_edges_cf.add((u, v))
                
        if len(graph_edges) > best_c4_free_edges:
            best_c4_free_edges = len(graph_edges)
            best_c4_free_graph = graph_edges[:]
        attempts += 1
        
    # Decide which is better
    if total_edges > best_c4_free_edges:
        # Convert current_edges to list of [u,v,c]
        result = []
        for u, v in current_edges:
            c = edge_colors.get((u,v), 0)
            result.append([u, v, c])
        return result
    else:
        # Use C4-free graph, color all 0 (valid since no C4s)
        result = [[u, v, 0] for u, v in best_c4_free_graph]
        return result
