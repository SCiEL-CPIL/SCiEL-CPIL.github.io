import random
import time

def construct():
    deadline = time.time() + 110  # 10s margin for safety
    n = 20
    num_colors = 20
    
    best_edges = None
    best_count = 0
    
    # Pre-compute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Precompute witnesses for each edge (u, v)
    # witnesses[u][v] = list of (a, b) such that path u-a-b-v exists
    # We only need to check if (a, b) is an edge in the current graph.
    witnesses = {}
    for u in range(n):
        for v in range(u+1, n):
            ws = []
            for a in range(n):
                if a == u or a == v: continue
                for b in range(n):
                    if b == u or b == v or b == a: continue
                    # Path u-a-b-v
                    # We need (u,a), (a,b), (b,v) to be potential edges
                    # We don't know if they are edges yet, but we can precompute the geometry
                    # Actually, we can just iterate neighbors in the greedy step.
                    # Precomputing all geometric paths is 20*20*20*20 = 160k entries, which is fine.
                    ws.append((a, b))
            witnesses[(u, v)] = ws
            witnesses[(v, u)] = ws # Symmetric, but we only use (u,v) with u<v usually? 
            # In greedy, we iterate (u,v) from all_edges which are u<v.
            # So we only need witnesses for u<v.

    def greedy_build(seed, initial_edges=None, order_type='random'):
        """
        Builds a graph using greedy edge addition.
        Returns (edges, rejected_edges)
        """
        rng = random.Random(seed)
        
        # Initialize adjacency and neighbors
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        existing_set = set()
        rejected = [] # List of (u, v, blocking_witnesses)
        
        # If we have initial edges, add them first
        if initial_edges:
            for e in initial_edges:
                u, v, c = e
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
                existing_set.add((min(u, v), max(u, v)))
            remaining_edges = [e for e in all_edges if e not in existing_set]
        else:
            remaining_edges = all_edges[:]
            
        # Determine order of remaining edges
        if order_type == 'random':
            rng.shuffle(remaining_edges)
        elif order_type == 'natural':
            pass
        elif order_type == 'reverse':
            remaining_edges.reverse()
            
        for (u, v) in remaining_edges:
            # Check if edge can be added
            # Find all length-3 paths u-a-b-v
            # If any path has non-distinct colors, edge is impossible.
            # Otherwise, forbidden colors are the colors on those paths.
            
            impossible = False
            forbidden_colors = set()
            blockers = [] # To store which edges blocked it
            
            # Use precomputed witnesses
            for (a, b) in witnesses[(u, v)]:
                # Check if (u,a), (a,b), (b,v) are all edges
                # Note: witnesses contains ALL geometric paths. We must check existence.
                if adj[u][a] == -1 or adj[a][b] == -1 or adj[b][v] == -1:
                    continue
                    
                ca = adj[u][a]
                cb = adj[a][b]
                cv = adj[b][v]
                
                # Check distinctness of ca, cb, cv
                if ca == cb or cb == cv or ca == cv:
                    impossible = True
                    break
                # Add to forbidden
                forbidden_colors.add(ca)
                forbidden_colors.add(cb)
                forbidden_colors.add(cv)
                # Track blockers for local search
                # The edge (a,b) is a "blocker" in the sense that its color contributes to the constraint.
                # Actually, any of the three edges could be "removed" to relax the constraint?
                # No, if we remove (a,b), the path is broken, so the constraint disappears.
                # So (a,b) is a blocker.
                blockers.append((a, b))
            
            if not impossible:
                # Find a valid color
                colors = list(range(num_colors))
                rng.shuffle(colors)
                
                placed = False
                for c in colors:
                    if c not in forbidden_colors:
                        # Place edge
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                        placed = True
                        break
            else:
                rejected.append((u, v, blockers))
                
        return edges, rejected

    def validate(edges):
        """
        Validates the graph.
        Returns True if valid, False otherwise.
        """
        adj = [[-1] * n for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
            
        # Check all 4-cycles
        # For each 4-tuple (u, v, w, x) with distinct vertices
        # If (u,v), (v,w), (w,x), (x,u) are all edges, check colors
        # To avoid duplicates, we can iterate u < v < w < x and check the 3 possible cycles
        # Cycle 1: u-v-w-x-u
        # Cycle 2: u-v-x-w-u
        # Cycle 3: u-w-v-x-u
        
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] == -1: continue
                for w in range(v+1, n):
                    if adj[v][w] == -1: continue
                    for x in range(w+1, n):
                        # Check Cycle 1: u-v-w-x-u
                        if adj[w][x] != -1 and adj[x][u] != -1:
                            c1 = adj[u][v]
                            c2 = adj[v][w]
                            c3 = adj[w][x]
                            c4 = adj[x][u]
                            if len({c1, c2, c3, c4}) < 4:
                                return False
                        # Check Cycle 2: u-v-x-w-u
                        if adj[v][x] != -1 and adj[w][u] != -1:
                            c1 = adj[u][v]
                            c2 = adj[v][x]
                            c3 = adj[x][w]
                            c4 = adj[w][u]
                            if len({c1, c2, c3, c4}) < 4:
                                return False
                        # Check Cycle 3: u-w-v-x-u
                        if adj[u][w] != -1 and adj[v][x] != -1:
                            c1 = adj[u][w]
                            c2 = adj[w][v]
                            c3 = adj[v][x]
                            c4 = adj[x][u]
                            if len({c1, c2, c3, c4}) < 4:
                                return False
        return True

    # Phase 1: Standard Random Restarts
    order_types = ['random', 'natural', 'reverse']
    num_restarts = 200 # Increased restarts since greedy is faster with precomputed witnesses? 
    # Actually, witnesses iteration is still O(N^2) per edge check. 
    # But we avoid set lookups for neighbors? No, we still check adj.
    
    for ot in order_types:
        for i in range(num_restarts):
            if time.time() > deadline:
                break
            edges, rejected = greedy_build(i, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    if best_edges is None:
        best_edges = [[0, v, 0] for v in range(1, 20)]
        best_count = len(best_edges)

    # Phase 2: Targeted Local Search
    # Use the rejected edges from the best greedy run?
    # We need to re-run greedy to get the rejected list for the best solution.
    # Or we can just run local search on the current best_edges.
    
    current_best = best_edges[:]
    
    # Re-run greedy to get rejected edges for the current best configuration?
    # No, greedy is stochastic. The rejected edges depend on the order.
    # Instead, we can identify "missing" edges in current_best and try to add them by removing blockers.
    
    missing_edges = [e for e in all_edges if (min(e[0], e[1]), max(e[0], e[1])) not in set((min(u,v), max(u,v)) for u,v,_ in current_best)]
    
    max_refine_iterations = 2000
    refine_iter = 0
    
    while time.time() < deadline and refine_iter < max_refine_iterations:
        refine_iter += 1
        
        if not missing_edges:
            break
            
        # Pick a random missing edge
        u, v = random.choice(missing_edges)
        
        # Try to find a blocking witness for (u, v) in the current graph
        # We need to check if (u, v) is currently blocked.
        # Re-use logic from greedy_build
        adj_temp = [[-1] * n for _ in range(n)]
        for uu, vv, cc in current_best:
            adj_temp[uu][vv] = cc
            adj_temp[vv][uu] = cc
            
        blocked_by = []
        impossible = False
        forbidden = set()
        for (a, b) in witnesses[(u, v)]:
            if adj_temp[u][a] == -1 or adj_temp[a][b] == -1 or adj_temp[b][v] == -1:
                continue
            ca = adj_temp[u][a]
            cb = adj_temp[a][b]
            cv = adj_temp[b][v]
            if ca == cb or cb == cv or ca == cv:
                impossible = True
                break
            forbidden.add(ca)
            forbidden.add(cb)
            forbidden.add(cv)
            blocked_by.append((a, b))
            
        if impossible:
            # Cannot add even if we remove one edge? 
            # If ca==cb, removing (a,b) breaks the path. 
            # But if there are multiple paths, one might still block.
            # If impossible, we must remove an edge that is part of a "conflicting" path.
            # Let's pick a random edge from the graph that is part of a path with non-distinct colors.
            # This is harder. Let's just skip this edge for now.
            continue
            
        # If not impossible, we can add it if we pick a color not in forbidden.
        # But it's missing, so greedy must have failed to find a color or it was blocked by something else?
        # Wait, if it's not impossible and there is a color not in forbidden, why is it missing?
        # Maybe it was added later? No, it's missing.
        # Maybe the color set was exhausted?
        # If len(forbidden) == 20, then no color available.
        # In that case, we must remove a blocking edge to reduce forbidden colors.
        
        if len(forbidden) < 20:
            # We can add it! Why is it missing?
            # Maybe it was added in a different color? No, it's missing.
            # Maybe the greedy order caused it to be rejected earlier due to a different state?
            # Anyway, let's try to add it.
            available_colors = [c for c in range(num_colors) if c not in forbidden]
            c_new = random.choice(available_colors)
            new_edges = current_best + [[u, v, c_new]]
            # Validate
            if validate(new_edges):
                best_count = len(new_edges)
                best_edges = new_edges
                current_best = new_edges[:]
                # Update missing_edges
                missing_edges = [e for e in all_edges if (min(e[0], e[1]), max(e[0], e[1])) not in set((min(uu,vv), max(uu,vv)) for uu,vv,_ in current_best)]
                continue
        else:
            # Forbidden is full. Must remove a blocker.
            if blocked_by:
                # Pick a random blocker edge (a, b)
                a, b = random.choice(blocked_by)
                # Remove (a, b) from current_best
                new_edges = [e for e in current_best if not (min(e[0], e[1]) == min(a, b) and max(e[0], e[1]) == max(a, b))]
                # Try to add (u, v) with any color
                # Re-check forbidden colors for (u, v) in new_edges
                adj_temp2 = [[-1] * n for _ in range(n)]
                for uu, vv, cc in new_edges:
                    adj_temp2[uu][vv] = cc
                    adj_temp2[vv][uu] = cc
                    
                forbidden2 = set()
                impossible2 = False
                for (aa, bb) in witnesses[(u, v)]:
                    if adj_temp2[u][aa] == -1 or adj_temp2[aa][bb] == -1 or adj_temp2[bb][v] == -1:
                        continue
                    caa = adj_temp2[u][aa]
                    cbb = adj_temp2[aa][bb]
                    cvv = adj_temp2[bb][v]
                    if caa == cbb or cbb == cvv or caa == cvv:
                        impossible2 = True
                        break
                    forbidden2.add(caa)
                    forbidden2.add(cbb)
                    forbidden2.add(cvv)
                    
                if not impossible2:
                    available_colors2 = [c for c in range(num_colors) if c not in forbidden2]
                    if available_colors2:
                        c_new = random.choice(available_colors2)
                        candidate_edges = new_edges + [[u, v, c_new]]
                        if validate(candidate_edges):
                            best_count = len(candidate_edges)
                            best_edges = candidate_edges
                            current_best = candidate_edges[:]
                            missing_edges = [e for e in all_edges if (min(e[0], e[1]), max(e[0], e[1])) not in set((min(uu,vv), max(uu,vv)) for uu,vv,_ in current_best)]
                            continue

    return best_edges
