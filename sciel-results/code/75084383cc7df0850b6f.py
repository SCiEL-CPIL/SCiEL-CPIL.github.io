import time
import random

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # All possible edges in K16
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    m = len(all_edges)
    
    # Edge index mapping
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    
    # Build conflict graph: two edges conflict if they share a vertex or form a triangle
    # conflict[i] is a bitset of edges that conflict with edge i
    # Two edges (u,v) and (w,x) conflict if:
    #   - They share a vertex (u==w, u==x, v==w, v==x)
    #   - They form a triangle: {u,v} and {w,x} are disjoint but there's a vertex that connects
    #     Actually for triangle-free: edges (a,b) and (c,d) form a triangle if there exists e
    #     such that (a,e) and (b,e) or (c,e) and (d,e) etc.
    #     Wait, two edges (a,b) and (c,d) form a triangle if they share a vertex AND the third
    #     edge exists. But we're checking if ADDING both edges creates a triangle.
    #     Two edges create a triangle with a third edge if:
    #     - They share exactly one vertex: (a,b) and (a,c) -> triangle if (b,c) exists
    #     - They are disjoint: (a,b) and (c,d) -> no direct triangle from just these two
    #     
    #     Actually, a triangle is three edges forming a cycle of length 3. Two edges alone
    #     don't form a triangle. But when we select a set of edges, we need no three edges
    #     to form a triangle.
    #     
    #     For the conflict graph approach for maximum triangle-free subgraph:
    #     We want to find the maximum subset of edges with no triangle.
    #     This is equivalent to: find the maximum clique in the graph where two edge-nodes
    #     are connected if they can coexist (i.e., don't form a triangle together with any
    #     other selected edge).
    #     
    #     Actually, the standard approach: two edges "conflict" if they cannot both be in a
    #     triangle-free graph. Two edges (a,b) and (c,d) conflict if there exists a vertex e
    #     such that (a,e) and (b,e) are both edges... no that's not right either.
    #     
    #     Let me think more carefully. A set of edges is triangle-free iff for every triple
    #     of vertices {x,y,z}, not all three edges (x,y), (y,z), (z,x) are present.
    #     
    #     For the clique formulation: create a graph H where nodes are the edges of K16.
    #     Two nodes (edges e1, e2) are adjacent in H if e1 and e2 do NOT form a triangle
    #     with any edge. But this isn't quite right because triangle-freeness is a 3-wise
    #     constraint, not a 2-wise one.
    #     
    #     Alternative: Two edges (a,b) and (c,d) are incompatible if they share a vertex and
    #     the edge connecting the other two endpoints exists. I.e., if (a,b) and (a,c) are both
    #     selected, then (b,c) cannot be selected. So (a,b) and (a,c) are "in conflict" in the
    #     sense that selecting both forces (b,c) to be excluded.
    #     
    #     The proper formulation: We want max subset S of edges such that no triangle is fully
    #     contained in S. This is the maximum triangle-free subgraph problem.
    #     
    #     For branch and bound: we can use the fact that if we have edges (a,b) and (a,c) in S,
    #     then (b,c) must not be in S. This gives us constraints.
    #     
    #     Let me use a different approach: direct backtracking with strong pruning.

    # Precompute for each vertex, which edges are incident to it
    vertex_edges = [[] for _ in range(n)]
    for i, (u, v) in enumerate(all_edges):
        vertex_edges[u].append(i)
        vertex_edges[v].append(i)
    
    # For each pair of vertices, check if they're connected
    adj_matrix = [[False]*n for _ in range(n)]
    for u, v in all_edges:
        adj_matrix[u][v] = adj_matrix[v][u] = True
    
    # For efficient triangle checking: for each pair (u,v), the set of vertices w such that
    # (u,w) and (v,w) are both edges (i.e., w completes a triangle with edge (u,v))
    # tri_neighbors[u][v] = set of w where (u,w) and (v,w) exist
    # If we select edge (u,v), we cannot select any edge (u,w) AND (v,w) for the same w.
    
    # Let's use a smarter approach: 
    # State: which edges are selected (bitset of m bits)
    # For each vertex u, track which of its incident edges are selected
    # When considering adding edge (u,v), check: for each w, if (u,w) and (v,w) are both
    # already selected, then adding (u,v) creates a triangle.
    
    # Represent selected edges as a set for simplicity, but use bitsets for speed
    
    # Let's try a different structural approach: 
    # Use the Grötzsch graph or other known 4-chromatic triangle-free graphs as starting points,
    # then use local search to add edges.
    
    # Known 4-chromatic triangle-free graphs on 16 vertices:
    # The Clebsch graph (40 edges) is 4-chromatic and triangle-free.
    # Can we do better?
    
    # Let's use a more systematic search:
    # 1. Start with a known 4-chromatic triangle-free graph
    # 2. Try to add edges while maintaining triangle-free and 4-chromatic properties
    
    # Build Clebsch graph
    def build_clebsch():
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                xor = u ^ v
                dist = bin(xor).count('1')
                if dist == 1 or dist == 4:
                    edges.add((u, v))
        return edges
    
    clebsch = build_clebsch()
    
    # Verify it's triangle-free and not 3-colorable
    def is_triangle_free(edge_set):
        for (u, v) in edge_set:
            for w in range(n):
                if w == u or w == v:
                    continue
                if (min(u,w), max(u,w)) in edge_set and (min(v,w), max(v,w)) in edge_set:
                    return False
        return True
    
    def is_3_colorable(edge_set, adj):
        """Check 3-colorability with backtracking."""
        color = [-1] * n
        
        # Order vertices by degree descending
        degrees = [sum(adj[v]) for v in range(n)]
        order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                valid = True
                for u in range(n):
                    if adj[v][u] and color[u] == c:
                        valid = False
                        break
                if valid:
                    color[v] = c
                    if backtrack(idx+1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    # Build adjacency for Clebsch
    adj = [[False]*n for _ in range(n)]
    for u, v in clebsch:
        adj[u][v] = adj[v][u] = True
    
    assert is_triangle_free(clebsch), "Clebsch should be triangle-free"
    assert not is_3_colorable(clebsch, adj), "Clebsch should not be 3-colorable"
    
    best = set(clebsch)
    best_count = len(best)
    
    # Try to add edges greedily
    def try_add_edges(current_edges, adj, deadline):
        """Try to add edges to current graph while keeping triangle-free and 4-chromatic."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u, v in all_edges:
                if (u, v) in current_edges:
                    continue
                # Check if adding (u,v) keeps triangle-free
                can_add = True
                for w in range(n):
                    if w == u or w == v:
                        continue
                    uw = (min(u,w), max(u,w))
                    vw = (min(v,w), max(v,w))
                    if uw in current_edges and vw in current_edges:
                        can_add = False
                        break
                if can_add:
                    # Check if the resulting graph is still not 3-colorable
                    new_edges = current_edges | {(u, v)}
                    new_adj = [row[:] for row in adj]
                    new_adj[u][v] = new_adj[v][u] = True
                    if not is_3_colorable(new_edges, new_adj):
                        current_edges.add((u, v))
                        adj[u][v] = adj[v][u] = True
                        changed = True
        return current_edges, adj
    
    # Greedy improvement
    current = set(best)
    cur_adj = [row[:] for row in adj]
    current, cur_adj = try_add_edges(current, cur_adj, deadline - 10)
    
    if len(current) > best_count:
        best = set(current)
        best_count = len(current)
    
    # Random restarts with local search
    random.seed(12345)
    
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Start from best or a perturbed version
        if restart == 0:
            current = set(best)
        else:
            current = set(best)
            # Remove some random edges
            num_remove = random.randint(2, 6)
            if current:
                to_remove = random.sample(list(current), min(num_remove, len(current)))
                for e in to_remove:
                    current.discard(e)
        
        # Rebuild adjacency
        cur_adj = [[False]*n for _ in range(n)]
        for u, v in current:
            cur_adj[u][v] = cur_adj[v][u] = True
        
        # Greedy add
        current, cur_adj = try_add_edges(current, cur_adj, deadline - 5)
        
        if len(current) > best_count:
            best = set(current)
            best_count = len(current)
    
    # Try a more aggressive search: for each edge not in best, try swapping
    if time.time() < deadline - 10:
        for u, v in all_edges:
            if (u, v) in best:
                continue
            # Try removing one edge from best and adding (u,v)
            for u2, v2 in list(best):
                if time.time() > deadline - 5:
                    break
                trial = best - {(u2, v2)} | {(u, v)}
                if len(trial) != len(best):
                    continue
                # Check triangle-free
                if not is_triangle_free(trial):
                    continue
                # Check 4-chromatic
                trial_adj = [[False]*n for _ in range(n)]
                for a, b in trial:
                    trial_adj[a][b] = trial_adj[b][a] = True
                if not is_3_colorable(trial, trial_adj):
                    # Now try to add more edges
                    trial, trial_adj = try_add_edges(trial, trial_adj, deadline - 5)
                    if len(trial) > best_count:
                        best = set(trial)
                        best_count = len(trial)
    
    edges = [[u, v] for u, v in best]
    return edges
