import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    N = 16
    
    # Precompute all possible edges as (u, v) with u < v
    all_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            all_edges.append((u, v))
    
    num_edges = len(all_edges)
    
    # Build conflict graph: edge i conflicts with edge j if they form a triangle
    # Two edges (a,b) and (c,d) form a triangle if they share exactly one endpoint
    # and the other two endpoints are connected... wait, no.
    # Two edges (a,b) and (a,c) form a triangle if (b,c) is also an edge.
    # But for the conflict in independent set: two edges CONFLICT if they cannot both be in a triangle-free graph.
    # Actually, two edges don't conflict just by sharing a vertex. They conflict if adding both would create a triangle.
    # Wait - in a triangle-free graph, you can have multiple edges sharing a vertex.
    # The constraint is: no three edges (a,b), (b,c), (a,c) can all be present.
    # So edge (a,b) conflicts with edge (b,c) ONLY IF edge (a,c) is also present.
    # This is a 3-way constraint, not a pairwise one.
    
    # Actually, for the independent set formulation on the line graph:
    # The line graph of K_16 has nodes = all edges of K_16.
    # Two nodes (edges) are adjacent in the line graph if the corresponding edges share a vertex.
    # A triangle-free graph corresponds to an independent set in this line graph? NO!
    # That's not right either. An independent set in the line graph means no two edges share a vertex (a matching).
    
    # Let me reconsider. The correct formulation:
    # We want a set of edges with no triangle. This is NOT simply an independent set in the line graph.
    # 
    # The correct conflict: edge e1 and edge e2 are in conflict if there exists a third edge e3 such that
    # {e1, e2, e3} forms a triangle. But that's still a 3-way thing.
    #
    # Actually, the standard approach: two edges (u,v) and (u,w) are "compatible" as long as (v,w) is not in the set.
    # So we need a different formulation.
    
    # Let me use a direct backtracking search with a fast 3-coloring check.
    
    # Alternative approach: Use the known maximum edge count for triangle-free non-3-colorable graphs on 16 vertices.
    # By Turan, max triangle-free edges on 16 vertices is floor(16^2/4) = 64 (complete bipartite K_8,8).
    # K_8,8 is 2-colorable, so we need fewer edges but not 3-colorable.
    # 
    # The Mycielski construction gives us a graph. Let me compute what the current program gives:
    # Base: 7 vertices, 10 edges
    # After Mycielski: 15 vertices, 10 + 2*10 + 7 = 37 edges
    # Plus vertex 15 connected to 7 vertices: 37 + 7 = 44 edges
    # Score 2.2 means 44/2 = 22? No, score = value/seed. Seed score... 
    # Actually the score is just the number of edges divided by some reference.
    # Score 2.2 with 44 edges means reference = 44/2.2 = 20.
    # So we need more than 44 edges to beat score 2.2.
    
    # Let me try a different approach: start from a dense bipartite graph and add edges carefully.
    
    # Start with K_8,8 (bipartition {0..7} and {8..15}), which has 64 edges and is 2-colorable.
    # We need to add cross-edges within partitions to break 3-colorability while maintaining triangle-free.
    
    # Actually, let me think about this differently. A triangle-free graph that is not 3-colorable.
    # The smallest such graph is the Grötzsch graph (11 vertices, 20 edges).
    # Mycielski of C5 gives the Grötzsch graph.
    
    # For 16 vertices, let's use a smarter construction or search.
    
    # Let me try: take Mycielski of the 7-vertex base graph, which gives 15 vertices with 37 edges.
    # Then add vertex 15. Currently it's connected to 7 vertices (G' vertices).
    # Can we connect it to more vertices?
    
    # In the Mycielski construction M(G):
    # - Original vertices: 0..6
    # - Shadow vertices: 7..13 (shadow of u is 7+u)
    # - Top vertex: 14, connected to all shadow vertices
    # 
    # Triangle-free check:
    # - Original edges: no triangles (base is triangle-free)
    # - Shadow edges: shadow of u connected to original v if (u,v) is edge in base. 
    #   Two shadow vertices 7+u and 7+v are NOT connected to each other.
    # - Top to shadows: top connected to all shadows. Shadows don't connect to each other, so no triangle with top.
    # 
    # For vertex 15: it's connected to all shadows (7..13). 
    # Shadows are independent, so no two shadows are adjacent. Thus {15, 7+u, 7+v} cannot be a triangle 
    # since 7+u and 7+v are not adjacent. Good.
    # But 15 is also connected to shadow 7+u, and shadow 7+u is connected to original vertices that are neighbors of u in base.
    # So {15, 7+u, v} where v is neighbor of u in base: 15-7+u is edge, 7+u-v is edge, 15-v is NOT an edge (unless we add it).
    # So no triangle. Good.
    
    # Can we add more edges for vertex 15? It's already connected to all 7 shadows.
    # What about connecting to original vertices 0..6?
    # If we connect 15 to original vertex v, then for any shadow 7+u where u is neighbor of v in base:
    # 15-v (new edge), v-(7+u) (existing edge since (u,v) or (v,u) in base), 15-(7+u) (existing edge)
    # That forms a triangle! So we CANNOT connect 15 to any original vertex that has a neighbor whose shadow is connected to 15.
    # Since ALL shadows are connected to 15, and every original vertex has at least one neighbor, 
    # we cannot connect 15 to any original vertex.
    
    # What about connecting 15 to top vertex 14?
    # 15-14 (new), 14-(7+u) (existing for all u), 15-(7+u) (existing for all u)
    # That forms a triangle {15, 14, 7+u} for any u. So NO.
    
    # So the current construction with 44 edges might be near-optimal for this family.
    
    # Let me try a completely different approach: branch and bound with 3-coloring oracle.
    
    # Represent the graph as adjacency bitmasks
    adj = [0] * N
    
    # Start with a good initial solution
    # Use the Mycielski-based construction as initial
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0),
        (0, 3), (1, 4), (2, 5)
    ]
    
    initial_edges = []
    for u, v in base_edges:
        initial_edges.append((u, v))
        initial_edges.append((u + 7, v))
        initial_edges.append((v + 7, u))
    for u in range(7):
        initial_edges.append((u + 7, 14))
    for u in range(7):
        initial_edges.append((15, u + 7))
    
    best_edges = list(initial_edges)
    best_count = len(best_edges)
    
    def add_edge_to_adj(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge_from_adj(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def is_triangle_free(adj, u, v):
        """Check if adding edge (u,v) keeps the graph triangle-free."""
        # For each common neighbor w of u and v, we'd form triangle u-v-w-u
        common = adj[u] & adj[v]
        return common == 0
    
    def is_not_3_colorable(adj):
        """Check if graph is not 3-colorable using backtracking."""
        # Simple backtracking 3-coloring
        colors = [-1] * N
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(N), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == N:
                return True
            v = order[idx]
            for c in range(3):
                # Check if color c is available
                ok = True
                nbrs = adj[v]
                for nb in range(N):
                    if (nbrs >> nb) & 1:
                        if colors[nb] == c:
                            ok = False
                            break
                if ok:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return not backtrack(0)
    
    # Build initial adjacency
    for u, v in initial_edges:
        add_edge_to_adj(adj, u, v)
    
    # Get list of candidate edges not in the graph
    edge_set = set(initial_edges)
    candidates = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) not in edge_set:
                # Check if adding this edge would maintain triangle-free
                if is_triangle_free(adj, u, v):
                    candidates.append((u, v))
    
    # Sort candidates by some heuristic (maybe by how many triangles they'd break if added)
    # Actually, let's just try adding edges greedily with backtracking
    
    # Branch and bound: try to add edges to maximize count while keeping triangle-free and not 3-colorable
    current_edges = set(initial_edges)
    current_count = len(initial_edges)
    
    def search(candidates_idx, current_adj, current_edge_set, current_count):
        nonlocal best_count, best_edges
        nonlocal deadline, start_time
        
        if time.time() > deadline:
            return
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(current_edge_set)
        
        # Upper bound: current_count + remaining candidates
        if current_count + (len(candidates) - candidates_idx) <= best_count:
            return
        
        if candidates_idx >= len(candidates):
            return
        
        u, v = candidates[candidates_idx]
        
        # Option 1: Add edge (u, v)
        if is_triangle_free(current_adj, u, v):
            new_adj = list(current_adj)
            new_edge_set = set(current_edge_set)
            add_edge_to_adj(new_adj, u, v)
            new_edge_set.add((u, v))
            
            # Check if still not 3-colorable
            if is_not_3_colorable(new_adj):
                search(candidates_idx + 1, new_adj, new_edge_set, current_count + 1)
            
            remove_edge_from_adj(new_adj, u, v)
        
        # Option 2: Skip edge (u, v)
        search(candidates_idx + 1, current_adj, current_edge_set, current_count)
    
    # The search space is huge, so we need a smarter approach.
    # Let's try a local search / hill climbing approach instead.
    
    # Start from the initial solution and try to add edges one by one
    current_adj = list(adj)
    current_edge_set = set(initial_edges)
    
    # Try to add each candidate edge if it maintains triangle-free and not-3-colorable
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if (u, v) in current_edge_set:
                    continue
                if not is_triangle_free(current_adj, u, v):
                    continue
                # Try adding
                add_edge_to_adj(current_adj, u, v)
                current_edge_set.add((u, v))
                if is_not_3_colorable(current_adj):
                    improved = True
                else:
                    # Remove it
                    remove_edge_from_adj(current_adj, u, v)
                    current_edge_set.discard((u, v))
    
    if len(current_edge_set) > best_count:
        best_count = len(current_edge_set)
        best_edges = list(current_edge_set)
    
    # Try local search: swap edges to potentially increase count
    # This is complex, so let's try a different strategy.
    
    # Let me try a more systematic branch and bound with better ordering.
    
    # Rebuild candidate list
    current_adj = list(adj)  # Reset to initial
    current_edge_set = set(initial_edges)
    
    # Get all edges that could potentially be added (maintaining triangle-free)
    all_candidates = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) not in current_edge_set:
                all_candidates.append((u, v))
    
    # Order candidates: prefer edges that are "central" (high degree endpoints)
    def edge_score(e):
        u, v = e
        return bin(current_adj[u]).count('1') + bin(current_adj[v]).count('1')
    
    all_candidates.sort(key=edge_score, reverse=True)
    
    # Filter: only keep edges that can be added without creating a triangle
    # This is dynamic, so we do it during search.
    
    # Branch and bound with memoization of 3-colorability state
    # This is still exponential, but let's try with aggressive pruning.
    
    def branch_and_bound(idx, current_adj, current_edge_set, current_count, depth=0):
        nonlocal best_count, best_edges
        nonlocal deadline, start_time
        
        if time.time() > deadline:
            return
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(current_edge_set)
        
        # Prune: even if we add all remaining, can't beat best
        # (rough upper bound)
        remaining = len(all_candidates) - idx
        if current_count + remaining <= best_count:
            return
        
        if idx >= len(all_candidates):
            return
        
        u, v = all_candidates[idx]
        
        # Option 1: Try adding edge
        if is_triangle_free(current_adj, u, v):
            new_adj = list(current_adj)
            new_edge_set = set(current_edge_set)
            add_edge_to_adj(new_adj, u, v)
            new_edge_set.add((u, v))
            
            # Fast 3-colorability check
            if is_not_3_colorable(new_adj):
                branch_and_bound(idx + 1, new_adj, new_edge_set, current_count + 1, depth + 1)
            
            remove_edge_from_adj(new_adj, u, v)
        
        # Option 2: Skip
        branch_and_bound(idx + 1, current_adj, current_edge_set, current_count, depth + 1)
    
    branch_and_bound(0, list(adj), set(initial_edges), len(initial_edges))
    
    # Convert to required format
    result = []
    for u, v in best_edges:
        result.append([u, v])
    
    return result
