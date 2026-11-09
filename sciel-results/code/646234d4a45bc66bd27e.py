"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 115.0  # safety margin before 150s kill
    
    n = 16
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_to_idx = {e: k for k, e in enumerate(all_edges)}
    m_total = len(all_edges)
    
    # Build conflict graph: edge e1 conflicts with e2 if they share a vertex or form a triangle
    # Two edges (a,b) and (c,d) form a triangle if {a,b,c,d} has size 2 and the third vertex connects
    # Actually: (a,b) and (c,d) form a triangle if there's a vertex v such that v is adjacent to both
    # endpoints in a triangle. More precisely, edges (a,b) and (c,d) form a triangle if:
    # - They share a vertex, say b=c, and edge (a,d) exists -> triangle a-b-d
    # - Or they don't share a vertex but {a,b} ∩ {c,d} = {} and one of the cross-edges exists
    # Wait, two edges form a triangle only if they share exactly one vertex AND the third edge exists.
    # Actually no: two edges (u,v) and (w,x) form a triangle if there exist vertices such that 
    # all three pairs are edges. Two edges can be part of a triangle if:
    # Case 1: They share a vertex. (u,v) and (v,w) -> need (u,w)
    # Case 2: They don't share a vertex. (u,v) and (w,x) with {u,v}∩{w,x}=∅. 
    #   For these to be in a triangle, we'd need a third edge connecting them, but a triangle 
    #   has exactly 3 vertices, so two disjoint edges can't be in the same triangle.
    # So: two edges conflict (can't both be in triangle-free graph) iff they share a vertex AND
    # the third edge completing the triangle also exists. But wait - we're selecting edges, so 
    # we need: if we select (u,v) and (v,w), we CANNOT also select (u,w).
    # 
    # Better approach: build a "conflict" relation where edge e1 and e2 are in conflict if
    # selecting both would create a triangle. This happens iff they share a vertex v, and the
    # other endpoints u,w have edge (u,w) in our selected set. But this is 3-way, not 2-way.
    #
    # Alternative: use the standard approach. A set of edges is triangle-free iff for every 
    # triangle (a,b,c), at most 2 of the 3 edges are selected. This is a 3-uniform constraint.
    #
    # For the conflict graph / maximum clique approach:
    # Node = potential edge. Two nodes are adjacent if the two edges CAN coexist in a 
    # triangle-free graph. Two edges can coexist unless they share a vertex AND the third 
    # edge of the triangle is also selected. But that's not a pairwise constraint.
    #
    # Hmm, this doesn't directly reduce to maximum clique. Let me think differently.
    #
    # Actually, the correct formulation: two edges (u,v) and (w,x) are "incompatible" (cannot
    # both be in a triangle-free graph) if and only if they share exactly one vertex, say v=w,
    # AND the edge (u,x) is also present. But since we're choosing which edges to include, 
    # the incompatibility depends on a third edge. So it's not a simple pairwise conflict.
    #
    # Let me use a different strategy: backtracking search with pruning.
    # 
    # Strategy: 
    # 1. Start with a known good graph (Mycielski of C5 = Grötzsch, 11 vertices, 20 edges)
    #    plus 5 isolated vertices.
    # 2. Try to add edges one by one, checking triangle-freeness and 3-colorability.
    # 3. Use local search / simulated annealing to find denser graphs.
    #
    # Better: Use a branch-and-bound or greedy + local improvement approach.
    
    # First, let's define helper functions
    
    def is_triangle_free(adj):
        """Check if adjacency matrix represents a triangle-free graph."""
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    def is_3_colorable(adj, timeout=2.0):
        """Check 3-colorability via backtracking with forward checking."""
        t_start = time.time()
        color = [-1] * n
        
        # Order vertices by degree (descending) for better pruning
        degrees = [sum(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        pos = {v: i for i, v in enumerate(order)}
        
        # For each vertex, precompute neighbors that come earlier in order
        def backtrack(idx):
            if time.time() - t_start > timeout:
                return False
            if idx == n:
                return True
            v = order[idx]
            # Get colors used by already-colored neighbors
            used = set()
            for u in range(n):
                if adj[v][u] and color[u] != -1:
                    used.add(color[u])
            for c in range(3):
                if c not in used:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    # Start with Grötzsch graph + isolated vertices
    adj = [[0]*n for _ in range(n)]
    # Grötzsch: C5 on 0-4, vertices 5-9 are Mycielski expansion, vertex 10 is top
    for i in range(5):
        a, b = i, (i+1) % 5
        adj[a][b] = adj[b][a] = 1
        adj[a][5+b] = adj[5+b][a] = 1
        adj[b][5+a] = adj[5+a][b] = 1
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    
    best_adj = [row[:] for row in adj]
    best_count = count_edges(adj)
    
    # Now try to add edges greedily, then do local search
    # Try adding each possible missing edge
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
    
    # Greedy addition
    random.shuffle(missing)
    for (i, j) in missing:
        if time.time() > deadline - 10:
            break
        adj[i][j] = adj[j][i] = 1
        if is_triangle_free(adj) and is_3_colorable(adj, timeout=1.0):
            new_count = count_edges(adj)
            if new_count > best_count:
                best_count = new_count
                best_adj = [row[:] for row in adj]
        else:
            adj[i][j] = adj[j][i] = 0
    
    # Local search: try to swap edges or add/remove to improve
    def local_search(adj, iters=5000):
        """Try to add edges via local search with 3-colorability check."""
        adj = [row[:] for row in adj]
        count = count_edges(adj)
        for it in range(iters):
            if time.time() > deadline - 5:
                break
            # Try adding a random missing edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j or adj[i][j]:
                continue
            adj[i][j] = adj[j][i] = 1
            if is_triangle_free(adj) and is_3_colorable(adj, timeout=0.5):
                count += 1
            else:
                adj[i][j] = adj[j][i] = 0
        return adj, count
    
    adj_ls, count_ls = local_search(adj, iters=5000)
    if count_ls > best_count:
        best_count = count_ls
        best_adj = adj_ls
    
    # Try more aggressive: start from scratch with different initializations
    # Use the MIS-based approach: build graph by connecting new vertices to independent sets
    
    def build_with_mis():
        """Build triangle-free graph by adding vertices connected to independent sets."""
        adj = [[0]*n for _ in range(n)]
        # Start with C5
        for i in range(5):
            a, b = i, (i+1) % 5
            adj[a][b] = adj[b][a] = 1
        
        # Add vertices 5-15 one by one, connecting to a maximal independent set
        for v in range(5, n):
            # Find independent sets in current graph
            # Simple: try to connect v to a large independent set
            # Independent set: no two connected vertices
            best_set = []
            # Try all subsets of current vertices (small enough)
            current_vertices = list(range(v))
            # Greedy independent set
            for attempt in range(10):
                random.shuffle(current_vertices)
                indep = []
                for u in current_vertices:
                    if all(not adj[u][w] for w in indep):
                        indep.append(u)
                if len(indep) > len(best_set):
                    best_set = indep
            
            for u in best_set:
                adj[v][u] = adj[u][v] = 1
        
        return adj
    
    # Try multiple MIS-based constructions
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        adj_trial = build_with_mis()
        if not is_triangle_free(adj_trial):
            continue
        if not is_3_colorable(adj_trial, timeout=1.0):
            continue
        # Improve with local search
        adj_improved, count_improved = local_search(adj_trial, iters=3000)
        if count_improved > best_count:
            best_count = count_improved
            best_adj = adj_improved
    
    # Final: try to add edges to best_adj
    adj_final = [row[:] for row in best_adj]
    missing_final = [(i, j) for i in range(n) for j in range(i+1, n) if not adj_final[i][j]]
    random.shuffle(missing_final)
    for (i, j) in missing_final:
        if time.time() > deadline - 2:
            break
        adj_final[i][j] = adj_final[j][i] = 1
        if is_triangle_free(adj_final) and is_3_colorable(adj_final, timeout=0.5):
            best_count = count_edges(adj_final)
            best_adj = adj_final
        else:
            adj_final[i][j] = adj_final[j][i] = 0
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
