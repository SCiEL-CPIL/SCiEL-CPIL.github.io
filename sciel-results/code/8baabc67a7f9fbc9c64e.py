import time
import sys

def construct():
    n = 16
    # Precompute all possible edges as a list of (u, v) with u < v
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    
    m = len(all_edges) # 120 edges
    
    # Map edge index to (u, v)
    edge_uv = [None] * m
    for i, (u, v) in enumerate(all_edges):
        edge_uv[i] = (u, v)
        
    # Precompute for each edge i, the set of edges j that form a triangle with i.
    # If we include edge i, we cannot include any edge in conflict[i].
    conflict = [0] * m
    # Also, if we include edge i, we cannot include edge k if k is (u, w) or (v, w) where w is connected to both?
    # No, triangle is (u,v), (v,w), (w,u).
    # So for edge (u,v), it conflicts with any edge (x,y) such that {x,y} intersects {u,v} and the third vertex is in the intersection of neighborhoods?
    # Actually, simpler:
    # Edge i=(u,v) conflicts with edge j=(x,y) if:
    # 1. x,y are both in {u,v}? No, that's the same edge.
    # 2. One of x,y is u or v, and the other is w. Then we need edge (u,w) or (v,w) to exist? No.
    # A triangle is formed by 3 edges. 
    # If we pick edge (u,v), we cannot pick any edge (u,w) AND (v,w) simultaneously.
    # This is not a simple pairwise conflict in the line graph sense unless we view it as:
    # The graph is triangle-free iff the set of edges contains no 3 edges that form a triangle.
    # This is a 3-uniform hypergraph independent set problem, not a simple graph independent set.
    # However, the directive says "independent set search on the line graph...". 
    # Wait, the line graph of K_n has vertices as edges of K_n. Two vertices in L(K_n) are adjacent if the edges in K_n share a vertex.
    # An independent set in L(K_n) is a set of edges in K_n that are pairwise disjoint (a matching).
    # That is NOT what we want. We want a triangle-free graph.
    # A triangle-free graph is a set of edges such that no 3 edges form a triangle.
    # This is an independent set in the 3-uniform hypergraph of triangles.
    # The directive's description "independent set on line graph" is mathematically incorrect for triangle-free graphs (that would be matchings).
    # However, I must follow the directive's *intent*: "branch-and-bound to directly find the largest such set".
    # I will implement a branch and bound for the 3-uniform hypergraph independent set (triangle-free).
    
    # To make B&B efficient, we need good ordering and pruning.
    
    # Initial Lower Bound: Grötzsch Graph + 5 isolated vertices
    # Grötzsch graph edges (11 vertices 0..10):
    # Cycle 0-1-2-3-4-0
    # Mycielski copies 5..9
    # Root 10
    groetzsch_edges = []
    for i in range(5):
        groetzsch_edges.append((i, (i + 1) % 5))
    for i in range(5):
        u = i
        w = (i + 1) % 5
        groetzsch_edges.append((u, 5 + w))
        groetzsch_edges.append((w, 5 + u))
    for i in range(5):
        groetzsch_edges.append((5 + i, 10))
        
    # Extend Grötzsch to maximal triangle-free on 16 vertices
    def extend_to_maximal(edge_set):
        adj = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Try to add all possible edges
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        # Check if (u,v) creates a triangle
                        # Triangle if (adj[u] & adj[v]) != 0
                        if (adj[u] & adj[v]) == 0:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True
                            
        # Convert back to edge list
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
        return edges

    def is_3_colorable(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        color = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            # Heuristic: try colors 0, 1, 2
            # Check constraints
            neighbors = adj[v]
            c0 = False
            c1 = False
            c2 = False
            # Check if any neighbor has color 0, 1, or 2
            # We can iterate neighbors
            for u in range(n):
                if (neighbors >> u) & 1:
                    if color[u] == 0: c0 = True
                    if color[u] == 1: c1 = True
                    if color[u] == 2: c2 = True
                    if c0 and c1 and c2:
                        return False # No color available
            
            if not c0:
                color[v] = 0
                if backtrack(v + 1): return True
                color[v] = -1
            if not c1:
                color[v] = 1
                if backtrack(v + 1): return True
                color[v] = -1
            if not c2:
                color[v] = 2
                if backtrack(v + 1): return True
                color[v] = -1
            return False
        
        return backtrack(0)

    # Get initial best
    initial_edges = extend_to_maximal(groetzsch_edges)
    best_edges = initial_edges
    best_count = len(initial_edges)
    
    # Branch and Bound
    # We iterate through edges.
    # State: index i, current_adj (list of ints), current_edge_count
    # Pruning:
    # 1. If current_edge_count + (m - i) <= best_count, return.
    # 2. If current graph is not triangle-free (shouldn't happen if we prune correctly), return.
    # 3. If current graph is not 3-colorable, we can try to add more edges?
    #    Actually, we want to MAXIMIZE edges.
    #    If we have a valid graph (triangle-free, not 3-colorable), we record it.
    #    But we can still add more edges to it as long as it stays triangle-free.
    #    So we don't stop. We continue branching.
    #    However, if we add an edge and it becomes 3-colorable, that's fine, we just haven't found a solution in that branch yet.
    #    We only care about the final leaf or any node that is a valid solution?
    #    No, we want the MAXIMAL number of edges.
    #    So we should only evaluate the solution when we've decided on all edges?
    #    Or we can evaluate at any point?
    #    If we have a valid graph with k edges, and we add an edge, it might become invalid (3-colorable).
    #    But we want to maximize edges.
    #    So we should explore adding edges as much as possible.
    #    The search tree is: for each edge, include or exclude.
    #    At the leaf, we have a specific graph.
    #    We check if it's triangle-free (it will be, by construction) and not 3-colorable.
    #    If yes, update best.
    #    
    #    Optimization:
    #    If at any point the current graph is not 3-colorable, we can still add more edges.
    #    But adding edges might make it 3-colorable? No, adding edges preserves non-3-colorability?
    #    Wait. If G is not 3-colorable, is G+e not 3-colorable?
    #    Yes. If G+e is 3-colorable, then G is a subgraph of a 3-colorable graph, so G is 3-colorable.
    #    Contradiction.
    #    So, if the current graph is not 3-colorable, ANY supergraph is also not 3-colorable.
    #    This is a HUGE optimization!
    #    
    #    So, if we reach a node where the current graph is not 3-colorable, we can greedily add ALL remaining edges that do not create triangles!
    #    Because the result will still be not 3-colorable, and we want to maximize edges.
    
    start_time = time.time()
    timeout = 110
    
    def solve(i, adj, count, is_not_3c):
        nonlocal best_edges, best_count
        
        if time.time() - start_time > timeout:
            return
            
        # Pruning: even if we add all remaining edges, can we beat best?
        # Max possible edges = count + (m - i)
        if count + (m - i) <= best_count:
            return
            
        # If we have already found a non-3-colorable graph, we can just add all possible remaining edges
        if is_not_3c:
            # Add all edges from i to m-1 that don't create triangles
            for j in range(i, m):
                u, v = edge_uv[j]
                if not ((adj[u] >> v) & 1):
                    if (adj[u] & adj[v]) == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        count += 1
            
            if count > best_count:
                # Reconstruct edges
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            edges.append((u, v))
                best_count = count
                best_edges = edges
            return
            
        # If i == m, check if we have a solution
        if i == m:
            if not is_3_colorable_from_adj(adj):
                if count > best_count:
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj[u] >> v) & 1:
                                edges.append((u, v))
                    best_count = count
                    best_edges = edges
            return
            
        u, v = edge_uv[i]
        
        # Option 1: Include edge (u, v)
        # Check if it creates a triangle
        if (adj[u] & adj[v]) == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check if new graph is not 3-colorable
            # We can check 3-colorability here. It's expensive.
            # But if it IS 3-colorable, we still need to explore.
            # If it is NOT 3-colorable, we can shortcut.
            
            # Optimization: Only check 3-colorability if count is high enough to potentially beat best?
            # Or always check?
            # Let's check. If it becomes not 3-colorable, we set flag.
            # To save time, we can skip the check if we are far from best?
            # No, the shortcut is valuable.
            
            # Heuristic: Check 3-colorability only if count + (m-i) > best_count (which is true by pruning)
            # And maybe if count is large.
            
            # Let's just check.
            new_is_not_3c = not is_3_colorable_from_adj(adj)
            
            solve(i + 1, adj, count + 1, new_is_not_3c)
            
            # Undo
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
        # Option 2: Exclude edge (u, v)
        solve(i + 1, adj, count, is_not_3c)
        
    def is_3_colorable_from_adj(adj):
        # Fast check using bitmasks
        # We need to color vertices 0..15
        # We can use a recursive backtracking with bitsets for available colors
        # Colors: 0, 1, 2
        # For each vertex v, available colors are those not used by neighbors.
        
        color = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            
            # Determine available colors
            # Neighbors of v
            neighbors = adj[v]
            
            # Check which colors are forbidden
            # c0 forbidden if any neighbor has color 0
            # c1 forbidden if any neighbor has color 1
            # c2 forbidden if any neighbor has color 2
            
            # We can precompute masks for neighbors of each color?
            # Since we are in recursion, we can just loop neighbors.
            
            forbid0 = False
            forbid1 = False
            forbid2 = False
            
            for u in range(n):
                if (neighbors >> u) & 1:
                    if color[u] == 0: forbid0 = True
                    if color[u] == 1: forbid1 = True
                    if color[u] == 2: forbid2 = True
                    if forbid0 and forbid1 and forbid2:
                        return False
            
            if not forbid0:
                color[v] = 0
                if backtrack(v + 1): return True
                color[v] = -1
            if not forbid1:
                color[v] = 1
                if backtrack(v + 1): return True
                color[v] = -1
            if not forbid2:
                color[v] = 2
                if backtrack(v + 1): return True
                color[v] = -1
            return False
        
        return backtrack(0)

    # Start B&B
    # Initial adj is empty
    adj = [0] * n
    solve(0, adj, 0, False)
    
    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    edges = construct()
    print(f"Found graph with {len(edges)} edges")
