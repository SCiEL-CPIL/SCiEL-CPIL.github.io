"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _max_triangle_free_induced(adj, n):
    """Find the largest triangle-free induced subgraph using branch and bound."""
    # For each vertex, compute its non-neighbor set (including itself)
    # A set S is triangle-free iff for every vertex v in S, the neighbors of v in S form an independent set
    
    # Use a heuristic: try to find large independent sets in the "triangle graph"
    # Actually, let's use a direct approach: for each candidate set, check if triangle-free
    
    # Better approach: use the fact that a set is triangle-free iff it contains no triangle.
    # We can use branch and bound on subsets.
    
    # Precompute for each vertex, the set of vertices that form a triangle with it
    # tri_neighbors[v] = {u : v,u,w form a triangle for some w}
    # Actually, a set S is triangle-free iff for all v in S, N(v) ∩ S is independent
    
    # Let's use a simpler approach: enumerate subsets by size (descending)
    # and check if any subset of that size is triangle-free.
    
    # For n=18, this is feasible with bit masks.
    # Represent adjacency as bit masks.
    adj_mask = [0] * n
    for v in range(n):
        mask = 0
        for u in range(n):
            if u != v and adj[v][u]:
                mask |= (1 << u)
        adj_mask[v] = mask
    
    # Precompute: for each pair (u,v), the common neighbors
    # tri[v] = set of vertices w such that {v,w} is an edge and they have a common neighbor
    # Actually, for checking if a set S is triangle-free:
    # S is triangle-free iff for every edge (u,v) in S, N(u) ∩ N(v) ∩ S = empty
    
    # Let's use a different approach: for each vertex v, define conflict[v] = {u : u and v share a common neighbor}
    # Then S is triangle-free iff S is an independent set in the conflict graph? No, that's not right.
    
    # S is triangle-free iff there is no triangle in S. A triangle is {a,b,c} where all three are pairwise adjacent.
    # So S is triangle-free iff for no three vertices a<b<c in S are all three pairwise adjacent.
    
    # Let's precompute for each pair (a,b) that are adjacent, the set of c > b such that (a,c) and (b,c) are also edges.
    # tri_with[a][b] = set of c > b such that a-b-c is a triangle
    
    # For branch and bound: we want the largest subset with no triangle.
    # This is equivalent to the maximum size of a subset with no three vertices forming a triangle.
    
    # Use a recursive approach with pruning.
    # State: current set (as bitmask), next vertex to consider, current size
    # Pruning: if current size + remaining vertices <= best found, prune
    
    best = [0]
    
    def is_triangle_free(mask):
        """Check if the set represented by mask is triangle-free."""
        # Get vertices in the set
        verts = []
        for v in range(n):
            if mask & (1 << v):
                verts.append(v)
        # Check all triples
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                for k in range(j+1, len(verts)):
                    a, b, c = verts[i], verts[j], verts[k]
                    if adj[a][b] and adj[b][c] and adj[a][c]:
                        return False
        return True
    
    # This is too slow. Let's use a smarter approach.
    
    # Precompute: for each pair (a,b) with a<b, if adj[a][b], compute common_neighbors[a][b]
    # Then a set S is triangle-free iff for all a<b in S with adj[a][b], 
    # common_neighbors[a][b] ∩ S = empty (excluding a and b themselves)
    
    # Let's use a different strategy: since n=18 is small, we can use the following:
    # For each vertex v, let N(v) be its neighbors. 
    # A set S is triangle-free iff for every v in S, N(v) ∩ S is an independent set.
    
    # Actually, the most efficient for n=18: use the complement.
    # A set S is NOT triangle-free iff it contains a triangle.
    # We want the largest S with no triangle.
    
    # Let's just do a brute force over subsets using bit operations, but smartly.
    # For n=18, there are 2^18 = 262144 subsets. We can check each one.
    
    # Precompute triangle masks: for each triple (a,b,c), create a mask.
    # Then for each subset mask, check if it contains any triangle mask.
    
    # Better: precompute for each mask whether it's triangle-free using DP.
    # tri_free[mask] = True if mask is triangle-free
    # tri_free[mask] = tri_free[mask without v] AND (no triangle formed by adding v to mask without v)
    
    # When adding vertex v to set S, a triangle is formed iff there exist u,w in S such that 
    # (u,v) and (w,v) and (u,w) are all edges.
    # So: when adding v to S, check if N(v) ∩ S contains an edge.
    
    # Let's precompute: for each v, adj_mask[v] (neighbors of v as bitmask)
    # For each subset S, to check if adding v makes it non-triangle-free:
    #   common = adj_mask[v] & S  (neighbors of v in S)
    #   Check if common contains an edge: for each u in common, check if adj_mask[u] & common != 0
    
    # This is still potentially slow. Let's just enumerate all subsets and check.
    
    # For n=18, 2^18 = 262144 subsets. For each, checking triangle-free is O(n^3) worst case = 5832.
    # Total: 262144 * 5832 ≈ 1.5 billion. Too slow in Python.
    
    # Let's use a smarter approach: precompute for each pair (u,v) whether they're adjacent,
    # and for each vertex v, the set of edges in N(v).
    
    # Actually, let's use the following: 
    # A set S is triangle-free iff for every v in S, the induced subgraph on N(v) ∩ S has no edges.
    
    # Let me just use a simpler heuristic for the score, and optimize for it.
    # The score is the largest triangle-free induced subgraph.
    # A lower bound: the size of the largest independent set (since independent sets are triangle-free).
    # An upper bound: n (trivial).
    
    # For the purpose of local search, let's use a fast approximation:
    # The largest triangle-free induced subgraph is at least the independence number.
    # Let's compute the independence number as a proxy, but actually we need the exact value.
    
    # Let me implement a proper branch and bound for maximum triangle-free induced subgraph.
    
    # Key insight: a set S is triangle-free iff it contains no three mutually adjacent vertices.
    # This is the complement of the "3-clique" problem, but for induced subgraphs.
    
    # Let's use: S is triangle-free iff for all v in S, N_S(v) is an independent set.
    
    # Branch and bound:
    # - Start with all vertices
    # - If the current set is triangle-free, update best
    # - Otherwise, find a triangle in the set, and branch: remove one vertex from the triangle
    
    # Find a triangle in a set:
    def find_triangle(verts):
        """Find a triangle in the induced subgraph on verts. Return (a,b,c) or None."""
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                for k in range(j+1, len(verts)):
                    a, b, c = verts[i], verts[j], verts[k]
                    if adj[a][b] and adj[b][c] and adj[a][c]:
                        return (a, b, c)
        return None
    
    # This is O(n^3) per call. For n=18, that's 816 triples. Acceptable.
    
    # Branch and bound:
    def bt(verts, depth):
        if not verts:
            return 0
        # Check if verts is triangle-free
        tri = find_triangle(verts)
        if tri is None:
            return len(verts)
        # Branch: remove one of the three vertices from the triangle
        a, b, c = tri
        # Try removing each of a, b, c
        best = 0
        for v in (a, b, c):
            new_verts = [x for x in verts if x != v]
            val = bt(new_verts, depth + 1)
            if val > best:
                best = val
        return best
    
    # This could be exponential. Let's add memoization or pruning.
    # Actually, for n=18, let's just use a greedy approach for speed in local search,
    # and only do exact computation at the end.
    
    # For local search, use a fast heuristic: 
    # The largest triangle-free induced subgraph >= independence number
    # And <= n - (minimum vertex cover of triangles) ... not helpful
    
    # Let's use: for each vertex, greedily build a triangle-free set.
    # Or: use the fact that a random set of size k is likely triangle-free if the graph is sparse.
    
    # Actually, let's just compute the exact value using a smarter method.
    # For n=18, we can use the following:
    # Precompute all triangles. There are at most C(18,3) = 816 triangles.
    # We want the largest subset that hits no triangle. This is the "maximum subset avoiding all triangles"
    # which is equivalent to: the complement of the minimum hitting set of triangles.
    # The minimum hitting set of triangles is the minimum vertex cover of the hypergraph of triangles.
    
    # So: max triangle-free induced subgraph size = 18 - min vertex cover of triangle hypergraph.
    
    # Finding the minimum vertex cover of a 3-uniform hypergraph is NP-hard, but for 816 hyperedges on 18 vertices,
    # we can use branch and bound.
    
    # Let's collect all triangles.
    triangles = []
    for a in range(n):
        for b in range(a+1, n):
            if adj[a][b]:
                for c in range(b+1, n):
                    if adj[a][c] and adj[b][c]:
                        triangles.append((a, b, c))
    
    if not triangles:
        return n  # No triangles, so all n vertices form a triangle-free induced subgraph
    
    # Now find the minimum vertex cover of these triangles.
    # Use branch and bound.
    
    best_cover = [len(triangles)]  # Upper bound: one vertex per triangle
    
    def cover_bt(remaining_tris, used, depth):
        if not remaining_tris:
            if len(used) < best_cover[0]:
                best_cover[0] = len(used)
            return
        if len(used) >= best_cover[0]:
            return
        # Pick a triangle
        tri = remaining_tris[0]
        rest = remaining_tris[1:]
        # Try adding each vertex of the triangle to the cover
        for v in tri:
            if v not in used:
                new_used = used | {v}
                # Remove all triangles that contain v
                new_remaining = [t for t in rest if v not in t]
                cover_bt(new_remaining, new_used, depth + 1)
    
    cover_bt(triangles, frozenset(), 0)
    return n - best_cover[0]

def construct():
    n = 18
    deadline = time.time() + 100  # 100 second budget
    
    # Start with the tripartite graph K6,6,6
    # Parts: 0-5, 6-11, 12-17
    # Edges: all pairs between different parts
    adj = [[False]*n for _ in range(n)]
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = True
                    adj[b][a] = True
    
    # Convert to edge list
    def to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    # Check K4-free
    def is_k4_free(adj):
        for a in range(n):
            for b in range(a+1, n):
                if adj[a][b]:
                    for c in range(b+1, n):
                        if adj[a][c] and adj[b][c]:
                            for d in range(c+1, n):
                                if adj[a][d] and adj[b][d] and adj[c][d]:
                                    return False
        return True
    
    # Initial score
    current_score = _max_triangle_free_induced(adj, n)
    best_adj = [row[:] for row in adj]
    best_score = current_score
    
    # Local search: flip edges (add or remove) and check K4-free constraint
    # We can only add an edge if it doesn't create a K4
    # We can always remove an edge
    
    random.seed(42)
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        # Pick a random pair of vertices
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        if adj[u][v]:
            # Try removing the edge
            adj[u][v] = False
            adj[v][u] = False
            new_score = _max_triangle_free_induced(adj, n)
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
            else:
                # Revert
                adj[u][v] = True
                adj[v][u] = True
        else:
            # Try adding the edge, but only if it doesn't create a K4
            # Check: for all w, if adj[u][w] and adj[v][w], then w cannot be connected to both
            # Actually, adding (u,v) creates a K4 iff there exist w,x such that 
            # u-w, u-x, v-w, v-x, w-x are all edges, plus u-v
            # Simpler: after adding (u,v), check if any K4 is formed
            adj[u][v] = True
            adj[v][u] = True
            
            if not is_k4_free(adj):
                adj[u][v] = False
                adj[v][u] = False
                continue
            
            new_score = _max_triangle_free_induced(adj, n)
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
            else:
                # Revert
                adj[u][v] = False
                adj[v][u] = False
    
    # Return the best graph found
    return to_edges(best_adj)
# EVOLVE-BLOCK-END
