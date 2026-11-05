"""Polarity graph C4-free core + optimized extra edge + local search."""
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Generate all 31 points in PG(2,5)
    # Points are 1D subspaces of F_5^3, represented by canonical form
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Canonical form: first nonzero coordinate is 1
                vec = [a, b, c]
                # Find first nonzero
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5)  # modular inverse in F5
                        vec = [(v * inv) % 5 for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Build polarity graph: connect points if they are orthogonal
    # For a point p = (a,b,c) in PG(2,5), the orthogonal complement under the standard dot product
    # Two points p and q are adjacent if p·q = 0 mod 5
    adj = np.zeros((n, n), dtype=int)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append([i, j])
    
    # The polarity graph should have 90 edges and 0 C4s
    base_edges = list(edges)
    base_adj = adj.copy()
    
    # Count C4s in base graph (should be 0)
    def count_c4s(adj_matrix):
        # Number of 4-cycles = (1/8) * sum over all pairs of vertices of (common neighbors)^2 - ... 
        # Actually: number of C4 subgraphs = (1/8) * sum_{u,v} (A^2)_{uv}^2 - ... 
        # More precisely: count of 4-cycles = (1/8) * (sum of (A^2)_{ij} for i!=j choose 2 pattern)
        # Easier: for each pair of vertices, count common neighbors, then sum C(common,2) over all pairs
        # Each C4 is counted 4 times (once per pair of opposite vertices)
        # So C4 count = (1/4) * sum_{u<v} C(common_neighbors(u,v), 2)
        A2 = adj_matrix @ adj_matrix
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                common = A2[i, j]
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 4
    
    base_c4 = count_c4s(base_adj)
    
    # Find all non-edges in the base graph
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if base_adj[i, j] == 0:
                non_edges.append([i, j])
    
    # For each candidate edge to add, count how many new C4s it creates
    # Adding edge (u,v) creates a C4 for each pair of common neighbors of u and v
    # (specifically, for each w that is a common neighbor of u and v, if there's a path u-w-x-v or u-x-w-v)
    # Actually, adding edge (u,v) creates C4s of the form u-w-x-v-u where w,x are common neighbors
    # More precisely: new C4s = number of common neighbors of u and v that are connected through a length-2 path
    # A 4-cycle containing the new edge (u,v) must be u-v-x-y-u where x is neighbor of v, y is neighbor of u, and x-y is an edge
    # So new C4s = number of edges (x,y) where x is neighbor of v (not u) and y is neighbor of u (not v) and x!=y
    # Actually simpler: new C4s involving edge (u,v) = number of common neighbors of u and v... no.
    # A C4 containing edge (u,v): u-v-a-b-u where a~v, b~u, a~b, a!=b, a!=u, b!=v
    # So for each common neighbor a of u and v (wait, a needs to be neighbor of v and b neighbor of u with a~b)
    # Let me just compute: for candidate edge (u,v), new C4s = sum over all pairs (a,b) where a~v, b~u, a~b, a!=b, a!=u, b!=v, and {u,v,a,b} forms a cycle
    # Simpler approach: new C4s = number of 2-paths from u to v that don't use edge (u,v) directly
    # i.e., number of vertices w such that u~w and w~v (common neighbors) -- but that gives triangles, not C4s
    # For C4: u-v-x-y-u. x must be neighbor of v, y must be neighbor of u, and x~y.
    # So new C4s = for each x in N(v)\{u}, for each y in N(u)\{v}, if x~y and x!=y, count 1
    # But each C4 is counted once per edge, so this gives the exact count of new C4s.
    
    def count_new_c4s(adj_matrix, u, v):
        """Count new C4s created by adding edge (u,v)."""
        count = 0
        for x in range(n):
            if x == u or x == v:
                continue
            if adj_matrix[v, x] == 1:  # x is neighbor of v
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    if adj_matrix[u, y] == 1 and adj_matrix[x, y] == 1:
                        # u-v-x-y-u is a 4-cycle
                        count += 1
        # Each C4 is counted exactly once (once per the edge u-v we're adding)
        # Wait, actually each C4 u-v-x-y-u is counted once when we fix the edge (u,v)
        # But we might double count if we consider both orientations... no, we fixed u,v as the added edge
        # Actually this counts each 4-cycle containing edge (u,v) exactly once
        return count
    
    # Find the edge that creates the fewest new C4s
    best_edge = None
    best_new_c4s = float('inf')
    
    for edge in non_edges:
        u, v = edge
        new_c4s = count_new_c4s(base_adj, u, v)
        if new_c4s < best_new_c4s:
            best_new_c4s = new_c4s
            best_edge = edge
    
    # Add the best edge
    final_edges = base_edges + [best_edge]
    final_adj = base_adj.copy()
    u, v = best_edge
    final_adj[u, v] = 1
    final_adj[v, u] = 1
    
    # Verify we have exactly 91 edges
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    
    # Check for duplicates
    edge_set = set()
    for e in final_edges:
        key = tuple(sorted(e))
        assert key not in edge_set, f"Duplicate edge {e}"
        edge_set.add(key)
    
    # Verify all vertices are in range
    for e in final_edges:
        assert 0 <= e[0] <= 30 and 0 <= e[1] <= 30
    
    # Count total C4s for verification
    total_c4 = count_c4s(final_adj)
    
    # Try local search to improve: try swapping edges to reduce C4 count
    # Time check
    if time.time() > deadline - 20:
        return final_edges
    
    # Local search: try to find a better configuration
    # We can try removing one edge and adding another to reduce C4 count
    current_adj = final_adj.copy()
    current_edges = final_edges[:]
    current_c4 = total_c4
    
    for iteration in range(100):
        if time.time() > deadline - 10:
            break
        
        improved = False
        # Try all possible swaps: remove edge e1, add edge e2
        # This is expensive, so sample
        import random
        random.seed(42 + iteration)
        
        for _ in range(50):
            # Pick a random edge to remove
            idx = random.randint(0, len(current_edges) - 1)
            e1 = current_edges[idx]
            u1, v1 = e1
            
            # Find a random non-edge to add
            attempts = 0
            while attempts < 20:
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2 or current_adj[u2, v2] == 1 or (u2, v2) == (u1, v1) or (v2, u2) == (u1, v1):
                    attempts += 1
                    continue
                e2 = [min(u2, v2), max(u2, v2)]
                
                # Try the swap
                new_adj = current_adj.copy()
                new_adj[u1, v1] = 0
                new_adj[v1, u1] = 0
                new_adj[u2, v2] = 1
                new_adj[v2, u2] = 1
                
                new_c4 = count_c4s(new_adj)
                if new_c4 < current_c4:
                    # Accept the swap
                    current_adj = new_adj
                    current_edges[idx] = e2
                    current_c4 = new_c4
                    improved = True
                    break
                attempts += 1
        
        if not improved:
            break
    
    # Convert back to list of edges
    final_result = []
    for i in range(n):
        for j in range(i+1, n):
            if current_adj[i, j] == 1:
                final_result.append([i, j])
    
    assert len(final_result) == 91, f"Expected 91 edges, got {len(final_result)}"
    
    return final_result
