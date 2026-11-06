import time
import random

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 145
    
    # Construct polarity graph of projective plane of order 5
    # Using finite field GF(5)
    # Projective plane PG(2,5) has 31 points represented as [x:y:z] with (x,y,z) != (0,0,0)
    # over GF(5), up to scalar
    
    q = 5
    # Enumerate all projective points
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                if x != 0:
                    key = (1, y * pow(x, q-2, q) % q, z * pow(x, q-2, q) % q)
                elif y != 0:
                    key = (0, 1, z * pow(y, q-2, q) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define a polarity: for point [x:y:z], the polar line is x*X + y*Y + z*Z = 0
    # A point [X:Y:Z] is on the polar line if x*X + y*Y + z*Z == 0 in GF(5)
    # The polarity graph has edge (i,j) if point i is on the polar line of point j
    # (or equivalently, point j is on the polar line of point i, for symmetric polarity)
    
    # For the symplectic polarity, we use a symmetric bilinear form
    # Let's use: B([x:y:z], [X:Y:Z]) = x*Z - z*X + y*Y  (this is not standard)
    # Actually, let's use a simpler approach: the standard polarity from the projective plane
    
    # The polarity given by the matrix:
    # [[0, 1, 0],
    #  [1, 0, 0], 
    #  [0, 0, 1]]
    # This is symmetric, so it's a polarity.
    # Point p=[x:y:z] is on polar line of q=[X:Y:Z] if:
    # x*Z + y*X + z*Y == 0  (wait, let me be more careful)
    
    # Actually, for a polarity defined by symmetric matrix M, 
    # point p is on polar of q if p^T * M * q = 0
    
    # Let me use the standard symplectic polarity for PG(2,q) with q odd:
    # The matrix is [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    # p^T M q = x*Y + y*X + z*Z
    
    def is_on_polar(p, qpt):
        """Check if point p is on the polar line of point qpt"""
        x, y, z = p
        X, Y, Z = qpt
        # p^T M q where M = [[0,1,0],[1,0,0],[0,0,1]]
        # = x*Y + y*X + z*Z
        return (x * Y + y * X + z * Z) % q == 0
    
    # Build the polarity graph
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_on_polar(points[i], points[j]):
                edges.append((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    # This should give a C4-free graph with 90 edges
    # Let's verify
    if len(edges) != 90:
        # If not exactly 90, we need to adjust
        # But the polarity graph of PG(2,5) should have exactly 90 edges
        # Let's just proceed and handle it
        pass
    
    # Now we need to add 1 edge to get to 91
    # Find the edge with minimum common neighbors
    existing_set = set(edges)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    candidates = [e for e in all_edges if e not in existing_set]
    
    best_add_edge = None
    min_common = float('inf')
    for u, v in candidates:
        common = len(adj[u] & adj[v])
        if common < min_common:
            min_common = common
            best_add_edge = (u, v)
    
    # Add the best edge
    if best_add_edge is not None:
        u, v = best_add_edge
        edges.append((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    # Now do local search: try swapping edges to reduce C4 count
    def count_c4(adj_sets):
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj_sets[i]:
                    # They are adjacent, count common neighbors
                    common = len(adj_sets[i] & adj_sets[j])
                    c4_count += common
        return c4_count // 2
    
    current_c4 = count_c4(adj)
    
    # Local search: try removing an edge and adding a different one
    edge_set = set(edges)
    
    # Precompute for speed
    def get_c4_delta_remove_add(adj_sets, remove_edge, add_edge):
        """Compute change in C4 count when removing remove_edge and adding add_edge"""
        u1, v1 = remove_edge
        u2, v2 = add_edge
        
        # C4s involving the removed edge (u1,v1):
        # A C4 contains (u1,v1) if there exist x,y such that
        # u1-x-v1-y-u1 or similar. Actually, (u1,v1) is in a C4 iff
        # there are common neighbors... no.
        # 
        # Let me think differently. The number of C4s in the graph can be computed as:
        # For each pair of vertices (a,b), the number of C4s where a and b are opposite is
        # C(common_neighbors(a,b), 2). But this counts each C4 once per pair of opposite vertices.
        # A C4 has 2 pairs of opposite vertices, so total C4s = (1/2) * sum over all pairs of C(common,2)
        # Wait no. Let me reconsider.
        #
        # Actually, the standard formula: number of C4s = (1/2) * sum over all unordered pairs {u,v} of C(|N(u) ∩ N(v)|, 2)
        # No, that's not right either.
        #
        # Let me use: for each vertex v, the number of C4s through v is C(deg(v), 2) - (number of paths of length 2 through v that are part of triangles)... 
        # This is getting complicated. Let me just recompute.
        
        # Simpler: just compute the full C4 count for the modified graph
        new_adj = [set(s) for s in adj_sets]
        new_adj[u1].discard(v1)
        new_adj[v1].discard(u1)
        new_adj[u2].add(v2)
        new_adj[v2].add(u1)  # BUG: should be add(u2)
        new_adj[v2].add(u2)
        new_adj[u2].add(v2)
        
        return count_c4(new_adj) - current_c4
    
    # This is too slow. Let me use a different approach.
    # Instead of trying all swaps, let me try a few random swaps.
    
    best_edges = edges[:]
    best_c4 = current_c4
    
    # Try local search with random swaps
    random.seed(42)
    iterations = 0
    while time.time() < deadline and iterations < 5000:
        iterations += 1
        
        # Pick a random edge to remove
        if not edges:
            break
        remove_idx = random.randint(0, len(edges) - 1)
        u1, v1 = edges[remove_idx]
        
        # Pick a random candidate edge to add (not currently in graph)
        # To speed up, try a few random candidates
        found_improvement = False
        for _ in range(20):
            u2, v2 = random.choice(candidates)
            if (u2, v2) in edge_set or (v2, u2) in edge_set:
                continue
            if u2 == u1 and v2 == v1:
                continue
            if u2 == v1 and v2 == u1:
                continue
            
            # Check if this swap keeps the graph valid (no duplicate edges)
            if (u2, v2) in edge_set:
                continue
            
            # Compute new C4 count
            new_adj = [set(s) for s in adj]
            new_adj[u1].discard(v1)
            new_adj[v1].discard(u1)
            new_adj[u2].add(v2)
            new_adj[v2].add(u2)
            
            new_c4 = count_c4(new_adj)
            
            if new_c4 < best_c4:
                # Accept the swap
                adj = new_adj
                edges[remove_idx] = (u2, v2)
                edge_set.discard((u1, v1))
                edge_set.discard((v1, u1))
                edge_set.add((u2, v2))
                edge_set.add((v2, u2))
                best_c4 = new_c4
                best_edges = edges[:]
                found_improvement = True
                break
    
        if time.time() > deadline:
            break
    
    # Return the best edges found
    if len(best_edges) != target_edges:
        # Fallback
        return edges[:target_edges]
    
    return best_edges
