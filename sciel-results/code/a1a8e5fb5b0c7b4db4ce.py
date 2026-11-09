import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s wall-clock
    
    n = 40
    target_edges = 401
    
    # Partition: A size 9, B size 31
    a_size = 9
    b_size = 31
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = set()
    
    # All A-B edges: 9 * 31 = 279
    for a in A:
        for b in B:
            edges.add((a, b))
    
    needed_in_B = target_edges - (a_size * b_size)  # 401 - 279 = 122
    
    # We need a triangle-free graph on B with 122 edges and max degree 8.
    # This ensures no triangle in B, and since A has no internal edges and no triangles
    # can form (A vertices are independent), the only triangles would need 3 vertices
    # in B or 2 in B + 1 in A. But A is independent so no edge within A.
    # A triangle with 2 in B and 1 in A: need edge between the 2 B-vertices AND both
    # connected to the A-vertex. Since all A-B edges exist, any B-B edge (u,v) creates
    # a triangle with any a in A. So the book size for edge (u,v) in B is |A| = 9.
    # For A-B edges: a triangle needs a third vertex connected to both. If edge (a,b),
    # need c such that (a,c) and (b,c) exist. (a,c) exists if c in B. (b,c) exists if
    # c in B and (b,c) is a B-B edge, or c in A (but no A-A edges). So triangles containing
    # (a,b) are: for each c in B with (b,c) in B-B edges, we get triangle (a,b,c).
    # So book size for (a,b) = degree of b in B-B graph.
    # Thus max book = max(9, max_B_degree).
    # If max_B_degree <= 9, max book = 9.
    # The theorem gives lower bound 40/6 ≈ 6.67, so 7 is the integer lower bound.
    # But with this design, we get 9. Can we do better?
    
    # Alternative: try to get max book = 7 or 8.
    # For max book = 7: need max_B_degree <= 7 and |A| <= 7. But |A|*|B| + needed_in_B = 401.
    # If |A| = 7, |B| = 33, A-B = 231, needed_in_B = 170. Max degree 7 on 33 vertices:
    # max edges = 33*7/2 = 115.5 < 170. Not possible.
    # If |A| = 8, |B| = 32, A-B = 256, needed_in_B = 145. Max degree 8: 32*8/2 = 128 < 145.
    # If |A| = 9, |B| = 31, A-B = 279, needed_in_B = 122. Max degree 8: 31*8/2 = 124 >= 122.
    # So max book = 9 is achievable with this bipartite + B-internal design.
    
    # Let's try to actually achieve max book = 7 with a different approach.
    # Actually, the theoretical minimum is 7. Let's try a more sophisticated construction.
    
    # For now, let's use the greedy approach with max degree 8 to get book size 9.
    # But let's also try to see if we can get 8.
    
    # Try max degree 8 first (book size 9)
    def try_build(max_deg, max_tries=50):
        B_edges = []
        degree = {v: 0 for v in B}
        adj = {v: set() for v in B}
        pairs = list(itertools.combinations(B, 2))
        
        for attempt in range(max_tries):
            if time.time() > deadline:
                return None
            random.shuffle(pairs)
            B_edges = []
            degree = {v: 0 for v in B}
            adj = {v: set() for v in B}
            
            for u, v in pairs:
                if len(B_edges) >= needed_in_B:
                    break
                if time.time() > deadline:
                    return None
                # Check triangle
                if adj[u] & adj[v]:
                    continue
                # Check degree
                if degree[u] >= max_deg or degree[v] >= max_deg:
                    continue
                B_edges.append((u, v))
                degree[u] += 1
                degree[v] += 1
                adj[u].add(v)
                adj[v].add(u)
            
            if len(B_edges) >= needed_in_B:
                return B_edges[:needed_in_B]
        
        return None
    
    B_edges = try_build(8)
    
    if B_edges is None:
        # Try max degree 9
        B_edges = try_build(9)
    
    if B_edges is None:
        # Fallback: random fill
        all_pairs = list(itertools.combinations(range(n), 2))
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if len(edges) >= target_edges:
                break
            if (u, v) not in edges:
                edges.add((u, v))
    else:
        for u, v in B_edges:
            edges.add((u, v))
    
    # Verify
    if len(edges) != target_edges:
        all_pairs = list(itertools.combinations(range(n), 2))
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if len(edges) >= target_edges:
                break
            if (u, v) not in edges:
                edges.add((u, v))
    
    result = [[u, v] for u, v in edges]
    return result
