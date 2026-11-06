import time
import random
import numpy as np

def construct():
    """
    Construct a graph on 31 vertices with exactly 91 edges that minimizes the number of 4-cycles.
    
    Strategy:
    1. Start with the polarity graph of PG(2,5), which is a C4-free graph with 90 edges.
    2. Add one edge to reach 91 edges, choosing the edge that minimizes new 4-cycles.
    3. Use local search (edge swaps) to further minimize the number of 4-cycles.
    """
    deadline = time.time() + 110  # Keep well under 120s limit
    
    # Step 1: Build the polarity graph of PG(2,5)
    # PG(2,5) has 31 points. The polarity graph connects points that are orthogonal under a polarity.
    # For q=5, we can use the standard construction.
    
    # Generate all 31 points of PG(2,5) as normalized vectors
    points = []
    for x in range(6):
        for y in range(6):
            for z in range(6):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                vec = [x, y, z]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 4, 5)  # modular inverse in GF(5)
                        vec = [(v * inv) % 5 for v in vec]
                        break
                points.append(tuple(vec))
    
    # Remove duplicates (shouldn't be any since we normalized)
    points = list(set(points))
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # The polarity: x·y = x0*y0 + x1*y1 + x2*y2 = 0 mod 5
    # But for a proper polarity in PG(2,5), we use a symmetric bilinear form.
    # Standard polarity for q=5: <(x0,x1,x2), (y0,y1,y2)> = x0*y2 + x1*y1 + x2*y0
    # This is symmetric.
    
    def polarity_inner(v, w):
        return (v[0]*w[2] + v[1]*w[1] + v[2]*w[0]) % 5
    
    # Build edges: connect i,j if polarity_inner(points[i], points[j]) == 0 and i != j
    edges = set()
    for i in range(31):
        for j in range(i+1, 31):
            if polarity_inner(points[i], points[j]) == 0:
                edges.add((i, j))
    
    # The polarity graph should have 90 edges (C4-free)
    # Let's verify
    base_edges = list(edges)
    n_base = len(base_edges)
    
    # Step 2: Add one edge to reach 91
    # Find the non-edge that creates the fewest new 4-cycles
    all_pairs = set((i, j) for i in range(31) for j in range(i+1, 31))
    non_edges = all_pairs - edges
    
    def count_c4s(edge_set):
        """Count the number of 4-cycles in the graph defined by edge_set."""
        n = 31
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        c4_count = 0
        # Count pairs of common neighbors for each pair of vertices
        # A 4-cycle a-b-c-d-a corresponds to two vertices a,c sharing two common neighbors b,d
        # Number of 4-cycles = sum over all pairs {a,c} of C(common_neighbors(a,c), 2) / 2
        # Wait, that's not quite right. Let me think again.
        # A 4-cycle is a sequence a-b-c-d-a where all edges exist and a,b,c,d distinct.
        # Each 4-cycle is counted 8 times if we count (a,c) pairs and their common neighbors.
        # Actually, for each pair of vertices (a,c), let k = |N(a) ∩ N(c)|.
        # The number of 4-cycles where a and c are opposite vertices is C(k, 2).
        # Each 4-cycle has two pairs of opposite vertices, so total 4-cycles = (1/2) * sum_{a<c} C(|N(a)∩N(c)|, 2)
        
        for a in range(n):
            for c in range(a+1, n):
                common = adj[a] & adj[c]
                k = len(common)
                if k >= 2:
                    c4_count += k * (k - 1) // 2
        return c4_count // 2
    
    # Find best edge to add
    best_edge = None
    best_c4 = float('inf')
    for e in non_edges:
        new_edges = edges | {e}
        c4 = count_c4s(new_edges)
        if c4 < best_c4:
            best_c4 = c4
            best_edge = e
    
    edges.add(best_edge)
    current_c4 = best_c4
    
    # Step 3: Local search - try edge swaps to reduce C4 count
    # Represent edges as a set of frozensets or sorted tuples
    edge_list = list(edges)
    
    def fast_c4_delta_add_remove(edge_set, add_edge, remove_edge):
        """Compute the change in C4 count when adding add_edge and removing remove_edge."""
        n = 31
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Remove remove_edge
        u1, v1 = remove_edge
        adj[u1].discard(v1)
        adj[v1].discard(u1)
        
        # Add add_edge
        u2, v2 = add_edge
        adj[u2].add(v2)
        adj[v2].add(u2)
        
        # Count C4s
        c4_count = 0
        for a in range(n):
            for c in range(a+1, n):
                common = adj[a] & adj[c]
                k = len(common)
                if k >= 2:
                    c4_count += k * (k - 1) // 2
        return c4_count // 2
    
    # Local search
    best_edges = set(edges)
    best_c4_val = current_c4
    
    while time.time() < deadline:
        improved = False
        random.shuffle(edge_list)
        
        for e1 in edge_list:
            if time.time() > deadline:
                break
            if e1 not in best_edges:
                continue
            # Try swapping e1 with a non-edge
            non_edges_local = all_pairs - best_edges
            random_sample = list(non_edges_local)
            random.shuffle(random_sample)
            
            for e2 in random_sample[:50]:  # Try up to 50 random non-edges
                if time.time() > deadline:
                    break
                # Compute new C4 count
                new_edges = set(best_edges)
                new_edges.discard(e1)
                new_edges.add(e2)
                
                new_c4 = count_c4s(new_edges)
                
                if new_c4 < best_c4_val:
                    best_edges = new_edges
                    best_c4_val = new_c4
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try more aggressive: remove one edge, add two, remove one more
            # Or just accept current best
            break
    
    return [list(e) for e in sorted(best_edges)]
