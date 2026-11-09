"""Program for fewest four-cycles in graphs with 31 vertices and 91 edges."""
import numpy as np
from itertools import combinations
import time

def construct():
    """
    Construct a graph with 31 vertices and exactly 91 edges minimizing the number of C4 subgraphs.
    
    Approach:
    1. Build a C4-free base graph with 90 edges using a polarity graph construction on PG(2,5).
    2. Add one edge carefully to minimize the number of new C4s created.
    3. Perform local search (edge swaps) to further minimize C4 count.
    """
    n = 31
    target_edges = 91
    
    # Step 1: Build a C4-free base graph with 90 edges
    # Use a polarity graph construction for q=5
    # The vertices are the 31 points of PG(2,5)
    # We represent points as homogeneous coordinates [x,y,z] over GF(5)
    
    # Generate all points in PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                pt = [x, y, z]
                for i in range(3):
                    if pt[i] != 0:
                        inv = pow(pt[i], 3, 5)  # inverse in GF(5)
                        pt = [(pt[j] * inv) % 5 for j in range(3)]
                        break
                if tuple(pt) not in seen:
                    seen.add(tuple(pt))
                    points.append(pt)
    
    # Should have 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define a polarity: A point p = (x,y,z) is incident to point q = (u,v,w) 
    # if x*w + y*u + z*v = 0 in GF(5) (symmetric bilinear form)
    # This defines a symmetric relation, so we need to handle self-incidence carefully
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            # Check polarity condition: p[0]*q[2] + p[1]*q[0] + p[2]*q[1] == 0 mod 5
            val = (p[0] * q[2] + p[1] * q[0] + p[2] * q[1]) % 5
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Count edges in base graph
    edge_count = sum(adj[i, j] for i in range(n) for j in range(i + 1, n))
    
    # If we don't have exactly 90 edges, adjust
    if edge_count < 90:
        # Add random non-edges
        import random
        random.seed(42)
        added = 0
        attempts = 0
        while added < 90 - edge_count and attempts < 10000:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j and adj[i, j] == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                added += 1
            attempts += 1
    elif edge_count > 90:
        # Remove random edges
        import random
        random.seed(42)
        removed = 0
        attempts = 0
        while removed < edge_count - 90 and attempts < 10000:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j and adj[i, j] == 1:
                adj[i, j] = 0
                adj[j, i] = 0
                removed += 1
            attempts += 1
    
    # Now we have 90 edges. Add one more edge to get 91.
    # To minimize C4s, add an edge between two vertices with minimum common neighbors
    min_common = float('inf')
    best_pair = None
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                common = np.sum(adj[i] & adj[j])
                if common < min_common:
                    min_common = common
                    best_pair = (i, j)
    
    if best_pair is not None:
        i, j = best_pair
        adj[i, j] = 1
        adj[j, i] = 1
    
    # Now we have 91 edges. Perform local search to minimize C4 count.
    # C4 count = sum over all pairs of vertices (u,v) of C(common_neighbors(u,v), 2)
    
    def count_c4(adj):
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = np.sum(adj[i] & adj[j])
                c4_count += common * (common - 1) // 2
        return c4_count
    
    current_c4 = count_c4(adj)
    
    # Local search: try swapping edges to reduce C4 count
    # A swap: remove edge (a,b) and add edge (c,d) where (a,b) != (c,d)
    # This maintains 91 edges
    
    import random
    random.seed(42)
    no_improvement = 0
    max_iterations = 50000
    iteration = 0
    
    while no_improvement < 5000 and iteration < max_iterations:
        iteration += 1
        
        # Pick a random edge to remove
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 1]
        if len(edges) != 91:
            break
        a, b = random.choice(edges)
        
        # Pick a random non-edge to add
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]
        c, d = random.choice(non_edges)
        
        # Ensure we're not just swapping the same edge
        if (a == c and b == d) or (a == d and b == c):
            continue
        
        # Try the swap
        adj[a, b] = 0
        adj[b, a] = 0
        adj[c, d] = 1
        adj[d, c] = 1
        
        new_c4 = count_c4(adj)
        
        if new_c4 < current_c4:
            current_c4 = new_c4
            no_improvement = 0
        else:
            # Revert
            adj[a, b] = 1
            adj[b, a] = 1
            adj[c, d] = 0
            adj[d, c] = 0
            no_improvement += 1
    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                result.append([i, j])
    
    assert len(result) == 91, f"Expected 91 edges, got {len(result)}"
    
    return result
