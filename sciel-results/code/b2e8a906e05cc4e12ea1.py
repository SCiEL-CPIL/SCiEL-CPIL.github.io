import numpy as np

def construct():
    # Step 1: Generate the 31 points of PG(2, 5)
    # Points are represented as (x, y, z) in F_5, normalized such that the first non-zero component is 1.
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    p = (1, (y * pow(x, 4, 5)) % 5, (z * pow(x, 4, 5)) % 5)
                elif y != 0:
                    p = (0, 1, (z * pow(y, 4, 5)) % 5)
                else:
                    p = (0, 0, 1)
                
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    # We should have 31 points
    # Verify count
    if len(points) != 31:
        # Fallback if generation logic has issues, though it should be correct
        # Standard PG(2,5) has (5^3-1)/(5-1) = 31 points
        pass

    n = 31
    # Map points to indices 0..30
    # The order doesn't matter as long as it's consistent
    
    # Step 2: Construct Adjacency Matrix for Polarity Graph
    # Polarity defined by M = diag(1, 1, -1) = diag(1, 1, 4) in F_5
    # P ~ Q iff P^T M Q == 0 and P != Q
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            # Compute dot product with M
            # x1*x2 + y1*y2 + 4*z1*z2
            val = (p1[0]*p2[0] + p1[1]*p2[1] + 4*p1[2]*p2[2]) % 5
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
    
    # Step 3: Compute A^3
    # A^3[i, j] is the number of paths of length 3 from i to j.
    # Adding edge (i, j) creates (A^3[i, j]) new 4-cycles.
    A2 = A @ A
    A3 = A2 @ A
    
    # Step 4: Find non-edge with minimum A^3 value
    min_val = float('inf')
    best_edge = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                val = A3[i, j]
                if val < min_val:
                    min_val = val
                    best_edge = (i, j)
    
    # Step 5: Construct the list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
    
    # Add the new edge
    edges.append([best_edge[0], best_edge[1]])
    
    # Ensure we have exactly 91 edges
    # The polarity graph has 90 edges. We added 1.
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    return edges
