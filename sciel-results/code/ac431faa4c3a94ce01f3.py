import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    n = 31
    q = 5
    
    # Generate points in PG(2, 5)
    points = []
    # Finite points (x, y, 1)
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # Points at infinity (x, 1, 0)
    for x in range(q):
        points.append((x, 1, 0))
    # Point (1, 0, 0)
    points.append((1, 0, 0))
    
    # Verify count
    assert len(points) == n, f"Expected 31 points, got {len(points)}"

    edges = []
    edge_set = set()
    
    # Build adjacency based on polarity: x1*x2 + y1*y2 + z1*z2 = 0 mod 5
    # Note: Current program uses +z1*z2. Inspiration 1 uses -z1*z2.
    # Let's try the current program's polarity first, as it's the baseline.
    # If it yields 6 C4s, we can try the other polarity.
    
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            dot = (pi[0] * pj[0] + pi[1] * pj[1] + pi[2] * pj[2]) % q
            if dot == 0:
                edges.append([i, j])
                edge_set.add((i, j))
                edge_set.add((j, i))
                
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    # Find the non-edge that minimizes the number of NEW C4s
    # New C4s = Number of paths of length 3 between u and v in the current graph
    
    best_edge = None
    min_c4s = n * n # Large number
    
    # Precompute neighbors for speed
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                # Count paths of length 3: i - a - b - j
                # Sum over a in N(i) of |N(a) \cap N(j)|
                c4_count = 0
                for a in neighbors[i]:
                    # Count common neighbors of a and j
                    # Since n is small, this is fast
                    c4_count += int(np.sum(adj[a] & adj[j]))
                
                if c4_count < min_c4s:
                    min_c4s = c4_count
                    best_edge = [i, j]
                    
                if time.time() - start_time > DEADLINE:
                    break
        if time.time() - start_time > DEADLINE:
            break
            
    if best_edge:
        edges.append(best_edge)
        
    return edges
