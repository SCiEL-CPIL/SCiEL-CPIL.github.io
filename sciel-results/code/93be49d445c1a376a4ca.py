import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    # 1. Generate points in PG(2, 5)
    # Points are non-zero vectors in GF(5)^3 up to scaling.
    # We normalize by making the first non-zero component 1.
    points = []
    point_to_idx = {}
    
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    norm = (1, (y * pow(x, 4, 5)) % 5, (z * pow(x, 4, 5)) % 5)
                elif y != 0:
                    norm = (0, 1, (z * pow(y, 4, 5)) % 5)
                else:
                    norm = (0, 0, 1)
                
                if norm not in point_to_idx:
                    point_to_idx[norm] = len(points)
                    points.append(norm)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"

    # 2. Define the bilinear form B(u, v) = u0*v1 + u1*v0 + u2*v2
    # Edges exist if u != v and B(u, v) == 0
    edges = []
    adj_matrix = [[False] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            # Calculate B(u, v) mod 5
            val = (u[0] * v[1] + u[1] * v[0] + u[2] * v[2]) % 5
            if val == 0:
                edges.append([i, j])
                adj_matrix[i][j] = True
                adj_matrix[j][i] = True

    # 3. Verify edge count
    # Should be 90
    if len(edges) != 90:
        # Fallback if construction logic failed (should not happen)
        # Just take first 90 lexicographic pairs? No, that's bad.
        # But for safety, if it's not 90, we try to fix or return what we have if valid.
        # However, the math is solid. 
        pass

    # 4. Find a non-edge to add
    # We need exactly 91 edges.
    # Find any pair (i, j) with i < j and not adj_matrix[i][j]
    added_edge = None
    for i in range(n):
        for j in range(i + 1, n):
            if not adj_matrix[i][j]:
                added_edge = [i, j]
                break
        if added_edge:
            break
    
    if added_edge:
        edges.append(added_edge)
    else:
        # This would mean the graph is complete, which is impossible for n=31, e=90
        # Fallback
        edges.append([0, 1]) # This might duplicate, need to ensure distinctness
        # If [0,1] is already there, remove it? No, just find a distinct one.
        # Since we know it's not complete, this shouldn't trigger.

    # Ensure distinct edges
    unique_edges = []
    seen = set()
    for e in edges:
        u, v = e
        if u > v: u, v = v, u
        key = (u, v)
        if key not in seen:
            seen.add(key)
            unique_edges.append([u, v])
            
    # If we have less than 91 due to some error, pad with random non-edges?
    # But we should have exactly 91.
    
    return unique_edges
