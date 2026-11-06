import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate points of PG(2,5)
    # Points are 1D subspaces of GF(5)^3
    # Canonical representative: first non-zero component is 1
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    pts = (1, (y * pow(x, 4, 5)) % 5, (z * pow(x, 4, 5)) % 5)
                elif y != 0:
                    pts = (0, 1, (z * pow(y, 4, 5)) % 5)
                else:
                    pts = (0, 0, 1)
                if pts not in points:
                    points.append(pts)
    
    n = len(points)
    # n should be 31
    
    # 2. Build adjacency list based on polarity B(u,v) = u1*v1 + u2*v2 + u3*v3 = 0
    # Note: This is a symmetric bilinear form.
    # u is adjacent to v if u != v and B(u,v) == 0
    
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            # B(u,v) = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
            b_val = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if b_val == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    # Verify edge count
    # Should be 90
    if len(edges) != 90:
        # Fallback if construction is wrong (should not happen)
        # Just return first 91 lexicographic pairs as in seed
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]

    # 3. Find the non-edge that creates the fewest 4-cycles
    # A new 4-cycle using edge (u,v) corresponds to a path of length 3 u-x-y-v in the base graph.
    # Count = number of x in N(u) such that x and v have a common neighbor.
    
    min_count = float('inf')
    best_edge = None
    
    # Precompute common neighbors for speed?
    # Since graph is C4-free, common neighbors is 0 or 1.
    # We can just check intersection of neighbor sets.
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            if j in adj[i]:
                continue
            
            # Calculate number of length-3 paths from i to j
            count = 0
            for x in adj[i]:
                # Check if x and j have a common neighbor
                # common neighbor y must be in adj[x] and adj[j]
                # Since C4-free, if they share one, it's unique.
                # Intersection of sets
                if adj[x] & adj[j]:
                    count += 1
            
            if count < min_count:
                min_count = count
                best_edge = [i, j]
    
    if best_edge is None:
        # Fallback
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]

    # 4. Add the best edge
    edges.append(best_edge)
    
    return edges
