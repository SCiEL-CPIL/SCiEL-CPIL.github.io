import random
import time
import math

def construct():
    start_time = time.time()
    deadline = 120.0
    n = 31
    
    # 1. Construct the polarity graph of PG(2, 5)
    # Points are equivalence classes of non-zero vectors in GF(5)^3
    points = []
    point_to_index = {}
    
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make the first non-zero coordinate 1
                vec = [x, y, z]
                # Find first non-zero
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5) # Fermat's little theorem: a^(p-2) = a^-1 mod p
                        vec = [(v * inv) % 5 for v in vec]
                        break
                key = tuple(vec)
                if key not in point_to_index:
                    point_to_index[key] = len(points)
                    points.append(key)
    
    # Verify we have 31 points
    if len(points) != 31:
        # Fallback if construction fails (should not happen)
        return [[i, j] for i in range(n) for j in range(i+1, n)][:91]

    # 2. Build adjacency
    # Polarity: P is on polar line of Q iff x_P*x_Q + y_P*z_Q + z_P*y_Q == 0 mod 5
    # Note: The form x*x' + y*y' + z*z' is symmetric.
    # For a polarity in PG(2,q), we usually use a symmetric or alternating form.
    # x*x' + y*y' + z*z' = 0 works for q=5?
    # Let's check absolute points: x^2 + y^2 + z^2 = 0.
    # (1,2,0): 1+4+0=5=0. Yes.
    
    neighbors = [set() for _ in range(n)]
    edges = set()
    
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i == j:
                continue
            # Check if p is on the polar line of q
            # p = (px, py, pz), q = (qx, qy, qz)
            # Condition: px*qx + py*qz + pz*qy == 0
            val = (p[0]*q[0] + p[1]*q[2] + p[2]*q[1]) % 5
            if val == 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
                edges.add((i, j))
    
    # Check edge count
    # Expected edges: 90
    # If not 90, something is wrong with the polarity definition or counting.
    # In PG(2,5), each point is on 6 lines.
    # The polar line of a point has 6 points.
    # If the point is absolute, it is one of them.
    # Degree of absolute point: 5 (excluding self).
    # Degree of non-absolute point: 6.
    # Number of absolute points in this polarity?
    # x^2+y^2+z^2=0 has 26 solutions? No.
    # Number of points on conic x^2+y^2+z^2=0 in PG(2,5).
    # A conic in PG(2,q) has q+1 points. So 6 absolute points.
    # 6 points with degree 5, 25 points with degree 6.
    # Sum of degrees = 6*5 + 25*6 = 30 + 150 = 180.
    # Edges = 180 / 2 = 90.
    # Correct.
    
    if len(edges) != 90:
        # Fallback to random if construction is flawed
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        return [list(e) for e in all_edges[:91]]

    # 3. Find the best edge to add
    # We need to add 1 edge to make 91.
    # Minimize C4s created.
    # C4s created = number of paths of length 3 between u and v in G.
    
    min_c4_count = float('inf')
    best_edge = None
    
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in edges:
                continue
            
            # Count paths u-a-b-v
            # a in N(u), b in N(v), a != b, (a,b) in E
            count = 0
            for a in neighbors[u]:
                for b in neighbors[v]:
                    if a == b:
                        continue
                    if (a, b) in edges or (b, a) in edges:
                        count += 1
            
            if count < min_c4_count:
                min_c4_count = count
                best_edge = (u, v)
                
                if min_c4_count == 1:
                    # Can't do better than 1
                    # We can break early if we assume symmetry, but let's just take the first one found with 1
                    # Actually, breaking immediately might miss if 0 is possible (it's not), so 1 is optimal.
                    break
        if min_c4_count == 1:
            break
            
    if best_edge is None:
        # Should not happen
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        # Find a non-edge
        for e in all_edges:
            if e not in edges:
                best_edge = e
                break
                
    # Add the best edge
    u, v = best_edge
    edges.add((u, v))
    
    return [[e[0], e[1]] for e in edges]
