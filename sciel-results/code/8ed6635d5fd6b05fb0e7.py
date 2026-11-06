import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Field F9 = F3[x] / (x^2 + 1)
    # Elements are pairs (a, b) in F3^2
    # 0: (0,0), 1: (1,0), 2: (2,0), 3: (0,1), 4: (1,1), 5: (2,1), 6: (0,2), 7: (1,2), 8: (2,2)
    
    def f9_add(a, b):
        # a, b are integers 0..8 representing elements
        # Map to (r, s)
        def to_pair(x):
            return (x % 3, x // 3)
        
        def from_pair(r, s):
            return r + 3 * s
        
        a1, a2 = to_pair(a)
        b1, b2 = to_pair(b)
        return from_pair((a1 + b1) % 3, (a2 + b2) % 3)

    def f9_mul(a, b):
        def to_pair(x):
            return (x % 3, x // 3)
        
        def from_pair(r, s):
            return r + 3 * s
        
        a1, a2 = to_pair(a)
        b1, b2 = to_pair(b)
        # (a1 + a2*x) * (b1 + b2*x) = a1b1 + (a1b2 + a2b1)x + a2b2x^2
        # x^2 = -1 = 2
        r = (a1 * b1 + 2 * a2 * b2) % 3
        s = (a1 * b2 + a2 * b1) % 3
        return from_pair(r, s)

    # Generate points in PG(2, 9)
    # A point is a vector (x, y, z) in F9^3, not all zero, modulo scaling.
    # Canonical form: first non-zero coordinate is 1.
    points = []
    seen = set()
    
    # Iterate over all non-zero vectors
    # There are 9^3 - 1 = 728 non-zero vectors.
    # Each point has 8 non-zero scalars. 728 / 8 = 91 points.
    
    for x in range(9):
        for y in range(9):
            for z in range(9):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Find canonical representative
                # If x != 0, scale by 1/x
                # If x == 0 and y != 0, scale by 1/y
                # If x == 0 and y == 0, z != 0, scale by 1/z
                
                if x != 0:
                    # x_inv
                    # We need inverse in F9
                    # For a in F9, a * a_inv = 1
                    # We can precompute inverses or just find it
                    inv_x = None
                    for k in range(9):
                        if k == 0: continue
                        if f9_mul(x, k) == 1:
                            inv_x = k
                            break
                    cx = f9_mul(x, inv_x) # should be 1
                    cy = f9_mul(y, inv_x)
                    cz = f9_mul(z, inv_x)
                    rep = (1, cy, cz)
                elif y != 0:
                    inv_y = None
                    for k in range(9):
                        if k == 0: continue
                        if f9_mul(y, k) == 1:
                            inv_y = k
                            break
                    cx = f9_mul(x, inv_y) # 0
                    cy = f9_mul(y, inv_y) # 1
                    cz = f9_mul(z, inv_y)
                    rep = (0, 1, cz)
                else: # z != 0
                    inv_z = None
                    for k in range(9):
                        if k == 0: continue
                        if f9_mul(z, k) == 1:
                            inv_z = k
                            break
                    cx = f9_mul(x, inv_z) # 0
                    cy = f9_mul(y, inv_z) # 0
                    cz = f9_mul(z, inv_z) # 1
                    rep = (0, 0, 1)
                
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)
    
    n = len(points)
    if n != 91:
        # Fallback to seed if generation fails
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # Construct edges
    # P ~ Q if dot(P, Q) == 0 and P != Q
    edges = []
    for i in range(n):
        p = points[i]
        for j in range(i + 1, n):
            q = points[j]
            dot = f9_add(f9_add(f9_mul(p[0], q[0]), f9_mul(p[1], q[1])), f9_mul(p[2], q[2]))
            if dot == 0:
                edges.append([i, j])
                
    if time.time() > deadline:
         # Return best found so far? 
         # The graph is fully constructed.
         pass

    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
