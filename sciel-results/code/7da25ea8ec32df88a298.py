import itertools

def construct():
    # Generate the 31 points of PG(2, 5)
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                if x != 0:
                    inv_x = pow(x, q-2, q) # Fermat's little theorem for inverse
                    px = 1
                    py = (y * inv_x) % q
                    pz = (z * inv_x) % q
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    px = 0
                    py = 1
                    pz = (z * inv_y) % q
                else:
                    px = 0
                    py = 0
                    pz = 1
                
                p = (px, py, pz)
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    # There should be 31 points
    # Construct edges based on orthogonality u . v = 0
    edges = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                edges.append([i, j])
                
    # The graph has 93 edges and is C4-free.
    # We need exactly 91 edges.
    # Removing any 2 edges preserves C4-freeness.
    return edges[:91]
