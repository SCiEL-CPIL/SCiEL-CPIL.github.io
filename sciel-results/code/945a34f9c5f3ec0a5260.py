def construct():
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize to ensure unique representation
                if x != 0:
                    inv = pow(x, q-2, q)
                    p = (1, (y * inv) % q, (z * inv) % q)
                elif y != 0:
                    inv = pow(y, q-2, q)
                    p = (0, 1, (z * inv) % q)
                else:
                    inv = pow(z, q-2, q)
                    p = (0, 0, 1)
                
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    n = len(points)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            if (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
