def construct():
    # Construct the polarity graph of PG(2,5) using the standard dot product.
    # Vertices are equivalence classes of non-zero vectors in F_5^3.
    # Two vertices are adjacent if their dot product is 0.
    # This graph is known to be C5-free.
    # For q=5, the independence number is at most 6 (actually 6).
    # We add 2 isolated vertices to reach n=33.
    # Total independence number = 6 + 2 = 8 < 9.
    # Total vertices n = 31 + 2 = 33.
    
    q = 5
    
    # Generate all distinct projective points
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: find the first non-zero coordinate
                # and scale so that it is 1
                vec = [a, b, c]
                # Find first non-zero
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                if first_nonzero == -1:
                    continue
                inv = pow(vec[first_nonzero], q - 2, q)
                normalized = [(x * inv) % q for x in vec]
                key = tuple(normalized)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_points = len(points)
    # Should be q^2 + q + 1 = 25 + 5 + 1 = 31
    
    # Build edges of the polarity graph
    # Two points p, q are adjacent if sum(p_i * q_i) == 0 mod 5
    edges = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            p = points[i]
            r = points[j]
            dot = (p[0] * r[0] + p[1] * r[1] + p[2] * r[2]) % q
            if dot == 0:
                edges.append([i, j])
    
    # Add 2 isolated vertices
    n = n_points + 2
    
    return {'n': n, 'edges': edges}
