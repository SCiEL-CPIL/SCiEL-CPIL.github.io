"""Construct small girth-regular graphs; for (5,9) return LUW graph with optimized construction."""

import time

def construct(k, g):
    deadline = time.time() + 110
    if (k, g) == (5, 9):
        return _small_5_9(deadline)
    return _luw(k, g, deadline)

def _small_5_9(deadline):
    # LUW with q=5, d=5: 5-regular bipartite graph, girth >= 10 >= 9
    # n = 2 * 5^5 = 6250 vertices
    return _luw(5, 9, deadline)

def _luw(k, g, deadline=None):
    if deadline is None:
        deadline = time.time() + 110
    
    q = k
    # Choose d such that girth >= d + 5 >= g, d odd
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute all vertices and their neighbors
    # Vertices are indexed by (side, x1, x2, ..., xd) where x1 is the "free" variable
    # For side 0: x = (x1, 0, 0, ..., 0) with x1 in 0..q-1
    # Actually, the LUW construction has vertices (side, x) where x is in GF(q)^d
    # but the first coordinate is free. Let me re-read.
    
    # The construction: points are (side, x) where side in {0,1}, x in GF(q)^d
    # Two vertices (0, p) and (1, l) are adjacent iff for all c=2..d:
    #   l[c] = p[c] + l[rule[c][0]] * p[rule[c][1]]  (for side 0 to side 1)
    #   p[c] = l[c] - l[rule[c][0]] * p[rule[c][1]]  (for side 1 to side 0)
    # Wait, that's not right. Let me re-read the original.
    
    # From the original code:
    # For vertex (side, x), neighbors are (1-side, y) where:
    #   y[1] = first (free variable, 0..q-1)
    #   For c = 2..d:
    #     If side == 0: y[c] = x[c] + y[a] * x[b]  where (a,b) = rule[c]
    #     If side == 1: y[c] = x[c] - x[a] * y[b]  where (a,b) = rule[c]
    
    # So for a given vertex (side, x), the neighbors are determined by y[1] (the free variable)
    # and then y[2..d] are computed sequentially.
    
    # Total vertices: 2 * q^d (but actually the connected component may be smaller)
    # For q=5, d=5: 2 * 3125 = 6250
    
    # Precompute neighbors for all vertices
    # Vertex encoding: side * q^d + x1 * q^(d-1) + x2 * q^(d-2) + ... + xd
    # Actually, let's use a simpler encoding: index = side * q^d + x_encoded
    # where x_encoded = x1 * q^(d-1) + x2 * q^(d-2) + ... + xd
    
    qd = q ** d
    total_vertices = 2 * qd
    
    # Precompute neighbor list for each vertex
    # For each vertex, we need to compute its q neighbors
    # This is O(2 * q^d * q * d) = O(2 * q^(d+1) * d)
    # For q=5, d=5: 2 * 15625 * 5 = 156250 operations, very fast
    
    neighbors = [[] for _ in range(total_vertices)]
    
    # Precompute powers of q
    q_powers = [1] * (d + 1)
    for i in range(1, d + 1):
        q_powers[i] = q_powers[i-1] * q
    
    def encode_x(x):
        """Encode x = (x1, x2, ..., xd) to an integer"""
        enc = 0
        for i in range(d):
            enc += x[i+1] * q_powers[d-1-i]
        return enc
    
    def decode_x(enc):
        """Decode integer to x = (x1, x2, ..., xd)"""
        x = [0] * (d + 1)
        for i in range(d):
            x[i+1] = enc % q
            enc //= q
        return x
    
    # For each vertex, compute neighbors
    for side in range(2):
        for x_enc in range(qd):
            vertex_idx = side * qd + x_enc
            x = decode_x(x_enc)
            
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                
                y_enc = encode_x(y)
                neighbor_idx = (1 - side) * qd + y_enc
                neighbors[vertex_idx].append(neighbor_idx)
            
            # Check deadline
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Build edge list (only add each edge once, from side 0)
    edges = []
    for side in range(2):
        for x_enc in range(qd):
            vertex_idx = side * qd + x_enc
            if side == 0:
                for nb in neighbors[vertex_idx]:
                    edges.append((vertex_idx, nb))
    
    return total_vertices, edges
