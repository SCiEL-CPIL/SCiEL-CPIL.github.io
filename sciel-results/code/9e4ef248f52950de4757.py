import random
import numpy as np

def construct(k, g):
    # Petersen graph has 10 vertices, 15 edges.
    # We lift it by a group Z_p.
    # n = 10 * p.
    # We need n >= 512, so p >= 52.
    # Try primes starting from 53.
    
    # Petersen graph edges (0-9)
    # Standard labeling: 0-4 outer, 5-9 inner.
    # Outer: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges_p = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    num_edges = 15
    # Map edge tuple to index
    edge_index = {}
    for i, e in enumerate(edges_p):
        edge_index[e] = i
        edge_index[(e[1], e[0])] = i
        
    # Build adjacency list for Petersen
    adj = [[] for _ in range(10)]
    for u, v in edges_p:
        adj[u].append(v)
        adj[v].append(u)

    target_g = g
    # Lower bound on n is 512.
    # n = 10 * p.
    # So p must be >= 52.
    
    primes_to_try = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Generate constraints: all non-backtracking closed walks of length 5 to g-1
    # with direction tracking
    constraints = []
    for L in range(5, target_g):
        for start in range(10):
            # State: (current_node, prev_node, length, edge_seq)
            # edge_seq is list of (e_idx, sign)
            stack = [(start, -1, 0, [])]
            while stack:
                curr, prev, l, seq = stack.pop()
                if l == L:
                    if curr == start:
                        constraints.append(seq)
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    e_idx = edge_index[(curr, nxt)]
                    if (curr, nxt) == edges_p[e_idx]:
                        sign = 1
                    else:
                        sign = -1
                    stack.append((nxt, curr, l + 1, seq + [(e_idx, sign)]))
    
    # Convert constraints to coefficient vectors for faster checking
    # Each constraint is a vector of length 15 with coefficients in {-1, 0, 1}
    constraint_vecs = []
    for w in constraints:
        vec = [0] * num_edges
        for e_idx, sign in w:
            vec[e_idx] += sign
        constraint_vecs.append(vec)
    
    # Try to find voltage assignments
    rng = random.Random(42)
    
    for p in primes_to_try:
        found = False
        
        # First try deterministic assignments based on powers of primitive root
        # Find primitive root mod p
        def is_primitive_root(g, p):
            # Check if g is a primitive root mod p
            # g must have order p-1
            # p-1 factors
            temp = p - 1
            factors = set()
            d = 2
            while d * d <= temp:
                if temp % d == 0:
                    factors.add(d)
                    while temp % d == 0:
                        temp //= d
                d += 1
            if temp > 1:
                factors.add(temp)
            
            for f in factors:
                if pow(g, (p - 1) // f, p) == 1:
                    return False
            return True
        
        # Try structured assignments first
        # v_i = alpha^i * c for various c, or v_i = i * alpha
        for base in range(1, min(p, 20)):
            for c in range(1, min(p, 10)):
                voltages = [(base ** i * c) % p for i in range(num_edges)]
                valid = True
                for vec in constraint_vecs:
                    s = 0
                    for i in range(num_edges):
                        s += vec[i] * voltages[i]
                    if s % p == 0:
                        valid = False
                        break
                if valid:
                    found = True
                    break
            if found:
                break
        
        # If structured didn't work, try random
        if not found:
            for attempt in range(2000):
                voltages = [rng.randint(0, p-1) for _ in range(num_edges)]
                valid = True
                for vec in constraint_vecs:
                    s = 0
                    for i in range(num_edges):
                        s += vec[i] * voltages[i]
                    if s % p == 0:
                        valid = False
                        break
                if valid:
                    found = True
                    break
        
        if found:
            # Construct the graph
            n = 10 * p
            edges_out = set()
            for i in range(num_edges):
                u, v = edges_p[i]
                ve = voltages[i]
                for fiber in range(p):
                    # u, fiber -> v, fiber + ve
                    v1 = u * p + fiber
                    v2 = v * p + (fiber + ve) % p
                    if v1 < v2:
                        edges_out.add((v1, v2))
                    else:
                        edges_out.add((v2, v1))
            
            return n, list(edges_out)
            
    # Fallback: Return the LUW graph if no small voltage graph found
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
