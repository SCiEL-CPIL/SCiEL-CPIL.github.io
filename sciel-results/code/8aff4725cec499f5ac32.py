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

    # Precompute all non-backtracking closed walks of lengths 5 to g-1
    # with direction tracking
    def get_all_walks(max_len):
        """Returns dict mapping length -> list of walks, where each walk is list of (e_idx, sign)"""
        walks_by_len = {L: [] for L in range(5, max_len)}
        
        for start in range(10):
            # State: (current_node, prev_node, length, edge_seq)
            stack = [(start, -1, 0, [])]
            while stack:
                curr, prev, l, seq = stack.pop()
                if l == max_len - 1:
                    # Check if we can close back to start
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if nxt == start:
                            # This completes a closed walk of length max_len
                            if (curr, start) in edge_index:
                                e_idx = edge_index[(curr, start)]
                                if (curr, start) == edges_p[e_idx]:
                                    sign = 1
                                else:
                                    sign = -1
                                full_seq = seq + [(e_idx, sign)]
                                walks_by_len[max_len].append(full_seq)
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    if (curr, nxt) in edge_index:
                        e_idx = edge_index[(curr, nxt)]
                        if (curr, nxt) == edges_p[e_idx]:
                            sign = 1
                        else:
                            sign = -1
                        stack.append((nxt, curr, l + 1, seq + [(e_idx, sign)]))
        
        return walks_by_len

    # Get all walks up to length g-1
    all_walks = get_all_walks(g)
    
    # Find a primitive root modulo p
    def find_primitive_root(p):
        """Find a primitive root modulo prime p"""
        if p == 2:
            return 1
        # Factor p-1
        phi = p - 1
        factors = set()
        temp = phi
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                factors.add(d)
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)
        
        # Try candidates
        for g_candidate in range(2, p):
            is_primitive = True
            for f in factors:
                if pow(g_candidate, phi // f, p) == 1:
                    is_primitive = False
                    break
            if is_primitive:
                return g_candidate
        return None

    # Try deterministic voltage assignments using primitive roots
    def try_deterministic_voltages(p, primitive_root):
        """Try voltage assignments based on primitive root powers"""
        # Strategy 1: v_i = primitive_root^i mod p
        voltages = [pow(primitive_root, i, p) for i in range(num_edges)]
        
        # Check all walks
        for L in range(5, g):
            for walk in all_walks[L]:
                s = 0
                for e_idx, sign in walk:
                    s += sign * voltages[e_idx]
                if s % p == 0:
                    return None
        
        return voltages
    
    # Strategy 2: Use different assignments
    def try_voltages_powers(p, primitive_root, exponent_shift=0):
        """v_i = primitive_root^(i + shift) mod p"""
        voltages = [pow(primitive_root, (i + exponent_shift) % (p-1), p) for i in range(num_edges)]
        
        for L in range(5, g):
            for walk in all_walks[L]:
                s = 0
                for e_idx, sign in walk:
                    s += sign * voltages[e_idx]
                if s % p == 0:
                    return None
        return voltages
    
    # Strategy 3: Use quadratic residues
    def try_voltages_quadratic(p, primitive_root):
        """v_i = primitive_root^(2*i) mod p for even i, primitive_root^(2*i+1) for odd i"""
        voltages = []
        for i in range(num_edges):
            if i % 2 == 0:
                voltages.append(pow(primitive_root, 2*i, p))
            else:
                voltages.append(pow(primitive_root, 2*i+1, p))
        
        for L in range(5, g):
            for walk in all_walks[L]:
                s = 0
                for e_idx, sign in walk:
                    s += sign * voltages[e_idx]
                if s % p == 0:
                    return None
        return voltages

    primes_to_try = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    for p in primes_to_try:
        if 10 * p < 512:
            continue
            
        primitive_root = find_primitive_root(p)
        if primitive_root is None:
            continue
        
        # Try deterministic strategies first
        found_voltages = None
        
        # Strategy 1: Basic powers
        found_voltages = try_deterministic_voltages(p, primitive_root)
        if found_voltages:
            voltages = found_voltages
        else:
            # Try with shifts
            for shift in range(20):
                found_voltages = try_voltages_powers(p, primitive_root, shift)
                if found_voltages:
                    voltages = found_voltages
                    break
            if not found_voltages:
                # Try quadratic
                found_voltages = try_voltages_quadratic(p, primitive_root)
                if found_voltages:
                    voltages = found_voltages
                else:
                    # Fall back to random with more attempts
                    rng = random.Random(42 + p)
                    found = False
                    for attempt in range(5000):
                        voltages = [rng.randint(1, p-1) for _ in range(num_edges)]
                        valid = True
                        for L in range(5, g):
                            for walk in all_walks[L]:
                                s = 0
                                for e_idx, sign in walk:
                                    s += sign * voltages[e_idx]
                                if s % p == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            found = True
                            break
                    if not found:
                        continue
        
        # Construct the graph
        n = 10 * p
        edges_out = set()
        
        for i in range(num_edges):
            u, v = edges_p[i]
            ve = voltages[i]
            for fiber in range(p):
                # (u, fiber) -> (v, fiber + ve)
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
