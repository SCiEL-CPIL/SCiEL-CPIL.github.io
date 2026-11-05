import numpy as np
from collections import defaultdict

def construct(k, g):
    # Petersen graph structure
    edges_p = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    num_edges = 15
    num_vertices = 10
    
    # Build adjacency
    adj = [[] for _ in range(num_vertices)]
    for u, v in edges_p:
        adj[u].append(v)
        adj[v].append(u)
    
    # Edge index for direction
    edge_dir = {}
    for i, (u, v) in enumerate(edges_p):
        edge_dir[(u, v)] = (i, 1)
        edge_dir[(v, u)] = (i, -1)
    
    # Generate all non-backtracking closed walks of length L
    def get_closed_walks(max_len):
        walks_by_len = {L: [] for L in range(2, max_len + 1)}
        for start in range(num_vertices):
            # Stack: (current, prev, length, edge_sequence)
            stack = [(start, -1, 0, [])]
            while stack:
                curr, prev, l, seq = stack.pop()
                if l >= max_len:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    new_seq = seq + [edge_dir[(curr, nxt)]]
                    if l + 1 <= max_len:
                        if nxt == start and l + 1 >= 2:
                            # Check it's a valid cycle (not just going back)
                            # The walk must not backtrack at any point
                            valid = True
                            for j in range(len(new_seq)):
                                if j > 0:
                                    # Check no immediate backtracking
                                    if new_seq[j][0] == new_seq[j-1][0] and new_seq[j][1] == -new_seq[j-1][1]:
                                        valid = False
                                        break
                            if valid:
                                walks_by_len[l+1].append(tuple(new_seq))
                        stack.append((nxt, curr, l+1, new_seq))
        return walks_by_len
    
    # Precompute all closed walks up to g-1
    max_walk_len = g - 1
    closed_walks = get_closed_walks(max_walk_len)
    
    # For each walk, compute the signed edge incidence vector
    # A walk has voltage product = identity iff for each edge, the net count is 0 mod p
    # Actually for additive voltages: sum of signed voltages = 0 mod p
    
    def walk_to_coeff(walk):
        """Convert walk to coefficient vector (net count per edge)."""
        coeff = [0] * num_edges
        for edge_idx, sign in walk:
            coeff[edge_idx] += sign
        return tuple(coeff)
    
    # Precompute coefficient vectors for all walks
    walk_coeffs = []
    for L in range(2, max_walk_len + 1):
        for walk in closed_walks[L]:
            coeff = walk_to_coeff(walk)
            # Skip trivial walks (back-and-forth)
            if all(c == 0 for c in coeff):
                continue
            walk_coeffs.append(coeff)
    
    # Remove duplicate coefficient vectors (keep unique)
    unique_coeffs = list(set(walk_coeffs))
    
    def check_voltage(voltages, p):
        """Check if voltage assignment has no zero-sum walk of length < g."""
        for coeff in unique_coeffs:
            s = 0
            for i in range(num_edges):
                if coeff[i] != 0:
                    s += coeff[i] * voltages[i]
            if s % p == 0:
                return False
        return True
    
    def build_lifted_graph(voltages, p):
        """Construct the lifted graph from voltage assignment."""
        n = num_vertices * p
        edges_out = set()
        for i in range(num_edges):
            u, v = edges_p[i]
            ve = voltages[i]
            for fiber in range(p):
                v1 = u * p + fiber
                v2 = v * p + (fiber + ve) % p
                if v1 < v2:
                    edges_out.add((v1, v2))
                else:
                    edges_out.add((v2, v1))
        return n, list(edges_out)
    
    # Find primitive root modulo prime
    def find_primitive_root(p):
        if p == 2:
            return 1
        n = p - 1
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        for g in range(2, p):
            ok = True
            for f in factors:
                if pow(g, (p-1)//f, p) == 1:
                    ok = False
                    break
            if ok:
                return g
        return None
    
    # Moore bound for (3,16): n >= 765
    # For Petersen lifts: n = 10*p, so p >= 77
    # Try primes starting from 79
    
    primes_to_try = []
    cand = 79
    while len(primes_to_try) < 50:
        is_prime = True
        if cand < 2:
            is_prime = False
        else:
            for d in range(2, int(cand**0.5) + 1):
                if cand % d == 0:
                    is_prime = False
                    break
        if is_prime:
            primes_to_try.append(cand)
        cand += 1
    
    for p in primes_to_try:
        alpha = find_primitive_root(p)
        if alpha is None:
            continue
        
        found = False
        best_voltages = None
        
        # Strategy 1: v_i = alpha^(a*i + b) mod p
        for a in range(1, min(50, p-1)):
            for b in range(0, min(20, p)):
                voltages = [pow(alpha, (a*i + b) % (p-1), p) for i in range(num_edges)]
                if check_voltage(voltages, p):
                    found = True
                    best_voltages = voltages
                    break
            if found:
                break
        
        if not found:
            # Strategy 2: v_i = alpha^(a*i^2 + b*i + c) mod p
            for a in range(1, min(20, p-1)):
                for b in range(0, min(15, p)):
                    for c in range(0, min(10, p)):
                        voltages = [pow(alpha, (a*i*i + b*i + c) % (p-1), p) for i in range(num_edges)]
                        if check_voltage(voltages, p):
                            found = True
                            best_voltages = voltages
                            break
                    if found:
                        break
                if found:
                    break
        
        if not found:
            # Strategy 3: Use different primitive roots with linear patterns
            for g in range(2, min(30, p)):
                is_prim = True
                n = p - 1
                d = 2
                factors = []
                temp_n = n
                while d * d <= temp_n:
                    if temp_n % d == 0:
                        factors.append(d)
                        while temp_n % d == 0:
                            temp_n //= d
                    d += 1
                if temp_n > 1:
                    factors.append(temp_n)
                for f in factors:
                    if pow(g, (p-1)//f, p) == 1:
                        is_prim = False
                        break
                if not is_prim:
                    continue
                for a in range(1, min(30, p-1)):
                    for b in range(0, min(15, p)):
                        voltages = [pow(g, (a*i + b) % (p-1), p) for i in range(num_edges)]
                        if check_voltage(voltages, p):
                            found = True
                            best_voltages = voltages
                            break
                    if found:
                        break
                if found:
                    break
        
        if not found:
            # Strategy 4: Deterministic pseudo-random
            for trial in range(500):
                voltages = []
                state = 12345 + trial * 7919
                for i in range(num_edges):
                    state = (state * 1103515245 + 12345) % (2**31)
                    v = (state % (p-1)) + 1
                    voltages.append(v)
                if check_voltage(voltages, p):
                    found = True
                    best_voltages = voltages
                    break
        
        if found:
            return build_lifted_graph(best_voltages, p)
    
    # Fallback with larger prime
    p = 257
    alpha = find_primitive_root(p)
    for trial in range(20000):
        state = 12345 + trial * 104729
        voltages = []
        for i in range(num_edges):
            state = (state * 1103515245 + 12345) % (2**31)
            v = (state % (p-1)) + 1
            voltages.append(v)
        if check_voltage(voltages, p):
            return build_lifted_graph(voltages, p)
    
    raise ValueError("Could not find valid voltage assignment")
