import numpy as np

def construct(k, g):
    # Petersen graph: 10 vertices, 15 edges, girth 5
    edges_p = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    num_edges = 15
    edge_index = {}
    for i, e in enumerate(edges_p):
        edge_index[e] = i
        edge_index[(e[1], e[0])] = i

    adj = [[] for _ in range(10)]
    for u, v in edges_p:
        adj[u].append(v)
        adj[v].append(u)

    # Generate all non-backtracking closed walks of length L in Petersen graph
    # Each walk is a list of (edge_idx, sign) where sign=+1 if traversing in canonical direction
    def get_walks_dir(max_len):
        walks = []
        for start in range(10):
            stack = [(start, -1, 0, [])]
            while stack:
                curr, prev, l, seq = stack.pop()
                if l == max_len:
                    if curr == start:
                        walks.append(tuple(seq))
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
        return walks

    # Precompute all constraint walks for lengths 5 to g-1
    target_g = g
    all_constraints = []
    for L in range(5, target_g):
        walks = get_walks_dir(L)
        all_constraints.extend(walks)

    # Convert constraints to coefficient vectors for efficient checking
    # Each constraint is a vector in Z_p^15
    # We'll represent as list of (edge_idx, sign) pairs
    # For speed, precompute as numpy arrays

    # Try primes starting from the smallest that gives n >= Moore bound
    # Moore bound for (3,16): 1 + 3*(1 + 2 + ... + 2^7) = 1 + 3*(2^8 - 1) = 1 + 3*255 = 766
    # Wait, let me recalculate. Moore bound for d-regular graph of girth g:
    # For odd g = 2r+1: n >= 1 + d * sum_{i=0}^{r-1} (d-1)^i
    # For even g = 2r: n >= d * sum_{i=0}^{r-1} (d-1)^i
    # g=16=2*8, so r=8: n >= 3 * sum_{i=0}^{7} 2^i = 3*(2^8-1) = 3*255 = 765
    # So n >= 765, meaning p >= 77 (since n=10p). But smallest known is 936, so p=93.6... 
    # Actually n must be 10*p, so p >= 77. Let's try primes >= 77.
    # But the current program tries from 53. The Moore bound says n>=765, so 10p>=765, p>=76.5, p>=77.
    # Primes: 79, 83, 89, 97, ...
    # But wait - the Moore bound is a lower bound. The actual minimum might be higher.
    # Smallest known is 936 vertices. 936/10 = 93.6, not an integer. So the known graph isn't a Petersen lift.
    # For Petersen lifts, n=10p. So we need 10p >= 765, i.e., p >= 77.
    # Let's try primes: 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293

    # Use deterministic algebraic voltage assignment
    # Strategy: for each prime p, try voltage assignments v_i = alpha^i for various choices
    # of primitive root alpha, and also other structured assignments.
    # Check if any assignment satisfies all constraints.

    def find_primitive_root(p):
        """Find a primitive root modulo prime p."""
        if p == 2:
            return 1
        # Factor p-1
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
        # Try candidates
        for g in range(2, p):
            ok = True
            for f in factors:
                if pow(g, (p-1)//f, p) == 1:
                    ok = False
                    break
            if ok:
                return g
        return None

    def check_voltages(voltages, p):
        """Check if voltage assignment satisfies all constraints."""
        for w in all_constraints:
            s = 0
            for e_idx, sign in w:
                s += sign * voltages[e_idx]
            if s % p == 0:
                return False
        return True

    # List of primes to try (must have 10*p >= 765, so p >= 77)
    primes_to_try = [79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]

    for p in primes_to_try:
        alpha = find_primitive_root(p)
        if alpha is None:
            continue

        # Try multiple deterministic voltage assignments
        found = False
        # Assignment 1: v_i = alpha^i
        voltages = [pow(alpha, i, p) for i in range(num_edges)]
        if check_voltages(voltages, p):
            found = True
        else:
            # Assignment 2: v_i = alpha^(i+1) * (i+1) mod p
            voltages = [(pow(alpha, i+1, p) * (i+1)) % p for i in range(num_edges)]
            if check_voltages(voltages, p):
                found = True
            else:
                # Assignment 3: v_i = alpha^(i^2)
                voltages = [pow(alpha, (i*i) % (p-1), p) for i in range(num_edges)]
                if check_voltages(voltages, p):
                    found = True
                else:
                    # Assignment 4: v_i = alpha^(i*7) (different step)
                    voltages = [pow(alpha, (i*7) % (p-1), p) for i in range(num_edges)]
                    if check_voltages(voltages, p):
                        found = True
                    else:
                        # Assignment 5: v_i = alpha^(i*11)
                        voltages = [pow(alpha, (i*11) % (p-1), p) for i in range(num_edges)]
                        if check_voltages(voltages, p):
                            found = True
                        else:
                            # Assignment 6: v_i = (i+1)^3 mod p
                            voltages = [pow(i+1, 3, p) for i in range(num_edges)]
                            if check_voltages(voltages, p):
                                found = True
                            else:
                                # Assignment 7: v_i = alpha^(i^2 + i)
                                voltages = [pow(alpha, ((i*i + i) % (p-1)), p) for i in range(num_edges)]
                                if check_voltages(voltages, p):
                                    found = True
                            if not found:
                                # Try more structured assignments with different primitive roots
                                for mult in [2, 3, 5, 13, 17, 19, 23, 29, 31, 37]:
                                    if mult < p:
                                        voltages = [pow(alpha, (i * mult) % (p-1), p) for i in range(num_edges)]
                                        if check_voltages(voltages, p):
                                            found = True
                                            break

        if found:
            # Construct the lifted graph
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

    # Fallback: return a valid graph (may not be minimal)
    # Use a larger prime and exhaustive-ish search
    p = 257
    alpha = find_primitive_root(p)
    # Try many random-like but deterministic assignments
    import random
    rng = random.Random(12345)
    for attempt in range(5000):
        voltages = [rng.randint(1, p-1) for _ in range(num_edges)]
        if check_voltages(voltages, p):
            n = 10 * p
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
    raise ValueError("Could not find valid voltage assignment")
