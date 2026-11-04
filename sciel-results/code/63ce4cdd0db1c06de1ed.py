import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construct(k, g, d_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
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

    def verify_girth(n, edges, target_girth):
        """Verify girth >= target_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                v = queue.popleft()
                if dist[v] >= target_girth // 2:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v] and v != parent[w]:
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    def petersen_voltage_search(k, g, time_limit):
        """
        Voltage graph on Petersen graph base (10 vertices, 3-regular, girth 5).
        Lift over Z_p (p prime) to get 10p vertices.
        Use deterministic voltage assignments from GF(p) primitive elements.
        """
        # Petersen graph: vertices 0-4 (outer), 5-9 (inner)
        # Outer cycle: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        pet_edges = [(0,1),(1,2),(2,3),(3,4),(4,0),  # outer cycle
                      (5,7),(7,9),(9,6),(6,8),(8,5),  # inner star
                      (0,5),(1,6),(2,7),(3,8),(4,9)]  # spokes
        n_base = 10

        def build_lift(p, voltages):
            """Build the 10p-vertex lift graph."""
            n = n_base * p
            edges = []
            for (u, v) in pet_edges:
                w = voltages[(u, v)]
                for i in range(p):
                    u_node = u * p + i
                    v_node = v * p + ((i + w) % p)
                    edges.append((u_node, v_node))
            return n, edges

        def compute_girth_fast(n, edges, max_check):
            """Return the actual girth or max_check+1 if no shorter cycle found."""
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            for u in range(n):
                if len(adj[u]) != k:
                    return 0
            best_girth = max_check + 1
            half = max_check // 2
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                found_short = False
                while queue:
                    v = queue.popleft()
                    if dist[v] >= half:
                        continue
                    for w in adj[v]:
                        if dist[w] == -1:
                            dist[w] = dist[v] + 1
                            parent[w] = v
                            queue.append(w)
                        elif w != parent[v] and v != parent[w]:
                            cycle_len = dist[v] + dist[w] + 1
                            if cycle_len < best_girth:
                                best_girth = cycle_len
                                if best_girth <= max_check:
                                    found_short = True
                if found_short:
                    return best_girth
            return best_girth

        # Find primitive element in GF(p)
        def primitive_root(p):
            if p == 2:
                return 1
            factors = set()
            n = p - 1
            d = 2
            while d * d <= n:
                if n % d == 0:
                    factors.add(d)
                    while n % d == 0:
                        n //= d
                d += 1
            if n > 1:
                factors.add(n)
            for g in range(2, p):
                if all(pow(g, (p-1)//f, p) != 1 for f in factors):
                    return g
            return 2

        # Try primes p, lift gives 10p vertices
        # Moore bound for (3,16): 2*(1+3+9+27+81) = 216, so 10p >= 216 => p >= 22
        # Try primes: 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
        best_result = None
        best_n = float('inf')

        primes_to_try = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

        for p in primes_to_try:
            if time.time() > time_limit:
                break
            n = 10 * p
            if n >= best_n:
                continue

            gen = primitive_root(p)

            # Strategy: assign voltages using powers of primitive root
            # For each edge, assign a distinct power to maximize zero-sum avoidance
            # The Petersen graph has 15 edges. We need to assign voltages in Z_p.

            # Deterministic approach: use the structure
            # Outer cycle edges: assign voltages that sum to 0 mod p (necessary for no short cycles in outer)
            # Actually for girth >= 16, we need all cycles in the lift to have length >= 16.
            # A cycle in the base of length L lifts to cycles of length L, 2L, ... depending on voltage sum.
            # For the lift to have girth >= 16, every closed walk in the base of length < 16
            # must have non-zero voltage sum (or the lift cycle must be >= 16).

            # Key insight: Petersen has girth 5. Cycles of length 5, 6, 8, 9 in base.
            # For girth 16: no cycle of length 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 in lift.
            # A base cycle of length L with voltage sum s: in lift, it creates cycles of length
            # L * p / gcd(s, p) = L * p (since p is prime and s != 0) or L (if s = 0).
            # So we need: for every base cycle of length L < 16, voltage sum != 0 mod p.
            # Also for walks of length < 16 that aren't cycles, we need the lift to not close.

            # Simplified: ensure all base cycles of length <= 15 have non-zero voltage sum.
            # Petersen cycles: 5 (many), 6 (some), 8, 9, 10, 12, 15...

            # Let's try several deterministic voltage patterns:
            patterns = []

            # Pattern 1: sequential powers
            for start_pow in range(min(p, 20)):
                v = {}
                for idx, e in enumerate(pet_edges):
                    v[e] = (pow(gen, start_pow + idx, p)) % p
                patterns.append(v)

            # Pattern 2: alternating
            for offset in range(min(p, 10)):
                v = {}
                for idx, e in enumerate(pet_edges):
                    if idx % 2 == 0:
                        v[e] = (offset + idx) % p
                    else:
                        v[e] = (p - 1 - (offset + idx) % p) % p
                patterns.append(v)

            # Pattern 3: use different primitive roots / multipliers
            for mult in range(1, min(p, 15)):
                v = {}
                for idx, e in enumerate(pet_edges):
                    v[e] = (mult * (idx + 1)) % p
                patterns.append(v)

            # Pattern 4: quadratic residues
            for offset in range(min(p, 10)):
                v = {}
                for idx, e in enumerate(pet_edges):
                    v[e] = ((offset + idx) * (offset + idx)) % p
                patterns.append(v)

            # Evaluate patterns
            for v in patterns:
                if time.time() > time_limit:
                    break
                _, edges = build_lift(p, v)
                girth = compute_girth_fast(n, edges, g)
                if girth >= g:
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    break
                if girth >= g - 2:
                    # Close - try local perturbation
                    for _ in range(50):
                        if time.time() > time_limit:
                            break
                        v2 = dict(v)
                        e_rand = random.choice(pet_edges)
                        v2[e_rand] = random.randint(0, p - 1)
                        _, edges2 = build_lift(p, v2)
                        girth2 = compute_girth_fast(n, edges2, g)
                        if girth2 >= g:
                            if n < best_n:
                                best_n = n
                                best_result = (n, edges2)
                            break

            if best_result is not None:
                break

        return best_result

    if k != 3 or g != 16:
        return luw_construct(k, g)

    # Try Petersen voltage graph approach (structural change)
    pg_result = petersen_voltage_search(k, g, deadline - 15)
    if pg_result is not None:
        n, edges = pg_result
        if verify_girth(n, edges, g):
            return n, edges

    # Fallback: try LUW with various d
    best_n = float('inf')
    best_edges = None
    for d in range(7, 12):
        if time.time() > deadline:
            break
        n, edges = luw_construct(k, g, d_override=d)
        if n < best_n:
            if verify_girth(n, edges, g):
                best_n = n
                best_edges = edges

    if best_edges is not None:
        return best_n, best_edges

    return luw_construct(k, g)
