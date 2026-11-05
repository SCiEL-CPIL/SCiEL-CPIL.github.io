import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                return t == 1
        return False

    def modinv(a, m):
        if m <= 1:
            return None
        try:
            return pow(a, -1, m)
        except (ValueError, ZeroDivisionError):
            return None

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    def check_girth_nb(adj, n, g):
        """Check girth using non-backtracking walks from every vertex."""
        for u in range(n):
            dist = {}
            queue = deque()
            for v in adj[u]:
                dist[(v, u)] = 1
                queue.append((v, u))
            while queue:
                x, prev = queue.popleft()
                d = dist[(x, prev)]
                if d >= g - 1:
                    continue
                for y in adj[x]:
                    if y == prev:
                        continue
                    if y == u:
                        if d + 1 < g:
                            return False
                        continue
                    key = (y, x)
                    if key not in dist:
                        dist[key] = d + 1
                        queue.append((y, x))
        return True

    def is_connected(adj, n):
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return False
        if not is_connected(adj, n):
            return False
        if not check_girth_nb(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    def luw_construct(k, g):
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

    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    def cyclic_interval_search(k, g, max_n, deadline):
        """
        Search for a k-regular graph on Z_n with girth >= g using cyclic intervals.
        Vertex i connects to i +/- d_1, ..., i +/- d_{k//2}, and optionally i + d_{k//2+1} if k odd.
        For k=5: neighbors are i +/- d1, i +/- d2, i + d3 (asymmetric for the 5th edge).
        Actually for k=5, we need 5 neighbors. Use i +/- d1, i +/- d2, i + d3.
        But this makes it not vertex-transitive. Let's use a different approach.
        
        For k=5, use: i -> i+d1, i-d1, i+d2, i-d2, i+d3 where d3 != d1, d2 and d3 not creating short cycles.
        To ensure regularity, we need the graph to be well-defined on Z_n.
        
        Alternative: use a circulant graph with connection set S of size k.
        For k=5: S = {d1, -d1, d2, -d2, d3} where we need to be careful.
        Actually for a circulant graph to be k-regular, |S| = k and S is symmetric (if d in S then -d in S) 
        unless we handle it differently. For odd k, we can have S not symmetric but the graph is still regular.
        
        Let's use: for k=5, S = {a, -a, b, -b, c} where c != 0, a, -a, b, -b.
        The graph is 5-regular if all these are distinct mod n.
        
        We need: no non-trivial sum of elements from S (with repetition allowed, length < g) equals 0 mod n.
        More precisely, no walk of length < g returns to start without backtracking.
        """
        if k != 5:
            return None
        
        # For k=5, try connection sets {a, -a, b, -b, c}
        # a, b, c must be distinct and nonzero, and c != +/-a, c != +/-b
        # Also need: no short cycles
        
        # Minimum possible n: need at least 2*max(a,b,c) + 1 roughly, but more precisely
        # We need n large enough so that all sums of length < g-1 don't wrap around to 0
        # unless they represent a valid non-backtracking walk that doesn't close
        
        # Strategy: scan n from Moore bound upward, and for each n try various (a, b, c)
        
        # For girth g, we need that no combination of at most g-1 steps (non-backtracking) sums to 0
        # This is complex. Instead, let's just build the graph and check.
        
        # Heuristic: a and b should be "small" relative to n to avoid wrapping, 
        # but large enough to avoid short cycles.
        
        # Let's try: a = 1, b = n//5, c = 2*n//5 (roughly)
        # Or better: use known constructions.
        
        # For girth 9, k=5: try n around 40-100 first
        # Known: there exist 5-regular girth-9 graphs on ~100 vertices
        
        # Let's systematically search
        # For each n, try several (a, b, c) combinations
        
        # The key constraint: the set of all sums of length L (for L < g) using elements from S
        # should not contain 0 (except for the trivial case).
        # But non-backtracking is harder to check analytically.
        
        # Simpler approach: just build and check.
        
        # To make this efficient, we can pre-filter:
        # For girth >= g, we need that no sequence of < g steps (non-backtracking) returns to start.
        # A necessary condition: for any sequence s1, s2, ..., sm with si in S, s_{i+1} != -s_i,
        # sum(si) != 0 mod n for m < g.
        
        # Let's just try many combinations and validate.
        
        best = None
        
        # Try n from a reasonable minimum upward
        # Moore bound for k=5, g=9 is 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1 + 5 + 20 + 80 + 320 = 426
        # So n >= 426. But we know solutions exist around 100-200 for smaller girth.
        # Wait, the Moore bound is a lower bound, so n >= 426 for k=5, g=9.
        # But the task says the smallest known is 1116 and best lower bound is 428.
        # So we're looking for n in [428, 1116].
        
        # This is a large search space. Let's be smart about it.
        
        # For circulant graphs with S = {a, -a, b, -b, c}, the girth is determined by
        # the shortest non-trivial relation sum(si) = 0 mod n with si in S and no consecutive inverses.
        
        # Let's try a structured search:
        # a = 1 (always)
        # b = some value
        # c = some value
        
        # For efficiency, limit the search to promising n values.
        
        # Actually, let's try a different approach: use the fact that for large n relative to the
        # step sizes, wrapping doesn't happen, so we can check integer sums.
        
        # If n > (g-1) * max(|a|,|b|,|c|), then no wrapping occurs for walks of length < g,
        # and we just need no integer combination of length < g (non-backtracking) sums to 0.
        
        # With a=1, this is easy: 1 + (-1) = 0 is a 2-step backtrack, which is excluded.
        # But 1 + 1 + (-1) + (-1) = 0 is a 4-step walk that backtracks.
        # Non-backtracking means s_{i+1} != -s_i.
        
        # Let's just do a direct search with validation.
        
        # To keep it fast, limit to n up to some reasonable bound and try few (a,b,c) per n.
        
        max_search_n = min(max_n, 2000)  # Don't go too high
        
        # Try n values, skipping those too small
        min_n = max(n_min, 2 * max(1, g) + 10)  # Rough lower bound
        
        for n in range(min_n, max_search_n + 1):
            if time.time() > deadline - 5.0:
                break
            
            # Try several (a, b, c) combinations
            # a = 1 always
            # b and c should be "well-separated"
            
            # Try b = n//5, c = 2*n//5, and variations
            candidates = []
            for b in range(2, n // 3 + 1):
                for c in range(b + 1, n // 2 + 1):
                    if c == n - b:  # c = -b mod n, skip
                        continue
                    if c == n - 1:  # c = -1 mod n, skip
                        continue
                    candidates.append((1, b, c))
            
            # Too many candidates. Let's be more selective.
            # Try a few specific patterns:
            
            test_sets = [
                (1, n // 5, 2 * n // 5),
                (1, n // 6, n // 3),
                (1, n // 7, 2 * n // 7),
                (1, n // 8, n // 4),
                (1, 2, n // 5),
                (1, 3, n // 5),
                (1, n // 5, 3 * n // 10),
                (1, n // 6, 5 * n // 12),
            ]
            
            for (a, b, c) in test_sets:
                if time.time() > deadline - 5.0:
                    break
                
                # Ensure all distinct and nonzero mod n
                steps = [a, -a % n, b, -b % n, c]
                if len(set(steps)) != 5 or 0 in steps:
                    continue
                
                # Build adjacency
                adj = [[] for _ in range(n)]
                for i in range(n):
                    for s in steps:
                        j = (i + s) % n
                        if j != i:
                            adj[i].append(j)
                
                # Check degree
                valid = True
                for i in range(n):
                    if len(adj[i]) != k:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check connected
                if not is_connected(adj, n):
                    continue
                
                # Check girth
                if check_girth_nb(adj, n, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return (n, edges)
        
        return None

    def action_on_point(x, a, b, c, d, q):
        if x == q:
            if c == 0:
                return q
            inv_c = modinv(c, q)
            if inv_c is None:
                return None
            return (a * inv_c) % q
        num = (a * x + b) % q
        den = (c * x + d) % q
        if den == 0:
            return q
        inv_den = modinv(den, q)
        if inv_den is None:
            return None
        return (num * inv_den) % q

    def inv_matrix(a, b, c, d, q):
        det = (a * d - b * c) % q
        det_inv = modinv(det, q)
        if det_inv is None:
            return None
        return (d * det_inv % q, -b * det_inv % q, -c * det_inv % q, a * det_inv % q)

    def is_derangement(a, b, c, d, q):
        if c == 0:
            return False
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False
        return pow(disc, (q - 1) // 2, q) != 1

    def build_from_gens(gens, q, k, g):
        n = q + 1
        adj = [set() for _ in range(n)]
        for i in range(n):
            x = i if i < q else q
            for gen in gens:
                y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                if y is None or y == i:
                    return None
                j = y if y < q else q
                adj[i].add(j)
                adj[j].add(i)
        for i in range(n):
            if len(adj[i]) != k:
                return None
        adj_list = [sorted(adj[i]) for i in range(n)]
        return validate(adj_list, n, k, g)

    def random_search_pgl(q, k, g, budget):
        n = q + 1
        if n < n_min or n >= best_n[0]:
            return None
        if q % 2 == 0 or not is_prime_power(q):
            return None

        def random_gen():
            while True:
                a = random.randrange(q)
                b = random.randrange(q)
                c = random.randrange(q)
                d = random.randrange(q)
                if (a * d - b * c) % q != 0:
                    return (a, b, c, d)

        def random_gens():
            out = []
            for _ in range(k):
                m = random_gen()
                if random.random() < 0.5:
                    mi = inv_matrix(*m, q)
                    if mi is not None:
                        out.extend([m, mi])
                        if len(out) >= k:
                            return out[:k]
                else:
                    out.append(m)
            return out

        seen = set()
        while time.time() < budget:
            gens = random_gens()
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            if any((a * d - b * c) % q == 0 for (a, b, c, d) in gens):
                continue
            if not all(is_derangement(*gen, q) for gen in gens):
                continue
            res = build_from_gens(gens, q, k, g)
            if res:
                return res
        return None

    # First, try cyclic interval construction (fast, deterministic)
    if time.time() < deadline - 10.0:
        res = cyclic_interval_search(k, g, best_n[0] - 1, deadline - 5.0)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res

    # Then, try PGL search if time remains
    if time.time() < deadline - 5.0:
        q_cursor = max(3, n_min - 1)
        while q_cursor + 1 < best_n[0] and time.time() < deadline - 3.0:
            budget = min(deadline - 3.0, time.time() + 1.0)
            res = random_search_pgl(q_cursor, k, g, budget)
            if res and res[0] < best_n[0]:
                best_n[0] = res[0]
                best_result = res
            q_cursor += 1

    return best_result

# EVOLVE-BLOCK-END
