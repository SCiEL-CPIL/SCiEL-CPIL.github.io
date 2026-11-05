import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    def remaining():
        return deadline - time.time()

    # ---------- arithmetic helpers ----------
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

    def is_quadratic_residue(a, q):
        """Check if a is a quadratic residue mod q (works for prime powers)."""
        a = a % q
        if a == 0:
            return True
        if not is_prime_power(q):
            return True  # conservative: assume residue
        # Find prime base p of q
        p = None
        for cand in range(2, int(math.isqrt(q)) + 1):
            if q % cand == 0:
                t = q
                while t % cand == 0:
                    t //= cand
                if t == 1:
                    p = cand
                    break
        if p is None:
            p = q  # q is prime
        # Reduce to mod p
        a_p = a % p
        if a_p == 0:
            return True
        # Euler's criterion mod p (p is prime)
        return pow(a_p, (p - 1) // 2, p) == 1

    def modinv(a, m):
        """Extended Euclidean algorithm for modular inverse (works for any modulus)."""
        if m <= 1:
            return None
        a = a % m
        if a == 0:
            return None
        old_r, r = a, m
        old_s, s = 1, 0
        while r != 0:
            qv = old_r // r
            old_r, r = r, old_r - qv * r
            old_s, s = s, old_s - qv * s
        if old_r != 1:
            return None
        return old_s % m

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:  # g = 2r + 1
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:  # g = 2r
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    # ---------- fast girth check using non-backtracking walks ----------
    def check_girth_fast(adj, n, g):
        """Check girth >= g using non-backtracking walks from each vertex.
        A cycle of length < g exists iff there's a non-backtracking walk
        of length < g that returns to its start."""
        for u in range(n):
            # BFS on directed edges (u, v) where v != parent
            # State: (current, previous)
            visited = set()
            queue = deque()
            for v in adj[u]:
                queue.append((v, u))
                visited.add((v, u))
            depth = 1
            while queue and depth < g:
                new_queue = deque()
                for (cur, prev) in queue:
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue
                        if nxt == u:
                            return False  # found cycle of length depth+1 < g
                        state = (nxt, cur)
                        if state not in visited:
                            visited.add(state)
                            new_queue.append(state)
                queue = new_queue
                depth += 1
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
        if not check_girth_fast(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline (deterministic) ----------
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

    # ---------- PGL(2, q) machinery (q odd prime power) ----------
    def action_on_point(x, a, b, c, d, q):
        if x == q:  # infinity
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
        """Check if Möbius transform has no fixed points on P^1(F_q)."""
        a, b, c, d = a % q, b % q, c % q, d % q
        if c == 0:
            return False
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False
        if is_quadratic_residue(disc, q):
            return False
        return True

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
        # Check degrees
        for i in range(n):
            if len(adj[i]) != k:
                return None
        # Validate girth
        adj_list = [list(s) for s in adj]
        return validate(adj_list, n, k, g)

    # deterministic structured generator set
    def try_structured_pgl(q, k, g):
        if q % 2 == 0 or not is_prime_power(q):
            return None
        n = q + 1
        if n < n_min:
            return None

        def is_nonsquare(a):
            a %= q
            if a == 0:
                return False
            return not is_quadratic_residue(a, q)

        # G_c: x -> (x + c)/x = (1, c, 1, 0); derangement iff 1 + 4c is a non-square
        cands = []
        for cc in range(1, q):
            if is_nonsquare(1 + 4 * cc):
                cands.append(cc)
            if len(cands) >= 12:
                break
        if len(cands) < 2:
            return None

        h_ok = is_nonsquare(8)
        h2_cands = []
        for cc in range(1, q):
            if is_nonsquare(1 - cc * cc):
                h2_cands.append(cc)
            if len(h2_cands) >= 10:
                break

        c1, c2 = cands[0], cands[1]
        g1 = (1, c1, 1, 0)
        g1i = inv_matrix(*g1, q)
        g2 = (1, c2, 1, 0)
        g2i = inv_matrix(*g2, q)
        if g1i is None or g2i is None:
            return None
        h = (1, 1, 1, q - 1)

        if h_ok:
            res = build_from_gens([g1, g1i, g2, g2i, h], q, k, g)
            if res:
                return res
            if len(cands) >= 3:
                g2b = (1, cands[2], 1, 0)
                g2bi = inv_matrix(*g2b, q)
                if g2bi is not None:
                    res = build_from_gens([g1, g1i, g2b, g2bi, h], q, k, g)
                    if res:
                        return res
            if len(cands) >= 4:
                g3b = (1, cands[3], 1, 0)
                g3bi = inv_matrix(*g3b, q)
                if g3bi is not None:
                    res = build_from_gens([g1, g1i, g3b, g3bi, h], q, k, g)
                    if res:
                        return res

        for cc in h2_cands[:6]:
            g3 = (1, q - cc, 1, cc)
            res = build_from_gens([g1, g1i, g2, g2i, g3], q, k, g)
            if res:
                return res
        return None

    # Improved random search in PGL(2, q) with better heuristics
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
            """Generate k generators, preferring involutions and elements of small order."""
            out = []
            attempts = 0
            max_attempts = k * 50
            while len(out) < k and attempts < max_attempts:
                attempts += 1
                m = random_gen()
                mi = inv_matrix(*m, q)
                if mi is None:
                    continue
                if not is_derangement(*m, q):
                    continue
                # Check if involution
                a2 = (m[0]*m[0] + m[1]*m[2]) % q
                b2 = (m[1]*(m[0]+m[3])) % q
                c2 = (m[2]*(m[0]+m[3])) % q
                d2 = (m[3]*m[3] + m[1]*m[2]) % q
                is_involution = (a2 == 1 and b2 == 0 and c2 == 0 and d2 == 1)
                
                if is_involution:
                    out.append(m)
                    # Involutions are self-inverse, don't add mi
                elif random.random() < 0.5:
                    out.append(m)
                    if mi != m and len(out) < k:
                        out.append(mi)
            while len(out) < k:
                m = random_gen()
                if is_derangement(*m, q):
                    out.append(m)
            return out[:k]

        seen = set()
        while time.time() < budget:
            gens = random_gens()
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            if not all(is_derangement(*gen, q) for gen in gens):
                continue
            res = build_from_gens(gens, q, k, g)
            if res:
                return res
        return None

    # ---------- Random regular graph construction via configuration model + local repair ----------
    def random_regular_graph(n, k, g, budget):
        """Try to build a k-regular graph with girth >= g using random pairing + local repair."""
        if n * k % 2 != 0:
            return None
        if n < n_min:
            return None

        while time.time() < budget:
            # Random 2-regular multigraph (configuration model)
            # Each vertex has k stubs
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            # Pair stubs to form edges
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if not valid:
                continue
            
            # Check degrees
            deg_ok = True
            for i in range(n):
                if len(adj[i]) != k:
                    deg_ok = False
                    break
            if not deg_ok:
                continue
            
            # Check girth
            if not is_connected(adj, n):
                continue
            if not check_girth_fast(adj, n, g):
                # Try local repair: find short cycle and break it
                # Find a short cycle
                for u in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[u] = 0
                    queue = deque([u])
                    cycle_found = False
                    while queue and not cycle_found:
                        x = queue.popleft()
                        for y in adj[x]:
                            if dist[y] == -1:
                                dist[y] = dist[x] + 1
                                parent[y] = x
                                queue.append(y)
                            elif y != parent[x] and parent[y] != x:
                                if dist[x] + dist[y] + 1 < g:
                                    # Found short cycle; try to break it
                                    # Find the cycle path
                                    # This is complex; skip for now
                                    cycle_found = True
                                    break
                    if cycle_found:
                        break
                continue
            
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return (n, edges)
        return None

    # ---------- main ----------
    # 1. deterministic baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # 2. deterministic structured PGL for small q
    q_max_struct = min(best_n[0] - 2, 300)
    for curr_q in range(max(3, n_min - 1), q_max_struct + 1):
        if time.time() > deadline - 40.0:
            break
        res = try_structured_pgl(curr_q, k, g)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            # Found a PGL graph at this q; try a few more q values briefly
            for curr_q2 in range(curr_q + 1, min(curr_q + 8, best_n[0] - 1)):
                if time.time() > deadline - 40.0:
                    break
                res2 = try_structured_pgl(curr_q2, k, g)
                if res2 and res2[0] < best_n[0]:
                    best_n[0] = res2[0]
                    best_result = res2
            return best_result

    # 3. random PGL search, time-boxed, q increasing
    q_cursor = max(3, n_min - 1)
    while q_cursor + 1 < best_n[0] and time.time() < deadline - 15.0:
        time_left = deadline - 15.0 - time.time()
        if time_left <= 0:
            break
        q_range = best_n[0] - 1 - q_cursor
        if q_range > 0:
            budget = min(time_left, max(0.3, 2.0 * (2.0 / (1.0 + q_range))))
        else:
            budget = time_left
        res = random_search_pgl(q_cursor, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            for _ in range(3):
                if time.time() > deadline - 15.0:
                    break
                q_cursor += 1
                if q_cursor + 1 >= best_n[0]:
                    break
                b2 = min(deadline - 15.0 - time.time(), 1.5)
                if b2 <= 0:
                    break
                res2 = random_search_pgl(q_cursor, k, g, b2)
                if res2 and res2[0] < best_n[0]:
                    best_n[0] = res2[0]
                    best_result = res2
            break
        q_cursor += 1

    # 4. Random regular graph search for n between moore_bound and best_n
    if time.time() < deadline - 5.0:
        # Try a few random n values
        for trial_n in range(n_min, min(best_n[0], n_min + 50)):
            if time.time() > deadline - 5.0:
                break
            budget = min(deadline - 5.0 - time.time(), 0.5)
            if budget <= 0:
                break
            res = random_regular_graph(trial_n, k, g, budget)
            if res and res[0] < best_n[0]:
                best_n[0] = res[0]
                best_result = res
                break

    return best_result

# EVOLVE-BLOCK-END
