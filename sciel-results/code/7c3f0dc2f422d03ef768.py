import time
import random
import math
from collections import deque
import sys

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

    def precompute_inverses(q):
        inv = [0] * q
        if q > 1:
            inv[1] = 1
            for i in range(2, q):
                inv[i] = (q - (q // i) * inv[q % i]) % q
        return inv

    def precompute_quadratic_residues(q):
        residues = [False] * q
        residues[0] = True
        for x in range(1, q):
            residues[(x * x) % q] = True
        return residues

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

    # ---------- graph validation ----------
    def check_girth_bfs(adj, n, g):
        """BFS from each vertex, checking for cycles shorter than g."""
        half_g = g // 2
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                dx = dist[x]
                if dx >= half_g:
                    continue
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dx + 1
                        parent[y] = x
                        if dist[y] < g:
                            queue.append(y)
                    else:
                        if y != parent[x] and parent[y] != x:
                            cycle_len = dx + dist[y] + 1
                            if cycle_len < g:
                                return False
        return True

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return False
            if i in adj[i]:
                return False
            if len(adj[i]) != len(set(adj[i])):
                return False
        # Connectivity
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
        if count != n:
            return False
        # Girth
        if not check_girth_bfs(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline ----------
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

    # ---------- PGL(2, q) machinery ----------
    def action_on_point(x, a, b, c, d, q, inv_table):
        if x == q:
            if c == 0:
                return q
            return (a * inv_table[c]) % q
        num = (a * x + b) % q
        den = (c * x + d) % q
        if den == 0:
            return q
        return (num * inv_table[den]) % q

    def is_derangement(a, b, c, d, q, is_res_table):
        a, b, c, d = a % q, b % q, c % q, d % q
        if c == 0:
            return False
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False
        if is_res_table[disc]:
            return False
        return True

    def build_from_gens(gens, q, k, g, inv_table, is_res_table):
        n = q + 1
        adj = [set() for _ in range(n)]
        for i in range(n):
            x = i if i < q else q
            for gen in gens:
                y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q, inv_table)
                if y is None or y == i:
                    return None
                j = y if y < q else q
                adj[i].add(j)
                adj[j].add(i)
        return validate(adj, n, k, g)

    # ---------- Find involutory derangements ----------
    def find_involutions(q, k, is_res_table, time_limit):
        """Find involutory derangements in PGL(2,q).
        An involution in PGL(2,q) has matrix [[a,b],[c,-a]] with a^2+bc != 0.
        """
        found = []
        start = time.time()
        for a in range(q):
            for c in range(1, q):
                for b in range(q):
                    val = (a * a + b * c) % q
                    if val == 0:
                        continue
                    d = (-a) % q
                    if is_derangement(a, b, c, d, q, is_res_table):
                        found.append((a, b, c, d))
                        if len(found) >= k * 10:
                            if time.time() - start > time_limit:
                                return found
            if time.time() - start > time_limit:
                break
        return found

    # ---------- Main ----------
    # 1. deterministic baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)

    # 2. Search PGL(2, q) for q < best_n
    # Try prime powers q where n = q+1 < best_n
    # We want the smallest q that gives a valid graph

    # Candidate q values: prime powers from n_min-1 up to best_n-2
    candidates = []
    for q in range(max(3, n_min - 1), best_n - 1):
        if q % 2 == 1 and is_prime_power(q):
            candidates.append(q)

    # Sort by q (smaller first)
    candidates.sort()

    for q in candidates:
        if time.time() > deadline - 10.0:
            break
        n = q + 1
        if n >= best_n:
            break

        inv_table = precompute_inverses(q)
        is_res_table = precompute_quadratic_residues(q)

        # Find involutory derangements
        invols = find_involutions(q, k, is_res_table, 3.0)
        if len(invols) < k:
            continue

        # Try random subsets of involutions
        random.shuffle(invols)
        seen_sets = set()
        attempts = 0
        max_attempts = 2000
        start_t = time.time()
        budget = 8.0

        while attempts < max_attempts and time.time() - start_t < budget:
            attempts += 1
            if len(invols) <= k:
                gens = invols[:]
            else:
                gens = random.sample(invols, k)

            key = tuple(tuple(g) for g in gens)
            if key in seen_sets:
                continue
            seen_sets.add(key)

            res = build_from_gens(gens, q, k, g, inv_table, is_res_table)
            if res:
                if res[0] < best_n:
                    best_n = res[0]
                    best_result = res
                break

        # If we found a valid graph at this q, any larger q gives larger n
        # So we can return
        if best_n < n_luw:
            break

    return best_result

# EVOLVE-BLOCK-END
