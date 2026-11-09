"""
Combined approach: use the polarity graph (a provably C4-free graph) as a strong
structured seed, then refine with fast bitmask-based local search.

Key structural change vs. the current program:
  * The current program's "structured" helper actually falls back to random,
    so it is effectively the same family as the random strategies.
  * Here we build the true polarity graph of PG(2, q) for the largest prime
    power q with q^2+q+1 <= n, which is C4-free and has min degree q.
    For n=46, q=6 (37 vertices, degree 6) is the largest available, but we
    can also use q=7 (57 vertices) and take a subgraph of it.  More
    importantly, the polarity graph gives a *dense, structured* starting
    point where every pair of non-adjacent vertices has at most one common
    neighbour, which makes the subsequent edge-addition local search far
    more effective than starting from a sparse random graph.

  * We also add a "complement perspective" repair: since the complement must
    have no K1,39, we can view the problem as making the complement sparse
    enough.  We try to add edges (i.e. remove non-edges) that reduce the
    maximum degree in the complement while preserving C4-freeness.
"""
import random
import time
import math
import numpy as np

def construct():
    start = time.time()
    deadline = 148.0  # safety margin before 150s kill

    # ---------- helpers ----------

    def is_prime(m):
        if m < 2:
            return False
        if m < 4:
            return True
        if m % 2 == 0 or m % 3 == 0:
            return False
        i = 5
        while i * i <= m:
            if m % i == 0 or m % (i + 2) == 0:
                return False
            i += 6
        return True

    def is_prime_power(m):
        if m < 2:
            return False
        if is_prime(m):
            return True
        # check small prime powers
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if p > m:
                break
            t = m
            while t % p == 0:
                t //= p
            if t == 1:
                # m = p^k
                return True
        return False

    def c4_free_check(n, adj):
        """Return True if graph is C4-free. adj is list of sets."""
        masks = []
        for i in range(n):
            m = 0
            for j in adj[i]:
                m |= (1 << j)
            masks.append(m)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common = masks[u] & masks[v]
                    if common & (common - 1):
                        return False
        return True

    def min_degree(adj):
        return min(len(adj[i]) for i in range(len(adj)))

    def valid(n, adj):
        return min_degree(adj) >= n - 39 and c4_free_check(n, adj)

    def to_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    def can_add(adj, u, v):
        """Can we add edge (u,v) without creating a C4?"""
        if v in adj[u]:
            return False
        common = adj[u] & adj[v]
        return len(common) <= 1

    # ---------- polarity graph construction ----------

    def build_polarity_graph(q):
        """
        Build the polarity graph of PG(2, q).
        Vertices: all 1-dimensional subspaces of F_q^3, i.e. points of PG(2,q).
        There are q^2+q+1 vertices.
        Two points are adjacent if they are conjugate under a suitable
        polarity.  We use the standard polarity defined by the symmetric
        matrix [[0,1,0],[1,0,0],[0,0,1]] over F_q (works for any q).
        Under this polarity, point [x:y:z] is adjacent to [x':y':z'] iff
        x*y' + y*x' + z*z' = 0  (dot product = 0).
        This is a symmetric bilinear form, so adjacency is symmetric.
        It is known to be C4-free and q-regular.
        """
        # Enumerate all projective points: representatives with first nonzero coord = 1
        points = []
        # Type 1: [1, a, b] for a, b in F_q
        for a in range(q):
            for b in range(q):
                points.append((1, a, b))
        # Type 2: [0, 1, a] for a in F_q
        for a in range(q):
            points.append((0, 1, a))
        # Type 3: [0, 0, 1]
        points.append((0, 0, 1))
        # Total: q^2 + q + 1
        n_pts = len(points)
        adj = [set() for _ in range(n_pts)]
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                # dot product mod q
                dp = (x1 * y2 + y1 * x2 + z1 * z2) % q
                if dp == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj, n_pts

    def polarity_seed(n, min_deg, seed=42):
        """
        Build a C4-free graph on n vertices with min degree >= min_deg
        using a polarity graph as seed.
        Strategy: find largest prime power q with q^2+q+1 <= n, build
        polarity graph on q^2+q+1 vertices, then add remaining vertices
        connected to a carefully chosen subset to maintain C4-freeness.
        """
        random.seed(seed)
        # Find best q
        best_q = 0
        for q in range(2, 50):
            if is_prime_power(q) and q * q + q + 1 <= n:
                best_q = q
        if best_q < 2:
            return None

        adj_base, base_n = build_polarity_graph(best_q)
        # Copy to n-vertex graph
        adj = [set() for _ in range(n)]
        for i in range(base_n):
            adj[i] = set(adj_base[i])

        # Add remaining vertices (if any)
        for v in range(base_n, n):
            # Connect v to a subset of existing vertices
            # To avoid C4, v should share at most 1 common neighbour with any
            # pair of its neighbours.  Simplest: connect to a small independent set.
            # Try to connect to ~min_deg vertices chosen to minimise C4 risk.
            candidates = list(range(base_n))
            random.shuffle(candidates)
            for u in candidates:
                if len(adj[v]) >= min_deg:
                    break
                if can_add(adj, v, u):
                    adj[v].add(u)
                    adj[u].add(v)

        # Try to boost degrees of base vertices if needed
        for u in range(n):
            if len(adj[u]) < min_deg:
                candidates = [v for v in range(n) if v != u and v not in adj[u]]
                random.shuffle(candidates)
                for v in candidates:
                    if len(adj[u]) >= min_deg:
                        break
                    if can_add(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)

        if min_degree(adj) >= min_deg and c4_free_check(n, adj):
            return adj
        return None

    # ---------- local search ----------

    def local_search(adj, n, min_deg, time_limit):
        """
        Iteratively try to add edges to increase min degree while preserving
        C4-freeness.  Returns True if valid at end.
        """
        deadline = time.time() + time_limit
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Process vertices with lowest degree first
            vertices = sorted(range(n), key=lambda i: len(adj[i]))
            for u in vertices:
                if len(adj[u]) >= min_deg:
                    continue
                candidates = [v for v in range(n) if v != u and v not in adj[u]]
                random.shuffle(candidates)
                for v in candidates:
                    if time.time() > deadline:
                        return c4_free_check(n, adj) and min_degree(adj) >= min_deg
                    if can_add(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        improved = True
                        if len(adj[u]) >= min_deg:
                            break
                if time.time() > deadline:
                    return c4_free_check(n, adj) and min_degree(adj) >= min_deg
        return c4_free_check(n, adj) and min_degree(adj) >= min_deg

    def random_repair(adj, n, min_deg, time_limit):
        """
        If graph has C4s, try removing edges to break them, then re-add.
        """
        deadline = time.time() + time_limit
        for _ in range(200):
            if time.time() > deadline:
                break
            if c4_free_check(n, adj):
                if min_degree(adj) >= min_deg:
                    return True
                # Try to add edges
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for u in range(n):
                        if len(adj[u]) >= min_deg:
                            continue
                        candidates = [v for v in range(n) if v != u and v not in adj[u]]
                        random.shuffle(candidates)
                        for v in candidates:
                            if len(adj[u]) >= min_deg:
                                break
                            if can_add(adj, u, v):
                                adj[u].add(v)
                                adj[v].add(u)
                                improved = True
                if c4_free_check(n, adj) and min_degree(adj) >= min_deg:
                    return True
                continue
            # Find a C4 and break it
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        common = adj[u] & adj[v]
                        if len(common) > 1:
                            cl = list(common)
                            w = cl[0]
                            # Remove edge u-w or v-w
                            if random.random() < 0.5:
                                adj[u].discard(w)
                                adj[w].discard(u)
                            else:
                                adj[v].discard(w)
                                adj[w].discard(v)
                            break
                else:
                    continue
                break
        return c4_free_check(n, adj) and min_degree(adj) >= min_deg

    def random_greedy(n, min_deg, seed):
        """Random greedy C4-free construction."""
        random.seed(seed)
        adj = [set() for _ in range(n)]
        # Start with a random sparse graph
        p = max(0.3, min(0.95, min_deg / max(1, n - 1)))
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    if can_add(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
        # Complete
        success = local_search(adj, n, min_deg, 2.0)
        if not success:
            success = random_repair(adj, n, min_deg, 3.0)
        return adj if success else None

    # ---------- main search ----------

    best = {'n': 39, 'edges': []}

    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        min_deg = n - 39
        found = False

        # Strategy 1: Polarity graph seed (structured)
        if time.time() < deadline - 5:
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                adj = polarity_seed(n, min_deg, seed=10000 + n * 7 + attempt)
                if adj is not None:
                    # Refine with local search
                    if local_search(adj, n, min_deg, 5.0):
                        best = {'n': n, 'edges': to_edges(adj, n)}
                        found = True
                        break
                if time.time() > deadline - 5:
                    break
        if found:
            break

        # Strategy 2: Random greedy with multiple seeds
        if time.time() < deadline - 5:
            max_attempts = 300
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                seed = attempt * 7919 + 12345 + n
                adj = random_greedy(n, min_deg, seed)
                if adj is not None:
                    best = {'n': n, 'edges': to_edges(adj, n)}
                    found = True
                    break
        if found:
            break

    # Final verification
    if best['n'] > 39:
        n = best['n']
        adj = [set() for _ in range(n)]
        for e in best['edges']:
            u, v = e
            adj[u].add(v)
            adj[v].add(u)
        if not valid(n, adj):
            best = {'n': 39, 'edges': []}

    return best
