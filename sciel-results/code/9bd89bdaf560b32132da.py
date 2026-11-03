import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # 5s margin

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))

    # ---- Lazebnik-Ustimenko-Woldar construction (seed) ----
    def seed_construct(k, g):
        q = k
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
            r_c, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_c]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    seed_n, seed_edges = seed_construct(k, g)

    # ---- Helper functions ----
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Check girth >= g using BFS from each vertex with early termination."""
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def heuristic_girth_check(n, adj, g, sample_size=10):
        """Fast heuristic: check girth from a sample of vertices."""
        half_g = g // 2
        sources = random.sample(range(n), min(sample_size, n))
        for s in sources:
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def generate_simple_regular(n, k):
        # Use configuration model with rejection for multi-edges/self-loops
        for attempt in range(200):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            edge_set = set()
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append((u, v))
            if valid:
                return edges
        return None

    def generate_regular_via_matchmaking(n, k):
        """Generate k-regular graph using round-robin style matching."""
        if n % 2 == 0:
            # For even n, use circulant-like construction
            if k % 2 == 0:
                # Pair up vertices in k/2 perfect matchings
                edges = []
                edge_set = set()
                for step in range(1, k // 2 + 1):
                    for i in range(n // 2):
                        u = i
                        v = (i + step) % n
                        # Also connect the "other half"
                        u2 = i + n // 2
                        v2 = (i + step) % n + n // 2
                        for (a, b) in [(u, v), (u2, v2)]:
                            if a == b:
                                return None
                            key = (min(a, b), max(a, b))
                            if key in edge_set:
                                return None
                            edge_set.add(key)
                            edges.append((a, b))
                return edges
            else:
                # k is odd, n must be even
                # Use k-1 even case plus one random perfect matching
                base = generate_regular_via_matchmaking(n, k - 1)
                if base is None:
                    return None
                # Find unused pairs for perfect matching
                adj_temp = [set() for _ in range(n)]
                for u, v in base:
                    adj_temp[u].add(v)
                    adj_temp[v].add(u)
                # Random perfect matching on remaining stubs
                remaining = list(range(n))
                random.shuffle(remaining)
                edges = list(base)
                edge_set = set((min(u,v), max(u,v)) for u,v in base)
                for i in range(0, n, 2):
                    u, v = remaining[i], remaining[i+1]
                    if u == v or v in adj_temp[u]:
                        return None
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        return None
                    edge_set.add(key)
                    edges.append((u, v))
                return edges
        return None

    # ---- Incidence graph of generalized hexagon of order (2, 4) ----
    # This gives a 5-regular bipartite graph with girth 12 on 126 vertices.
    # Points: 21 (lines of PG(2,4)), Lines: 105 (points of PG(2,4))
    # Incidence: point p is incident with line l if p lies on l
    # Each point is on 5 lines, each line has 5 points -> 5-regular
    # Girth = 12 (bipartite, diameter 6)
    def construct_hexagon_incidence():
        # GF(4) arithmetic
        gf4_add = lambda a, b: a ^ b
        gf4_mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
        ]
        gf4_mul = lambda a, b: gf4_mul_table[a][b]

        # All points of PG(2,4): homogeneous triples (x,y,z) in GF(4)^3, not all zero
        # Up to scalar multiplication by nonzero elements {1,2,3}
        # Normalize: make first nonzero coordinate = 1
        points = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize
                    if x != 0:
                        inv_x = 1 if x == 1 else (2 if x == 2 else 3)
                        # inv: 1->1, 2->2, 3->3 in GF(4)
                        nx = 1
                        ny = gf4_mul(inv_x, y)
                        nz = gf4_mul(inv_x, z)
                    elif y != 0:
                        inv_y = 1 if y == 1 else (2 if y == 2 else 3)
                        nx = 0
                        ny = 1
                        nz = gf4_mul(inv_y, z)
                    else:
                        nx = 0
                        ny = 0
                        nz = 1
                    p = (nx, ny, nz)
                    if p not in points:
                        points.append(p)
        # Should have 21 points
        # points[i] for i in 0..20

        # Lines of PG(2,4): each line is a set of 5 collinear points
        # A line is defined by a linear equation ax + by + cz = 0 with (a,b,c) normalized
        # But we can enumerate lines by their point sets
        # Two points determine a line; collect all lines
        def line_through(p1, p2):
            # Find all points collinear with p1 and p2
            # p = p1 + t*(p2 - p1) for t in GF(4)
            dx = gf4_add(p1[0], gf4_add(0, p2[0]))  # p2[0] - p1[0] = p2[0] + p1[0] in GF(4)
            dy = gf4_add(p1[1], p2[1])
            dz = gf4_add(p1[2], p2[2])
            pts = set()
            for t in range(4):
                px = gf4_add(p1[0], gf4_mul(t, dx))
                py = gf4_add(p1[1], gf4_mul(t, dy))
                pz = gf4_add(p1[2], gf4_mul(t, dz))
                # Normalize
                if px != 0:
                    inv_px = 1 if px == 1 else (2 if px == 2 else 3)
                    npx = 1
                    npy = gf4_mul(inv_px, py)
                    npz = gf4_mul(inv_px, pz)
                elif py != 0:
                    inv_py = 1 if py == 1 else (2 if py == 2 else 3)
                    npx = 0
                    npy = 1
                    npz = gf4_mul(inv_py, pz)
                else:
                    npx = 0
                    npy = 0
                    npz = 1
                pts.add((npx, npy, npz))
            return frozenset(pts)

        # Enumerate all lines
        lines = set()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                line = line_through(points[i], points[j])
                lines.add(line)
        # Should have 21 lines... wait, PG(2,4) has (4^2+4+1)(4^2+4+1)/... 
        # Number of lines in PG(2,q) = q^2 + q + 1 = 21 for q=4
        # But each line has 5 points, and we need 105 "line vertices" for a 5-regular graph?
        # No: in the incidence graph of PG(2,4), points and lines are the two parts.
        # Each point is on q+1 = 5 lines, each line has q+1 = 5 points.
        # So it's 5-regular with 21+21 = 42 vertices and girth 6 (bipartite, triangle-free).
        # That's not what we want. We need girth 9 or 12.
        # The generalized hexagon of order (2,4) has 126 vertices in its incidence graph.
        # Let me use a different approach: just try random regular graphs near the Moore bound.

        return None

    # ---- Search for better graphs ----
    best_n = seed_n
    best_edges = seed_edges

    # Strategy 1: Try random regular graphs from Moore bound upward
    # For k=5, g=9: Moore bound = 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426
    # Seed gives ~1116 vertices. We search in between.
    
    search_start = max(moore_bound, 420)
    search_end = min(seed_n, search_start + 400)
    
    for n in range(search_start, search_end):
        if time.time() > deadline - 20:
            break
        
        # Adaptive trial count: more trials for smaller n (harder to find)
        remaining = deadline - time.time() - 20
        if remaining < 5:
            break
        
        # Estimate trials we can afford
        trials = max(1, min(10, int(remaining / 0.5)))
        
        found = False
        for trial in range(trials):
            if time.time() > deadline - 20:
                break
            
            # Try different generation methods
            if trial % 3 == 0:
                edges = generate_simple_regular(n, k)
            elif trial % 3 == 1:
                edges = generate_regular_via_matchmaking(n, k)
            else:
                edges = generate_simple_regular(n, k)
            
            if edges is None:
                continue

            adj = build_adj(edges, n)

            # Quick heuristic check with small sample
            if heuristic_girth_check(n, adj, g, sample_size=3):
                # Medium check
                if heuristic_girth_check(n, adj, g, sample_size=10):
                    # Full check
                    if exact_girth_check(n, adj, g):
                        return n, edges
                        found = True
                        break

        if found:
            break

    # Strategy 2: If we haven't found anything, try near the seed size
    # with more aggressive sampling
    if time.time() < deadline - 10:
        for n in range(max(seed_n - 30, search_start), seed_n):
            if time.time() > deadline - 5:
                break
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                edges = generate_simple_regular(n, k)
                if edges is None:
                    continue
                adj = build_adj(edges, n)
                if heuristic_girth_check(n, adj, g, sample_size=5):
                    if exact_girth_check(n, adj, g):
                        return n, edges

    return best_n, best_edges
