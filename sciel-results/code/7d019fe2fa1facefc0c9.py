import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))

    def seed_construct(k, g):
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

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
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
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def try_circulant(k, g, n_max, time_limit):
        """
        Search for a circulant k-regular graph with girth >= g and n < n_max.
        Returns (n, edges) if found, else None.
        """
        if k > 10:
            return None
            
        def check_girth_circulant(n, S, g):
            dist = [-1] * n
            parent = [-1] * n
            dist[0] = 0
            q = deque([0])
            max_depth = (g - 1) // 2
            
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in S:
                    w = (u + v) % n
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        parent[w] = u
                        q.append(w)
                    else:
                        if w == parent[u]:
                            continue
                        cycle_len = dist[u] + dist[w] + 1
                        if cycle_len < g:
                            return False
            return True

        if k % 2 == 1:
            start_n = moore_bound
            if start_n % 2 != 0:
                start_n += 1
            
            for n in range(start_n, min(seed_n if 'seed_n' in locals() else 2000, start_n + 200)):
                if time.time() > time_limit:
                    break
                if n % 2 != 0:
                    continue
                
                half_n = n // 2
                candidates = list(range(1, half_n))
                
                if len(candidates) < 2:
                    continue
                
                for _ in range(50):
                    if time.time() > time_limit:
                        break
                    a, b = random.sample(candidates, 2)
                    S = [half_n, a, -a % n, b, -b % n]
                    if len(set(S)) != 5:
                        continue
                    
                    if check_girth_circulant(n, S, g):
                        edges = []
                        for i in range(n):
                            for s in S:
                                j = (i + s) % n
                                if i < j:
                                    edges.append((i, j))
                        return n, edges
        return None

    def construct_generalized_hexagon():
        """
        Construct the generalized hexagon H(4) of order (4,4).
        This is a bipartite 5-regular graph with 682 vertices and girth 12.
        
        Construction:
        - GF(4) = F2[x]/(x^2+x+1)
        - Points: 1-dim subspaces of {v in GF(4)^4 : v0^4+v1^4+v2^4+v3^4 = 0}
        - Lines: 1-dim subspaces of the dual space satisfying the same condition
        - Incidence: p incident to l iff sum(p_i * l_i) = 0 (bilinear form)
        """
        # GF(4) arithmetic
        gf4_add = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        gf4_mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        
        def gf4_add(a, b):
            return gf4_add[a][b]
        
        def gf4_mul(a, b):
            return gf4_mul[a][b]
        
        def gf4_pow(a, p):
            result = 1
            for _ in range(p):
                result = gf4_mul(result, a)
            return result
        
        def gf4_inv(a):
            if a == 0:
                return 0
            # In GF(4), a^3 = 1 for a != 0
            return gf4_pow(a, 2)
        
        def gf4_div(a, b):
            if b == 0:
                return 0
            return gf4_mul(a, gf4_inv(b))
        
        # Find all 1-dim subspaces of GF(4)^4 satisfying x0^4+x1^4+x2^4+x3^4 = 0
        def find_points():
            points = []
            seen = set()
            # Enumerate all vectors in GF(4)^4
            for x0 in range(4):
                for x1 in range(4):
                    for x2 in range(4):
                        for x3 in range(4):
                            # Check if not all zero
                            if x0 == 0 and x1 == 0 and x2 == 0 and x3 == 0:
                                continue
                            # Check if x0^4+x1^4+x2^4+x3^4 = 0
                            # In GF(4), a^4 = a for all a (Frobenius)
                            s = gf4_add(gf4_add(gf4_add(x0, x1), x2), x3)
                            if s != 0:
                                continue
                            # Find canonical representative: first nonzero coordinate should be 1
                            vec = [x0, x1, x2, x3]
                            for i in range(4):
                                if vec[i] != 0:
                                    inv = gf4_inv(vec[i])
                                    vec = [gf4_mul(v, inv) for v in vec]
                                    break
                            key = tuple(vec)
                            if key not in seen:
                                seen.add(key)
                                points.append(key)
            return points
        
        points = find_points()
        n_points = len(points)
        
        # For each point, find the 5 lines incident to it
        # A line is a 1-dim subspace {l0, l1, l2, l3} with l0^4+l1^4+l2^4+l3^4 = 0
        # Point p is incident to line l iff p0*l0 + p1*l1 + p2*l2 + p3*l3 = 0
        
        # We need to enumerate all lines (1-dim subspaces satisfying the condition)
        def find_lines():
            lines = []
            seen = set()
            for x0 in range(4):
                for x1 in range(4):
                    for x2 in range(4):
                        for x3 in range(4):
                            if x0 == 0 and x1 == 0 and x2 == 0 and x3 == 0:
                                continue
                            s = gf4_add(gf4_add(gf4_add(x0, x1), x2), x3)
                            if s != 0:
                                continue
                            vec = [x0, x1, x2, x3]
                            for i in range(4):
                                if vec[i] != 0:
                                    inv = gf4_inv(vec[i])
                                    vec = [gf4_mul(v, inv) for v in vec]
                                    break
                            key = tuple(vec)
                            if key not in seen:
                                seen.add(key)
                                lines.append(key)
            return lines
        
        lines = find_lines()
        n_lines = len(lines)
        
        # Build incidence: point i is incident to line j iff sum(p_k * l_k) = 0
        edges = []
        for i, p in enumerate(points):
            for j, l in enumerate(lines):
                s = 0
                for k in range(4):
                    s = gf4_add(s, gf4_mul(p[k], l[k]))
                if s == 0:
                    edges.append((i, n_points + j))
        
        return n_points + n_lines, edges

    # For k=5, g<=12, try the generalized hexagon H(4)
    if k == 5 and g <= 12:
        try:
            h4_n, h4_edges = construct_generalized_hexagon()
            # Verify it's 5-regular and has girth >= g
            adj = build_adj(h4_edges, h4_n)
            # Check regularity
            is_regular = all(len(adj[v]) == k for v in range(h4_n))
            if is_regular and exact_girth_check(h4_n, adj, g):
                return h4_n, h4_edges
        except Exception:
            pass

    seed_n, seed_edges = seed_construct(k, g)
    
    # Try circulant search
    result = try_circulant(k, g, seed_n, deadline)
    
    if result is not None:
        n, edges = result
        adj = build_adj(edges, n)
        if exact_girth_check(n, adj, g):
            return n, edges

    # Fallback to seed
    return seed_n, seed_edges
