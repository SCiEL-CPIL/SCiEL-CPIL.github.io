import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback to seed if we fail
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

    def check_girth(adj, n, g):
        """Check if graph has girth >= g using non-backtracking BFS from each vertex.
        Returns True if girth >= g, False otherwise."""
        for start in range(n):
            # BFS tracking (node, prev) pairs, distance
            visited = {}  # (node, prev) -> dist
            queue = []
            for nb in adj[start]:
                visited[(nb, start)] = 1
                queue.append((nb, start, 1))
            while queue:
                u, prev_u, dist = queue.pop(0)
                if dist >= g // 2:
                    continue
                for v in adj[u]:
                    if v == prev_u:
                        continue
                    if v == start:
                        if dist + 1 < g:
                            return False
                    elif (v, u) in visited:
                        if dist + 1 < g:
                            return False
                    else:
                        visited[(v, u)] = dist + 1
                        queue.append((v, u, dist + 1))
        return True

    def build_petersen_lift(m, rng):
        """Build a random m-lift of the Petersen graph.
        Petersen has 10 vertices, 15 edges.
        The lift has 10*m vertices.
        Each edge (u,v) in Petersen connects u_i to v_{i+sigma} for random permutation sigma.
        Returns adjacency list or None if invalid."""
        # Petersen graph edges
        petersen_edges = []
        # Outer pentagon: 0-1-2-3-4-0
        for i in range(5):
            petersen_edges.append((i, (i + 1) % 5))
        # Inner star: 5-7-9-6-8-5 (pentagram)
        inner = [5, 6, 7, 8, 9]
        for i in range(5):
            petersen_edges.append((inner[i], inner[(i + 2) % 5]))
        # Spokes: i to i+5
        for i in range(5):
            petersen_edges.append((i, i + 5))

        # For each Petersen edge, assign a random permutation of Z_m
        # Vertex (p, i) where p in 0..9, i in 0..m-1
        # Edge between (p, i) and (q, i + perm[p,q][i]) where perm is the permutation for edge (p,q)
        
        # Assign random permutations to each Petersen edge
        edge_perms = {}
        for (p, q) in petersen_edges:
            perm = list(range(m))
            rng.shuffle(perm)
            edge_perms[(p, q)] = perm
            edge_perms[(q, p)] = [(m - 1 - i) for i in range(m)]  # inverse: if p_i -> q_{i+sigma}, then q_j -> p_{j-sigma}
            # Actually let's be more careful
            # If edge (p,q) maps p_i to q_{sigma(i)}, then q_j connects to p_{sigma^{-1}(j)}
            inv_perm = [0] * m
            for i in range(m):
                inv_perm[perm[i]] = i
            edge_perms[(q, p)] = inv_perm

        n = 10 * m
        adj = [[] for _ in range(n)]
        
        for (p, q) in petersen_edges:
            perm = edge_perms[(p, q)]
            for i in range(m):
                u = p * m + i
                v = q * m + perm[i]
                adj[u].append(v)
                adj[v].append(u)
        
        # Verify it's 3-regular and simple
        for i in range(n):
            if len(adj[i]) != 3:
                return None
            if len(set(adj[i])) != 3:
                return None
            for nb in adj[i]:
                if nb < 0 or nb >= n:
                    return None
                if i not in adj[nb]:
                    return None
        
        return adj

    # Main search: try random Petersen lifts with increasing m
    rng = random.Random(42)
    
    # For girth g, we need m large enough. Girth of random lift ~ c * log(m)
    # For g=16, m around 100-500 should work
    # Try m values, for each try several random lifts
    
    best_result = None
    
    # Start with smaller m values and increase
    m_values = list(range(50, 501, 10))
    # Also add some specific values
    m_values.extend([60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 400, 500])
    m_values = sorted(set(m_values))
    
    for m in m_values:
        if time.time() > deadline - 10:
            break
        
        n = 10 * m
        
        # Try several random lifts for this m
        max_tries = 20 if m < 200 else 5
        for attempt in range(max_tries):
            if time.time() > deadline - 10:
                break
            
            adj = build_petersen_lift(m, rng)
            if adj is None:
                continue
            
            # Check girth
            if check_girth(adj, n, g):
                # Valid! Extract edges
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # If we didn't find one, try a few more with larger m
    for m in range(500, 1001, 50):
        if time.time() > deadline - 5:
            break
        n = 10 * m
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            adj = build_petersen_lift(m, rng)
            if adj is None:
                continue
            if check_girth(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Fallback to seed
    return seed_construct(k, g)
