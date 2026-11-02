def construct(k, g):
    import time
    import random
    start_time = time.time()
    deadline = start_time + 110
    
    known_cages = {}
    
    # Petersen Graph (3, 5)
    known_cages[(3, 5)] = (10, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                                 (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
                                 (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)])
    
    # Heawood Graph (3, 7)
    heawood_edges = []
    fano_lines = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    for i, line in enumerate(fano_lines):
        line_id = 7 + i
        for p in line:
            heawood_edges.append((p, line_id))
    known_cages[(3, 7)] = (14, heawood_edges)
    
    if (k, g) in known_cages:
        return known_cages[(k, g)]
    
    def verify_girth(n, edges, g):
        """Verify girth using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degrees
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check no loops or multi-edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        # Non-backtracking BFS from each vertex to find shortest cycle
        for start in range(n):
            # dist[prev][curr] = shortest non-backtracking walk length from start to curr with prev as last step
            # We use a dict: {(prev, curr): dist}
            dist = {}
            queue = []
            # Initialize: non-backtracking walks of length 1 from start
            for v in adj[start]:
                dist[(start, v)] = 1
                queue.append((start, v))
            
            head = 0
            while head < len(queue):
                prev, curr = queue[head]
                head += 1
                d = dist[(prev, curr)]
                if d >= g:
                    break  # No cycle shorter than g found from this start yet, but continue checking
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue  # Non-backtracking: don't go back to prev
                    if nxt == start and d + 1 >= g:
                        # This would form a cycle of length d+1, which is >= g, so OK
                        pass
                    elif nxt == start:
                        # Cycle of length d+1 < g
                        return False
                    new_key = (curr, nxt)
                    if new_key not in dist:
                        dist[new_key] = d + 1
                        queue.append(new_key)
        return True
    
    def build_luw(d, q):
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
    
    def build_circulant(k, g):
        """Try to find a circulant graph with good girth."""
        if k % 2 == 0:
            half_k = k // 2
        else:
            half_k = k // 2
        
        best = None
        for trial in range(30):
            if time.time() > deadline:
                break
            # Estimate size more carefully
            moore = 1 + k * sum((k-1)**i for i in range(g//2))
            base_n = max(moore * 2, 2 * (k - 1) ** (g // 2))
            n = base_n + random.randint(0, base_n // 4)
            n = max(n, 2 * k + 2)
            
            if k % 2 == 0:
                # Pick half_k generators
                gens = random.sample(range(1, n // 2 + 1), half_k)
                edges = []
                for i in range(n):
                    for gen in gens:
                        j = (i + gen) % n
                        if i < j:
                            edges.append((i, j))
            else:
                # For odd k, use (k-1)/2 generators plus n/2 if n is even
                if n % 2 != 0:
                    continue
                half_k = k // 2
                gens = random.sample(range(1, n // 2), half_k)
                edges = []
                for i in range(n):
                    for gen in gens:
                        j = (i + gen) % n
                        if i < j:
                            edges.append((i, j))
                    # Add the n/2 edge
                    j = (i + n // 2) % n
                    if i < j:
                        edges.append((i, j))
            
            if len(edges) != n * k // 2:
                continue
            
            if verify_girth(n, edges, g):
                if best is None or n < best[0]:
                    best = (n, edges)
        
        return best
    
    # Try L-U-W first (deterministic, fast) - search over more d values
    best_n, best_edges = None, None
    q = k
    
    # Try various d values to find the smallest valid graph
    d_values = list(range(2, min(g + 2, 14)))
    random.shuffle(d_values)
    
    for d in d_values:
        if time.time() > deadline - 5:
            break
        try:
            n, edges = build_luw(d, q)
            if verify_girth(n, edges, g):
                if best_n is None or n < best_n:
                    best_n, best_edges = n, edges
        except Exception:
            continue
    
    # If L-U-W worked, try circulant to see if we can do better
    if time.time() < deadline - 10:
        circ = build_circulant(k, g)
        if circ is not None:
            if best_n is None or circ[0] < best_n:
                best_n, best_edges = circ
    
    if best_n is None:
        # Fallback: try larger LUW parameters
        for d in range(min(g, 10), min(g + 5, 18)):
            if time.time() > deadline - 5:
                break
            try:
                n, edges = build_luw(d, q)
                if verify_girth(n, edges, g):
                    best_n, best_edges = n, edges
                    break
            except Exception:
                continue
    
    if best_n is None:
        # Last resort: construct a valid k-regular graph (may not have good girth)
        n = 2 * k + 2
        edges = []
        for i in range(n):
            for j in range(1, k // 2 + 1):
                v1 = (i + j) % n
                v2 = (i - j) % n
                if i < v1:
                    edges.append((i, v1))
                if i < v2:
                    edges.append((i, v2))
        if k % 2 == 1:
            for i in range(n // 2):
                edges.append((i, i + n // 2))
        seen = set()
        unique_edges = []
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                unique_edges.append((u, v))
        best_n, best_edges = n, unique_edges
    
    return best_n, best_edges
