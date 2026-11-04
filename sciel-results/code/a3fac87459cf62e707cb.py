import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before the 150s kill

    # --- Step 1: Guaranteed valid baseline from the seed algebraic construction ---
    # This matches the known best graph for n(3, 16) which is 936 vertices.
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

    baseline_n = len(order)
    baseline_edges = edges

    # For the specific case k=3, g=16, the baseline is the known best (936 vertices).
    # Return immediately to guarantee a valid result with no wasted time.
    if k == 3 and g == 16 and baseline_n == 936:
        return baseline_n, baseline_edges

    best_n = baseline_n
    best_edges = baseline_edges

    if k == 3 and g <= 10 and time.time() < deadline - 10.0:
        # Moore bound for k=3, girth g: M = 2 * sum_{i=0}^{g/2-1} (k-1)^i
        moore_bound = 2 * sum(2 ** i for i in range(g // 2 - 1)) if g % 2 == 0 else 2 * sum(2 ** i for i in range((g - 1) // 2 - 1)) + 1
        if moore_bound % 2 != 0:
            moore_bound += 1
        
        max_search_n = min(baseline_n, moore_bound + 40)
        
        def get_adj(edges, n):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        def check_girth_fast(adj, n, g):
            """
            Check if graph has girth >= g using non-backtracking walks.
            Girth < g iff there exists a closed non-backtracking walk of length L < g.
            We check for closed NB walks of length L = g-1 (if g is even, check odd lengths too? 
            Actually, if girth is g, there is no closed NB walk of length < g.
            So we just need to check if any diagonal of A_nb^L is non-zero for L < g.
            For efficiency, we can check the largest L first, or just check L = g-1 and L = g-2.
            If there is a cycle of length c < g, it corresponds to a closed NB walk of length c.
            So if we find a closed NB walk of length L < g, girth < g.
            Conversely, if no closed NB walk of length < g exists, girth >= g.
            We check L from g-1 down to 3.
            """
            # Build non-backtracking adjacency matrix
            # Vertices are directed edges (u, v)
            # Let M be the matrix where M[(u,v), (v,w)] = 1 if w != u
            # Number of directed edges is 2n
            # Map (u, v) to index
            idx = {}
            count = 0
            for u in range(n):
                for v in adj[u]:
                    idx[(u, v)] = count
                    count += 1
            size = count
            # A_nb is a sparse matrix of size 2n x 2n
            # We can use numpy for dense if 2n is small (e.g., < 200)
            if size <= 400:
                A = np.zeros((size, size), dtype=np.int8)
                for u in range(n):
                    for v in adj[u]:
                        for w in adj[v]:
                            if w != u:
                                A[idx[(u, v)], idx[(v, w)]] = 1
                # Check powers
                for L in range(g - 1, 2, -1):
                    # A^L diagonal
                    # We can compute A^L iteratively
                    if L == 1:
                        continue # no loops in simple graph
                    # Start with A
                    cur = A.copy()
                    for _ in range(L - 1):
                        cur = cur @ A
                    # Check diagonal
                    if np.any(np.diag(cur) > 0):
                        return False
                return True
            else:
                # Fallback to BFS if matrix too large
                return check_girth_bfs(adj, n, g)

        def check_girth_bfs(adj, n, g):
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    if dist[u] >= g // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                        elif v != start:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
            return True

        def generate_random_3_regular(n):
            if n % 2 != 0:
                return None
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges:
                    valid = False
                    break
                edges.add(key)
            if valid:
                return list(edges)
            return None

        def swap_edges(adj, edges_set, u, v, x, y):
            if u == x or u == y or v == x or v == y:
                return False
            if (min(u, x), max(u, x)) in edges_set:
                return False
            if (min(v, y), max(v, y)) in edges_set:
                return False
            
            edges_set.remove((min(u, v), max(u, v)))
            edges_set.remove((min(x, y), max(x, y)))
            edges_set.add((min(u, x), max(u, x)))
            edges_set.add((min(v, y), max(v, y)))
            
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            return True

        for n_try in range(moore_bound, max_search_n, 2):
            if time.time() > deadline - 5.0:
                break
            
            found = False
            for attempt in range(5):
                if time.time() > deadline - 5.0:
                    break
                
                edges = generate_random_3_regular(n_try)
                if edges is None:
                    continue
                
                edges_set = set(edges)
                adj = get_adj(edges, n_try)
                
                if check_girth_fast(adj, n_try, g):
                    best_n = n_try
                    best_edges = list(edges_set)
                    found = True
                    break
                
                for step in range(1000):
                    if time.time() > deadline - 5.0:
                        break
                    
                    e1 = random.choice(list(edges_set))
                    e2 = random.choice(list(edges_set))
                    if e1 == e2:
                        continue
                    
                    u, v = e1
                    x, y = e2
                    
                    if swap_edges(adj, edges_set, u, v, x, y):
                        if check_girth_fast(adj, n_try, g):
                            best_n = n_try
                            best_edges = list(edges_set)
                            found = True
                            break
                    else:
                        pass
                
                if found:
                    break

            if found:
                break

    return best_n, best_edges
