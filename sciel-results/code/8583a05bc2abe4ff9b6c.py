import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin from 150s kill

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
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
        """Non-backtracking BFS from every vertex to certify girth >= g."""
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

    def heuristic_girth_check(n, adj, g, sample_size=10):
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
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def generate_simple_regular(n, k):
        """Random regular graph via stub matching with rejection."""
        for attempt in range(100):
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

    # ---- Strategy: search for smaller graphs with higher girth ----
    best_n = seed_n
    best_edges = seed_edges

    # For k=5, g=9: Moore bound is 1+5*(1+4+16) = 1+5*21 = 106
    # The seed LZW construction gives some n. Try to find smaller valid graphs.
    # Random regular graphs near the Moore bound are very unlikely to have girth 9,
    # but we can try a few n values with many trials.

    # Also try a smarter construction: circulant-based or voltage graph approach
    # For k=5, g=9, try n from moore_bound upward

    # First, let's see what the seed gives
    # For k=5, g=9: d = max(3, 4 if 4%2==0 else 5) = max(3, 5) = 5
    # q=5, d=5, so vertices are (side, (None, y1, y2, y3, y4, y5)) -> 2*5^5 = 6250? 
    # Actually the LZW construction: side in {0,1}, x = (None, x1..xd) with x1..xd in Z_q
    # So 2 * q^d = 2 * 5^5 = 6250 vertices. That's way above 1116.
    # Wait, let me re-examine. The neighbours function iterates over 'first' in range(q),
    # and the vertex has coordinates y[1..d]. So each vertex is (side, (None, y1,...,yd)).
    # The BFS from start explores all reachable vertices. For the LZW construction with
    # q=5, d=5, the graph should have 2*q^d = 6250 vertices if it's connected and covers all.
    # But actually the construction may not be connected or may have fewer vertices.

    # Let me just use the seed result and try to improve.

    # Try random regular graphs for n in a range
    # For girth 9, we need no cycles of length 3,5,7. 
    # Random 5-regular graphs on n vertices: expected number of short cycles can be estimated.
    # For large n, random regular graphs have girth ~ (2/3)*log_{k-1}(n) approximately.
    # For girth 9, we need n to be large enough. The Moore bound is 106.
    # For n=106, a random 5-regular graph will almost certainly have triangles.
    # We need to go much larger.

    # Let's try a different approach: use the seed as baseline, and try to find
    # any valid graph with n < seed_n.

    # Try n values from moore_bound to seed_n-1
    # But be time-aware.

    time_budget = deadline - time.time()
    
    # Estimate how many trials we can do
    # Each trial: generate (O(n*k)) + heuristic check (O(n * sample * degree * depth))
    # For n ~ 500, this is fast. For n ~ 5000, slower.
    
    # Let's try a focused search
    # For k=5, g=9, known best is 1116. Try n from ~100 to seed_n.
    
    # Actually, let me reconsider. The random approach won't find girth-9 graphs
    # at small n. We need a structured approach.
    
    # Alternative: try to use the LZW construction with different parameters
    # or try a different algebraic construction.
    
    # For now, let's do a limited random search and return the best.
    
    max_search_n = min(seed_n, moore_bound + 2000)
    
    # Try a few n values with random regular graphs
    # Focus on n values where random graphs might have girth 9
    # For 5-regular, girth 9 requires roughly n >= ~1000 (heuristically)
    
    trial_n_values = []
    # Try some spread-out n values
    for n in range(max(moore_bound, 100), max_search_n, 50):
        trial_n_values.append(n)
    
    random.shuffle(trial_n_values)
    
    for n in trial_n_values:
        if time.time() > deadline - 10:
            break
        if n >= best_n:
            continue
        
        # Try a few random graphs at this n
        n_trials = max(1, int((deadline - 10 - time.time()) / 0.5))
        n_trials = min(n_trials, 20)
        
        for _ in range(n_trials):
            if time.time() > deadline - 10:
                break
            edges = generate_simple_regular(n, k)
            if edges is None:
                continue
            adj = build_adj(edges, n)
            if heuristic_girth_check(n, adj, g, sample_size=5):
                if exact_girth_check(n, adj, g):
                    best_n = n
                    best_edges = edges
                    break
    
    return best_n, best_edges
