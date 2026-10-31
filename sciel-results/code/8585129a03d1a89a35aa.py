import time
import random
import sys
from collections import deque

def construct(k, g):
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Fallback for non-hard cases: simple LUW
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

    if k != 3 or g != 16:
        return luw_construct(k, g)

    # Specific strategy for k=3, g=16
    # The task asks for minimal n. Known bounds: 512 <= n <= 936 (from prompt, though 936 is very recent).
    # LUW gives ~354k or more. We need a much smaller graph.
    # Randomized greedy construction with backtracking or restarts.
    
    # Since 120s is short for finding a 500-vertex girth-16 graph via pure random search without specific algebraic structure,
    # and the "inspiration" mentions optimized circulant search, let's try a random circulant or random lift approach.
    # However, the prompt explicitly suggests "random lift of Petersen graph... then local search".
    # Petersen graph has girth 5. A lift of degree 3.
    # A 3-lift of Petersen (10 vertices) has 30 vertices. Girth might be small.
    # To get girth 16, we need a much larger lift or a different base.
    
    # Actually, the best known small graphs are often algebraic.
    # Let's try a "Random Regular Graph" approach with girth checking, but restart if girth is too small.
    # This is unlikely to find girth 16 with n < 1000 in 120s from scratch without guidance.
    
    # Alternative: Use the known smallest graph structure if we can hardcode it? No, must construct.
    
    # Let's try a random 3-regular graph generator with a "girth filter" during generation.
    # This is the "Configuration Model with Girth Constraint" or similar.
    
    def random_3_regular_with_girth_check(n, g, seed=42):
        random.seed(seed)
        # Generate random 3-regular graph
        # Use configuration model
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        random.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        edges = set()
        for i in range(0, 2 * n * 3, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None, None # Self loop, invalid
            if (u, v) in edges or (v, u) in edges:
                return None, None # Multi-edge, invalid
            edges.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
            
        # Check girth
        # If girth >= g, return
        # This check is O(n * g^2) roughly, which is fine for n=1000.
        if check_girth(adj, g):
            return n, list(edges)
        return None, None

    def check_girth(adj, g):
        n = len(adj)
        # BFS from each vertex to find shortest cycle
        # Optimization: Only need to check up to depth g//2
        max_depth = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
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

    # Try random graphs of increasing size?
    # The lower bound is 512. The upper bound is 936.
    # Let's try to find a graph with n in [512, 1000].
    # Random search for girth 16 is extremely hard.
    
    # The "inspiration" mentioned "optimized circulant search".
    # Let's try random circulant graphs.
    # A circulant graph C_n(S) is 3-regular if |S|=3 and S is symmetric (s in S -> -s in S).
    # For n even, S = {a, -a, b, -b, ...} wait, degree 3 is odd, so n must be even and one element is n/2?
    # If n is even, S can contain n/2.
    # Let's try random sets S of size 3 for n in range(512, 1000).
    
    def check_circulant_girth(n, S, g):
        # Check if circulant graph with steps S has girth >= g
        # Cycle corresponds to sum of steps = 0 mod n
        # Shortest cycle is min non-zero sum of +/- steps
        # This is a subset sum problem, but for small g we can do BFS on the Cayley graph
        adj = [[] for _ in range(n)]
        for u in range(n):
            for s in S:
                v = (u + s) % n
                if v != u:
                    adj[u].append(v)
                # Ensure simple graph: if s == -s (i.e., 2s=0 mod n), it's a loop? No, 3-regular means 3 distinct neighbors.
                # If n is even and s=n/2, then u+s = u-n/2.
                # We need distinct neighbors.
        
        # Check degrees
        for u in range(n):
            if len(set(adj[u])) != 3:
                return False
        
        # Check girth
        return check_girth(adj, g)

    best_n = float('inf')
    best_edges = []
    
    # Try random circulants
    # n must be even for 3-regular circulant?
    # If n is odd, S must not contain n/2. But sum of degrees = 3n must be even, so n must be even.
    
    for n_try in range(512, 1000, 2):
        if time.time() > deadline:
            break
            
        # Try a few random S sets
        for _ in range(50):
            if time.time() > deadline:
                break
            # Generate random S
            # One step must be n/2 if we want to ensure symmetry? No.
            # S = {a, b, c}. -S = S.
            # So if a in S, -a in S.
            # If a != -a, then S = {a, -a, b, -b, ...} wait, size 3.
            # If a != -a, then a and -a are two distinct elements. The third must be c such that -c=c, i.e., 2c=0 mod n.
            # So c = n/2.
            # Thus S = {a, -a, n/2}.
            # a must be in 1..n/2-1.
            
            a = random.randint(1, n_try // 2 - 1)
            S = [a, n_try - a, n_try // 2]
            
            if check_circulant_girth(n_try, S, g):
                # Construct edges
                edges = []
                for u in range(n_try):
                    for s in S:
                        v = (u + s) % n_try
                        if u < v:
                            edges.append((u, v))
                        # Note: n/2 step connects u and u+n/2.
                        # a and -a connect u and u+a, u and u-a.
                if len(edges) == 3 * n_try // 2:
                    if n_try < best_n:
                        best_n = n_try
                        best_edges = edges
                        # If we found one, maybe keep searching for smaller?
                        # But 512 is the lower bound. If we find 512, we can stop.
                        if n_try == 512:
                            return n_try, edges

    if best_n != float('inf'):
        return best_n, best_edges

    # Fallback to LUW if no small circulant found
    return luw_construct(k, g)
