import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # 5s margin

    # Moore Bound Lower Bound
    if g % 2 == 0:
        r = g // 2
        lb = 1 + k * sum([(k-1)**i for i in range(r)])
    else:
        r = (g - 1) // 2
        lb = 2 * sum([(k-1)**i for i in range(r)])
        
    # Fallback: LUW Construction
    def luw_construct():
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
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

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

    # Girth Checker (BFS from each node, optimized for early exit)
    def get_girth(adj):
        n = len(adj)
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        # Cycle found
                        # Length is dist[u] + dist[v] + 1
                        # Note: This finds the shortest cycle passing through u? 
                        # Actually, BFS from start finds shortest path to all.
                        # The cycle length is dist[u] + dist[v] + 1.
                        # This is a valid cycle.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < g:
                            return min_girth
        return min_girth if min_girth != float('inf') else 2 * n

    # Random K-Regular Graph Generator (Configuration Model + Repair)
    def gen_random_k_reg(n, k):
        # Ensure n*k is even
        if (n * k) % 2 != 0:
            return None
            
        for attempt in range(50):
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                if all(len(a) == k for a in adj):
                    return adj
        return None

    # Local Search: 2-Switches to improve girth
    def optimize_girth(adj, k, g, time_limit):
        n = len(adj)
        current_girth = get_girth(adj)
        if current_girth >= g:
            return adj
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            # Pick a random edge (u, v)
            u = random.randint(0, n - 1)
            if not adj[u]: continue
            v = random.choice(adj[u])
            
            # Pick another random edge (w, x)
            w = random.randint(0, n - 1)
            if w == u or w == v:
                continue
            if not adj[w]: continue
            x = random.choice(adj[w])
            if x == u or x == v:
                continue
            
            # Check if (u, w) and (v, x) are already edges or loops
            if w in adj[u] or x in adj[v]:
                continue
            if u == w or v == x:
                continue
            if x in adj[u] or w in adj[v]:
                continue

            # Perform 2-switch
            # Remove (u, v) and (w, x)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            # Add (u, w) and (v, x)
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            new_girth = get_girth(adj)
            if new_girth > current_girth:
                current_girth = new_girth
                if current_girth >= g:
                    return adj
            # If not improved, we could revert, but for simplicity we keep it 
            # as it might lead to better states later (random walk).
            # However, for girth, we want to AVOID short cycles.
            # If new_girth < current_girth, we should probably revert.
            else:
                # Revert
                adj[u].remove(w)
                adj[w].remove(u)
                adj[v].remove(x)
                adj[x].remove(v)
                
                adj[u].append(v)
                adj[v].append(u)
                adj[w].append(x)
                adj[x].append(w)

        return adj

    # Main Search Logic
    # Try small n first
    max_n_to_try = 500 # Limit to keep it fast
    if lb <= max_n_to_try:
        # Try n from lb to max_n_to_try
        # Step size 1 to ensure parity if needed? k=5, n must be even? 5*n even => n even.
        start_n = lb if lb % 2 == 0 else lb + 1
        for n in range(start_n, max_n_to_try + 1, 2):
            if time.time() > deadline - 20:
                break
                
            adj = gen_random_k_reg(n, k)
            if not adj:
                continue
                
            # Check girth
            girth = get_girth(adj)
            if girth >= g:
                # Convert adj to edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            
            # If not good, try to optimize a bit
            # Only do this for a few n to save time
            if n < 200:
                adj = optimize_girth(adj, k, g, 2.0)
                girth = get_girth(adj)
                if girth >= g:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges

    # Fallback to LUW
    return luw_construct()
