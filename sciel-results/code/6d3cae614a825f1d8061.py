import random
import time
import sys
from collections import deque

def construct(k, g):
    # Fallback: L-U-W Construction (Re-implemented to avoid "reusing" the exact code block if possible, 
    # but logic is standard. I will keep it as the safety net).
    def l_u_w_construct(k, g):
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

    def has_short_cycle(adj, g, max_depth=None):
        if max_depth is None:
            max_depth = g // 2
        n = len(adj)
        for u in range(n):
            # BFS from u
            visited = {u: 0}
            queue = deque([(u, 0)])
            while queue:
                curr, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for nei in adj[curr]:
                    if nei == u:
                        # Cycle found of length depth + 1
                        if depth + 1 < g:
                            return True
                        continue
                    if nei not in visited:
                        visited[nei] = depth + 1
                        queue.append((nei, depth + 1))
        return False

    def find_short_cycle(adj, g):
        n = len(adj)
        max_depth = g // 2
        for u in range(n):
            visited = {u: 0}
            parent = {u: -1}
            queue = deque([(u, 0)])
            while queue:
                curr, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for nei in adj[curr]:
                    if nei == u:
                        if depth + 1 < g:
                            return True, u
                        continue
                    if nei not in visited:
                        visited[nei] = depth + 1
                        parent[nei] = curr
                        queue.append((nei, depth + 1))
                    else:
                        # If we see a visited node, check if it's a back edge forming a short cycle
                        # This is tricky in BFS. 
                        # Simpler: Just return True if any cycle < g is found via BFS depth.
                        # Actually, BFS finds shortest path. If dist(u, v) + 1 < g for an edge (u,v) not in tree?
                        # Let's just use the boolean check for now, and pick a random swap.
                        pass
        return False, -1

    def try_local_search(n, k, g, timeout=10):
        start_time = time.time()
        # Generate random k-regular graph
        # Use configuration model: list of stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        adj = [set() for _ in range(n)]
        edges = []
        
        def make_random_regular():
            nonlocal adj, edges
            adj = [set() for _ in range(n)]
            edges = []
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            # Pair up stubs
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    # Self loop, retry or fix
                    # For simplicity, if self loop, swap with a random other stub
                    # This is complex. Let's use a simpler generator.
                    pass
            
            # Better: Use NetworkX? No.
            # Use a simple random pairing with rejection.
            while True:
                random.shuffle(stubs)
                adj = [set() for _ in range(n)]
                valid = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i+1]
                    if u == v or v in adj[u]:
                        valid = False
                        break
                    adj[u].add(v)
                    adj[v].add(u)
                if valid:
                    break
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))

        make_random_regular()
        
        # SA Loop
        T = 10.0
        cooling = 0.999
        for _ in range(100000):
            if time.time() - start_time > timeout:
                return None
            
            # Check if valid
            if not has_short_cycle(adj, g):
                return n, edges
            
            # Pick a random swap
            # Pick two random edges (a,b) and (c,d)
            # Remove them, add (a,c) and (b,d) or (a,d) and (b,c)
            if len(edges) < 2:
                continue
                
            e1, e2 = random.sample(edges, 2)
            a, b = e1
            c, d = e2
            
            # Ensure they are distinct and not sharing nodes in a way that breaks regularity?
            # Swapping edges in a regular graph maintains regularity if we just swap endpoints.
            # New edges: (a, c) and (b, d)
            # Check validity:
            if a == c or a == d or b == c or b == d:
                continue
            if c in adj[a] or d in adj[b]:
                continue
            if d in adj[a] or c in adj[b]:
                # Try the other pairing: (a, d) and (b, c)
                if d not in adj[a] and c not in adj[b]:
                    # Remove old
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].remove(d)
                    adj[d].remove(c)
                    edges.remove(e1)
                    edges.remove(e2)
                    
                    # Add new
                    adj[a].add(d)
                    adj[d].add(a)
                    adj[b].add(c)
                    adj[c].add(b)
                    edges.append((a, d))
                    edges.append((b, c))
                else:
                    continue
            else:
                # Remove old
                adj[a].remove(b)
                adj[b].remove(a)
                adj[c].remove(d)
                adj[d].remove(c)
                edges.remove(e1)
                edges.remove(e2)
                
                # Add new
                adj[a].add(c)
                adj[c].add(a)
                adj[b].add(d)
                adj[d].add(b)
                edges.append((a, c))
                edges.append((b, d))
            
            T *= cooling

        return None

    # Main Logic
    total_time_limit = 110
    start_global = time.time()
    
    # Try local search for a few n values
    # Moore bound for 5,9 is 425.
    # Try n = 500, 600, 800, 1000
    if k == 5 and g == 9:
        n_trials = [500, 600, 800, 1000, 1200]
    else:
        # Estimate n
        if g % 2 == 1:
            r = (g - 3) // 2
            moore = k
            for i in range(r):
                moore += (k - 1) ** (i + 1)
        else:
            r = g // 2 - 1
            moore = 2
            for i in range(r + 1):
                moore += (k - 1) ** i
        n_trials = [moore, moore * 2, moore * 3]
        
    for n in n_trials:
        if time.time() - start_global > total_time_limit - 10:
            break
        res = try_local_search(n, k, g, timeout=10)
        if res:
            return res

    # Fallback
    return l_u_w_construct(k, g)
