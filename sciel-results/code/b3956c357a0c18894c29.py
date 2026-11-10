import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    n_target = 320
    
    def generate_random_k_regular(n, k):
        while True:
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            edges = []
            valid = True
            edge_set = set()
            for i in range(0, n * k, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edge_set or (v, u) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
                edges.append((u, v))
            
            if valid:
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                return adj

    def has_short_cycle(adj, n, g):
        """Returns True if any cycle shorter than g exists."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return True
        return False

    def local_search(adj, n, k, g, time_limit):
        """
        Greedy local search to eliminate short cycles.
        """
        # Build edge list for random selection
        # We need to update this if we swap, but for simplicity, 
        # we can just pick random u, v from adj.
        
        while time.time() < time_limit:
            if not has_short_cycle(adj, n, g):
                return adj
            
            # Find a short cycle edge
            # We can just pick a random edge and check if it's in a short cycle?
            # Or find one short cycle and pick an edge from it.
            # Let's find one short cycle.
            short_cycle_edge = None
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                found = False
                while q and not found:
                    u = q.popleft()
                    if dist[u] >= g // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                short_cycle_edge = (u, v)
                                found = True
                                break
                if found:
                    break
            
            if not short_cycle_edge:
                return adj # Should not happen if has_short_cycle is true
                
            u, v = short_cycle_edge
            
            # Try to swap (u,v) with a random edge (w,z)
            swapped = False
            for _ in range(50):
                # Pick random edge
                w = random.randint(0, n-1)
                z = random.choice(list(adj[w]))
                
                # Check disjointness
                if w in (u, v) or z in (u, v):
                    continue
                
                # Check validity of new edges (u,w) and (v,z)
                if w in adj[u] or z in adj[v]:
                    continue
                
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(z)
                adj[z].remove(w)
                
                adj[u].add(w)
                adj[w].add(u)
                adj[v].add(z)
                adj[z].add(v)
                
                swapped = True
                break
            
            if not swapped:
                # Stuck, break to restart
                break
                
        return adj

    # Main loop
    best_adj = None
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        adj = generate_random_k_regular(n_target, k)
        adj = local_search(adj, n_target, k, g, deadline)
        
        if not has_short_cycle(adj, n_target, g):
            best_adj = adj
            break

    if best_adj:
        edges = []
        for u in range(n_target):
            for v in best_adj[u]:
                if u < v:
                    edges.append((u, v))
        return n_target, edges

    # Fallback to LUW
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
