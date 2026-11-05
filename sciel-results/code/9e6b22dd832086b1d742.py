import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def generate_random_k_regular(n, k):
        while True:
            stubs = []
            for i in range(n):
                stubs.extend([i, i])
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, n*k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))
            if valid:
                adj = {i: [] for i in range(n)}
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                return adj, list(edges)

    def find_short_cycle(adj, n, g):
        # Returns a list of vertices in a short cycle, or None
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = deque([(start, 0)])
            visited[start] = 0
            while queue:
                u, depth = queue.popleft()
                if depth >= g // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = depth + 1
                        parent[v] = u
                        queue.append((v, depth + 1))
                    elif v != parent[u]:
                        cycle_len = depth + 1 - visited[v]
                        if cycle_len < g:
                            cycle = []
                            curr = u
                            while curr != v:
                                cycle.append(curr)
                                curr = parent[curr]
                            cycle.append(v)
                            return cycle
        return None

    def break_cycle(adj, edges, cycle, n, k):
        # Try to break the cycle by swapping an edge in the cycle with a random edge
        # Returns True if successful
        cycle_edges = set()
        for i in range(len(cycle)-1):
            u, v = cycle[i], cycle[i+1]
            cycle_edges.add((min(u,v), max(u,v)))
            
        # Try a few random swaps
        for _ in range(20):
            e1 = random.choice(list(cycle_edges))
            u1, v1 = e1
            e2 = random.choice(edges)
            u2, v2 = e2
            if e2 == e1 or u1 in (u2, v2) or v1 in (u2, v2):
                continue
            
            # Check if swap is valid
            # Remove e1, e2. Add (u1, u2), (v1, v2)
            # Check for loops/multi-edges
            new_edges = set(edges)
            new_edges.remove(e1)
            new_edges.remove(e2)
            
            e_new1 = (min(u1, u2), max(u1, u2))
            e_new2 = (min(v1, v2), max(v1, v2))
            
            if u1 == u2 or v1 == v2:
                continue
            if e_new1 in new_edges or e_new2 in new_edges:
                continue
            if e_new1 == e_new2:
                continue
                
            # Apply swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            
            edges.remove(e1)
            edges.remove(e2)
            edges.append(e_new1)
            edges.append(e_new2)
            
            return True
        return False

    # Try to find a graph
    for n in [320, 400, 500, 600, 800, 1000]:
        if time.time() > deadline:
            break
            
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            adj, edges = generate_random_k_regular(n, k)
            
            # Try to break all short cycles
            success = False
            for _ in range(1000): # Max iterations
                if time.time() > deadline:
                    break
                cycle = find_short_cycle(adj, n, g)
                if cycle is None:
                    success = True
                    break
                if not break_cycle(adj, edges, cycle, n, k):
                    break
                    
            if success:
                return n, edges

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
    index, order, edges_out = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_out.append((index[vertex], index[other]))
    return len(order), edges_out
