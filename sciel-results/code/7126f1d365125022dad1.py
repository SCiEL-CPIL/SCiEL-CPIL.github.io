import time
import random
import math
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # If k=4 and g=10, try to find the 320-vertex cage via search
    if k == 4 and g == 10:
        deadline = time.time() + 110  # Safety margin
        
        def random_4_regular(n):
            # Configuration model
            if n % 2 != 0:
                return None
            
            def random_perfect_matching(n):
                if n % 2 != 0: return None
                items = list(range(n))
                random.shuffle(items)
                match = {}
                for i in range(0, n, 2):
                    u, v = items[i], items[i+1]
                    match[u] = v
                    match[v] = u
                return match

            m1 = random_perfect_matching(n)
            if m1 is None: return None
            m2 = random_perfect_matching(n)
            if m2 is None: return None
            
            adj = [[] for _ in range(n)]
            for u in range(n):
                v1 = m1[u]
                v2 = m2[u]
                if v1 == v2:
                    return None # Collision, retry
                adj[u].append(v1)
                adj[v1].append(u)
                adj[u].append(v2)
                adj[v2].append(u)
                
            # Check for multi-edges
            for u in range(n):
                if len(set(adj[u])) < 4:
                    return None
            return adj

        def get_girth(adj):
            n = len(adj)
            best = float('inf')
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= best // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append(v)
                        elif dist[v] < dist[u]:
                            # v is ancestor of u in BFS tree
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < best:
                                best = cycle_len
            return best if best != float('inf') else 0

        def find_shortest_cycle(adj, target_len):
            n = len(adj)
            # Return a list of nodes in the cycle
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif dist[v] < dist[u]:
                            # Potential cycle
                            # Reconstruct path
                            path = []
                            cur = u
                            while cur != -1:
                                path.append(cur)
                                cur = parent[cur]
                            path.append(v)
                            # Verify length
                            if len(path) <= target_len:
                                return path
            return None

        # Start search
        n_target = 320
        adj = None
        for _ in range(1000): # Initial random tries
            adj_try = random_4_regular(n_target)
            if adj_try:
                adj = adj_try
                break
        
        if not adj:
            # Fallback to LUW
            return luw_construct(k, g)

        current_girth = get_girth(adj)
        
        while time.time() < deadline:
            if current_girth >= g:
                # Found valid graph
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_target, edges
            
            # Find a short cycle to break
            cycle = find_shortest_cycle(adj, current_girth)
            if not cycle:
                # Re-evaluate girth, might be stale or error
                current_girth = get_girth(adj)
                continue
            
            # Pick an edge in the cycle to swap
            # Cycle nodes: c0-c1-...-ck-c0
            u, v = cycle[0], cycle[1]
            
            # Simple heuristic: Just do random 2-opt swaps and check girth
            # Pick 2 random edges (a,b) and (c,d)
            a = random.randint(0, n_target - 1)
            b = random.choice(adj[a])
            c = random.randint(0, n_target - 1)
            d = random.choice(adj[c])
            
            if b == a or c == a or b == c or d == a or b == d or c == d:
                continue
            if a == c or a == d or b == c or b == d:
                continue
            
            # Swap
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            
            adj[a].append(c)
            adj[c].append(a)
            adj[b].append(d)
            adj[d].append(b)
            
            new_girth = get_girth(adj)
            if new_girth >= current_girth:
                current_girth = new_girth
            else:
                # Revert
                adj[a].remove(c)
                adj[c].remove(a)
                adj[b].remove(d)
                adj[d].remove(b)
                
                adj[a].append(b)
                adj[b].append(a)
                adj[c].append(d)
                adj[d].append(c)

        # If search fails, return LUW
        return luw_construct(k, g)

    else:
        return luw_construct(k, g)
