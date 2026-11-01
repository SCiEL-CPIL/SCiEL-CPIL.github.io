import time
import random
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def moore_bound(k, g):
        if g % 2 == 0:
            half = g // 2
            if k == 2:
                s = half
            else:
                s = ((k-1)**half - 1) // (k-2)
            return 2 * s
        else:
            half = (g - 1) // 2
            if k == 2:
                s = half
            else:
                s = ((k-1)**half - 1) // (k-2)
            return 2 * s + (k-1) * ((k-1)**(half-1))
            
    min_n = moore_bound(k, g)
    if min_n % 2 != 0:
        min_n += 1

    def generate_random_regular_graph(n, k):
        if n * k % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        edges = set()
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            if (v, u) in edges:
                return None
            edges.add((u, v))
        return list(edges)
    
    def check_girth(n, edges_set, g):
        # Returns girth and cycle nodes if girth < g
        adj = [[] for _ in range(n)]
        for u, v in edges_set:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = float('inf')
        min_cycle = None
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            # Reconstruct cycle
                            path1 = []
                            cur = u
                            while cur != -1:
                                path1.append(cur)
                                cur = parent[cur]
                            path1.reverse()
                            
                            path2 = []
                            cur = v
                            while cur != -1:
                                path2.append(cur)
                                cur = parent[cur]
                            path2.reverse()
                            
                            # Cycle: u ... start ... v ... u
                            # path1: start ... u
                            # path2: start ... v
                            cycle_nodes = []
                            cur = u
                            while cur != start:
                                cycle_nodes.append(cur)
                                cur = parent[cur]
                            cycle_nodes.append(start)
                            cur = v
                            while cur != start:
                                cycle_nodes.append(cur)
                                cur = parent[cur]
                            min_cycle = cycle_nodes
                            if min_girth >= g:
                                return min_girth, min_cycle
            if min_girth >= g:
                return min_girth, min_cycle
                
        return min_girth, min_cycle

    def local_search(n, edges, g, time_limit):
        edges_set = set()
        for u, v in edges:
            edges_set.add((u, v))
            
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        current_girth, current_cycle = check_girth(n, edges_set, g)
        
        start_ts = time.time()
        
        while time.time() - start_ts < time_limit:
            if current_girth >= g:
                return list(edges_set)
                
            if not current_cycle:
                break
                
            cycle_len = len(current_cycle)
            if cycle_len < 3:
                break
                
            edge_idx = random.randint(0, cycle_len - 1)
            u = current_cycle[edge_idx]
            v = current_cycle[(edge_idx + 1) % cycle_len]
            
            for _ in range(50):
                x = random.randint(0, n - 1)
                if x == u or x == v:
                    continue
                y = random.randint(0, n - 1)
                if y == u or y == v or y == x:
                    continue
                
                if (x, y) not in edges_set and (y, x) not in edges_set:
                    continue
                
                # Check validity
                if u == x or u == y or v == x or v == y:
                    continue
                if (u, x) in edges_set or (x, u) in edges_set:
                    continue
                if (v, y) in edges_set or (y, v) in edges_set:
                    continue
                
                # Swap
                edges_set.remove((u, v))
                edges_set.remove((x, y))
                edges_set.add((u, x))
                edges_set.add((v, y))
                
                adj[u].remove(v)
                adj[u].append(x)
                adj[v].remove(u)
                adj[v].append(y)
                adj[x].remove(y)
                adj[x].append(u)
                adj[y].remove(x)
                adj[y].append(v)
                
                new_girth, new_cycle = check_girth(n, edges_set, g)
                
                if new_girth > current_girth:
                    current_girth = new_girth
                    current_cycle = new_cycle
                else:
                    # Revert
                    edges_set.remove((u, x))
                    edges_set.remove((v, y))
                    edges_set.add((u, v))
                    edges_set.add((x, y))
                    
                    adj[u].remove(x)
                    adj[u].append(v)
                    adj[v].remove(y)
                    adj[v].append(u)
                    adj[x].remove(u)
                    adj[x].append(y)
                    adj[y].remove(v)
                    adj[y].append(x)
                break

    # Fallback LUW
    def luw_fallback(k, g):
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

    # Main Search
    n = min_n
    while time.time() < deadline - 5.0:
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
        limit = min(5.0, time_left / 5.0)
        
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            edges = generate_random_regular_graph(n, k)
            if edges is None:
                continue
            result = local_search(n, edges, g, limit)
            if result:
                final_girth, _ = check_girth(n, set(result), g)
                if final_girth >= g:
                    return n, result
        n += 2
        if n > 1500:
            break
            
    # Fallback
    if time.time() < deadline:
        return luw_fallback(k, g)
    else:
        # Emergency fallback: return a small invalid graph? No, must be valid.
        # If LUW takes too long? It shouldn't for small k.
        return luw_fallback(k, g)
