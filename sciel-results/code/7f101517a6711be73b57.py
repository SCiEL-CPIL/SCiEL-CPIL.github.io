import time
import random
import sys

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        d = max(3, (g + 2) // 2)
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

    D = (g - 2) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(D))
    
    deadline = time.time() + 110
    
    if k == 4 and g == 10:
        # Try to find the 320 cage
        for target_n in [320, 322, 324, 326, 328, 330]:
            if time.time() > deadline - 20:
                break
            result = try_find_cage(target_n, k, g, deadline - time.time() - 5)
            if result:
                return result
        
        return luw_construct(k, g)
    
    return luw_construct(k, g)

def try_find_cage(n, k, g, time_limit):
    start_time = time.time()
    
    def has_girth_at_least(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for u in range(n):
            if len(adj[u]) != k:
                return False
        
        half_g = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def find_one_short_cycle(n, adj, g):
        half_g = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                path1 = []
                                cur = u
                                while cur != start:
                                    path1.append(cur)
                                    cur = parent[cur]
                                path1.append(start)
                                path1.reverse()
                                
                                path2 = []
                                cur = v
                                while cur != start:
                                    path2.append(cur)
                                    cur = parent[cur]
                                path2.append(start)
                                path2.reverse()
                                
                                i = 0
                                while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
                                    i += 1
                                cycle = path1[i:] + path2[i-1::-1]
                                if len(cycle) >= 3:
                                    return cycle
        return None

    def random_regular_graph(n, k):
        while True:
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            valid = True
            seen = set()
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in seen or (v, u) in seen:
                    valid = False
                    break
                seen.add((u, v))
                edges.append((u, v))
            if valid:
                return edges

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    while time.time() - start_time < time_limit:
        edges = random_regular_graph(n, k)
        adj = build_adj(edges, n)
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        
        stall_count = 0
        while time.time() - start_time < time_limit:
            if has_girth_at_least(n, edges, g):
                return n, edges
            
            cycle = find_one_short_cycle(n, adj, g)
            if not cycle:
                return n, edges
            
            broken = False
            cycle_list = list(cycle)
            L = len(cycle_list)
            
            for _ in range(100):
                idx = random.randint(0, L - 1)
                u = cycle_list[idx]
                v = cycle_list[(idx + 1) % L]
                
                a, b = random.choice(edges)
                if a == u or a == v or b == u or b == v:
                    continue
                
                if (min(u,a), max(u,a)) in edge_set or (min(v,b), max(v,b)) in edge_set:
                    continue
                
                new_edge_set = set(edge_set)
                new_edge_set.discard((min(u,v), max(u,v)))
                new_edge_set.discard((min(a,b), max(a,b)))
                new_edge_set.add((min(u,a), max(u,a)))
                new_edge_set.add((min(v,b), max(v,b)))
                
                valid = True
                for e in new_edge_set:
                    if e[0] == e[1]:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_edges = [list(e) for e in new_edge_set]
                new_adj = build_adj(new_edges, n)
                
                edges = new_edges
                edge_set = new_edge_set
                adj = new_adj
                broken = True
                break
                
            if not broken:
                stall_count += 1
                if stall_count > 5:
                    break
            else:
                stall_count = 0
                
    return None
