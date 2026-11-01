import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # This guarantees a valid result, though potentially large N.
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
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    if k == 4 and g == 10:
        target_n = 320
        
        def gen_random_4_reg(n):
            while True:
                stubs = [i for i in range(n) for _ in range(4)]
                random.shuffle(stubs)
                edges = set()
                valid = True
                for i in range(0, n * 4, 2):
                    u = stubs[i]
                    v = stubs[i + 1]
                    if u == v:
                        valid = False
                        break
                    if (v, u) in edges:
                        valid = False
                        break
                    edges.add((u, v))
                if valid:
                    return list(edges)
        
        def get_adj(edges, n):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        def find_short_cycle(adj, n, limit):
            best_len = limit
            best_cycle = None
            
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                queue = [start]
                dist[start] = 0
                parent[start] = -2
                
                while queue:
                    next_q = []
                    for u in queue:
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                next_q.append(v)
                            elif v != parent[u] and dist[v] < dist[u]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < best_len:
                                    cycle = [u]
                                    cur = u
                                    while cur != v:
                                        cur = parent[cur]
                                        cycle.append(cur)
                                    best_len = cycle_len
                                    best_cycle = cycle
                                    if best_len == 3:
                                        return 3, best_cycle
                    queue = next_q
            
            return best_len, best_cycle

        def try_swap(edges_set, adj, u, v, x, y):
            if u == x or u == y or v == x or v == y:
                return None
            if (min(u,x), max(u,x)) in edges_set:
                return None
            if (min(v,y), max(v,y)) in edges_set:
                return None
            
            new_edges_set = set(edges_set)
            new_edges_set.discard((min(u,v), max(u,v)))
            new_edges_set.discard((min(x,y), max(x,y)))
            new_edges_set.add((min(u,x), max(u,x)))
            new_edges_set.add((min(v,y), max(v,y)))
            
            new_adj = [list(neigh) for neigh in adj]
            new_adj[u].remove(v)
            new_adj[v].remove(u)
            new_adj[x].remove(y)
            new_adj[y].remove(x)
            
            new_adj[u].append(x)
            new_adj[x].append(u)
            new_adj[v].append(y)
            new_adj[y].append(v)
            
            return new_edges_set, new_adj

        restart_deadline = start_time + 140
        
        while time.time() < restart_deadline:
            edges = gen_random_4_reg(target_n)
            edges_set = set((min(u,v), max(u,v)) for u,v in edges)
            adj = get_adj(edges, target_n)
            
            girth, cycle = find_short_cycle(adj, target_n, g)
            if girth >= g:
                return target_n, list(edges_set)
            
            local_deadline = time.time() + 10 # 10s per restart
            while time.time() < local_deadline and time.time() < restart_deadline:
                if girth >= g:
                    return target_n, list(edges_set)
                
                if not cycle:
                    break
                
                edge_idx = random.randint(0, len(cycle)-1)
                u = cycle[edge_idx]
                v = cycle[(edge_idx + 1) % len(cycle)]
                
                swapped = False
                for _ in range(100):
                    if time.time() >= local_deadline:
                        break
                    e1 = random.choice(list(edges_set))
                    x, y = e1
                    if x == u or x == v or y == u or y == v:
                        continue
                    
                    new_es, new_adj = try_swap(edges_set, adj, u, v, x, y)
                    if new_es is None:
                        continue
                    
                    new_girth, new_cycle = find_short_cycle(new_adj, target_n, g)
                    
                    if new_girth > girth:
                        edges_set = new_es
                        adj = new_adj
                        girth = new_girth
                        cycle = new_cycle
                        swapped = True
                        break
                    elif new_girth == girth and random.random() < 0.3:
                        edges_set = new_es
                        adj = new_adj
                        cycle = new_cycle
                        swapped = True
                        break
                
                if not swapped:
                    break
        
        n_luw, edges_luw = luw_construct()
        return n_luw, edges_luw
    else:
        n, edges = luw_construct()
        return n, edges
