import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback to seed if we fail
    def seed_construct(k, g):
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

    def get_seed_graph(k, g):
        n_seed, edges_seed = seed_construct(k, g)
        adj = [[] for _ in range(n_seed)]
        for u, v in edges_seed:
            adj[u].append(v)
            adj[v].append(u)
        return n_seed, adj

    def find_shortest_cycle(adj, n, max_len):
        """
        Finds the length of the shortest cycle in the graph.
        Returns g if girth >= g, otherwise the length of the shortest cycle.
        Uses BFS from each node.
        """
        min_cycle = max_len
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                if dist[u] >= max_len // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and v != start:
                        # Found a back edge
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle >= g:
                                return min_cycle
                    elif v == start and dist[u] + 1 < max_len:
                        cycle_len = dist[u] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle >= g:
                                return min_cycle
        return min_cycle

    def generate_random_cubic_graph(n):
        if n % 2 != 0 or n < 4:
            return None
        # Start with a ring
        adj = [[(i-1)%n, (i+1)%n] for i in range(n)]
        # Add a perfect matching
        for _ in range(1000):
            slots = list(range(n))
            random.shuffle(slots)
            adj_temp = [list(neighs) for neighs in adj]
            success = True
            for i in range(0, n, 2):
                u = slots[i]
                v = slots[i+1]
                if u == v or v in adj_temp[u]:
                    success = False
                    break
                adj_temp[u].append(v)
                adj_temp[v].append(u)
            if success:
                # Check if simple (no multi-edges) - the generation ensures no self-loops, 
                # but could create multi-edges if v was already a neighbor? 
                # We check `v in adj_temp[u]` before adding.
                # So it is simple.
                return adj_temp
        return None

    def try_improve(n_target, time_limit):
        """
        Try to find a 3-regular graph of size n_target with girth >= g.
        Returns (n, edges) if successful, else None.
        """
        adj = generate_random_cubic_graph(n_target)
        if not adj:
            return None
        
        start_t = time.time()
        best_cycle_len = find_shortest_cycle(adj, n_target, g)
        
        while time.time() - start_t < time_limit:
            if best_cycle_len >= g:
                # Success
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_target, edges

            # Find a cycle of length best_cycle_len
            # For simplicity, we just pick a random edge and try to break it.
            # A better way: find the specific cycle.
            # Here we use a heuristic: pick a random edge (u,v).
            # If it's part of a short cycle, the switch might help.
            
            all_edges = [(u, v) for u in range(n_target) for v in adj[u] if u < v]
            if not all_edges:
                break
            
            u, v = random.choice(all_edges)
            
            # Find a partner edge (x, y)
            # x, y must be distinct from u, v
            # and (u,x), (v,y) must not exist
            attempts = 0
            while attempts < 20:
                x, y = random.choice(all_edges)
                if x in (u, v) or y in (u, v):
                    attempts += 1
                    continue
                if x == y:
                    attempts += 1
                    continue
                
                # Check if switch is valid
                if x in adj[u] or y in adj[v]:
                    attempts += 1
                    continue
                
                # Perform switch
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].append(x)
                adj[x].append(u)
                adj[v].append(y)
                adj[y].append(v)
                
                # Check new girth
                new_cycle_len = find_shortest_cycle(adj, n_target, g)
                
                if new_cycle_len >= g:
                    # Success
                    edges = []
                    for i in range(n_target):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n_target, edges
                
                # Accept if better or equal (with some probability)
                if new_cycle_len > best_cycle_len:
                    best_cycle_len = new_cycle_len
                elif new_cycle_len == best_cycle_len:
                    # Keep with prob 0.5 to explore
                    if random.random() > 0.5:
                        pass
                    else:
                        # Revert
                        adj[u].remove(x)
                        adj[x].remove(u)
                        adj[v].remove(y)
                        adj[y].remove(v)
                        adj[u].append(v)
                        adj[v].append(u)
                        adj[x].append(y)
                        adj[y].append(x)
                else:
                    # Revert
                    adj[u].remove(x)
                    adj[x].remove(u)
                    adj[v].remove(y)
                    adj[y].remove(v)
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[x].append(y)
                    adj[y].append(x)
                    
                attempts += 1
        
        return None

    # Main
    n_seed, adj_seed = get_seed_graph(k, g)
    
    # Try to find a smaller graph
    # We try sizes from max(512, n_seed - 20) up to n_seed - 1
    # We allocate time based on remaining budget
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # We have time to search
        search_range = range(max(512, n_seed - 20), n_seed)
        time_per_size = max(0.5, (remaining_time - 5) / len(search_range))
        
        for n_target in search_range:
            if time.time() > deadline - 5:
                break
            result = try_improve(n_target, time_per_size)
            if result:
                return result
                
    # Fallback to seed
    edges_seed = []
    for u in range(n_seed):
        for v in adj_seed[u]:
            if u < v:
                edges_seed.append((u, v))
    return n_seed, edges_seed
