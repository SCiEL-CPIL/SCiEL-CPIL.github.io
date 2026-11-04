import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def generate_luw(k, g, d_param=None):
        """Generate a (3, 2d)-cage candidate via the LUW construction."""
        if d_param is None:
            d_param = max(4, g // 2)
        q = 4  # For k=3, q must be 4 (or a prime power for larger k)
        # The LUW construction for k=3 uses GF(4)
        # GF(4) = {0, 1, 2, 3} with 0=0, 1=1, 2=omega, 3=omega^2, omega^2=omega+1=3, omega^3=1
        add_tbl = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
        mul_tbl = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        neg_tbl = [0, 1, 2, 3]  # In GF(4), -x = x for x in {0,1}, -2=3, -3=2... wait
        # Actually in GF(4), characteristic is 2, so -x = x for all x.
        # But the LUW construction uses a specific rule.
        # Let's use the standard LUW rule for k=3.
        # The rule is defined by a sequence.
        # For k=3, the base is a path of length d, and the voltage is determined by a rule.
        
        # Actually, the standard LUW construction for (3, 2d)-cages:
        # Vertices are (side, x) where side in {0,1} and x is a vector in GF(4)^d (or similar).
        # The edges connect (0, x) to (1, y) where y is computed from x.
        # The specific rule for k=3 is well-known.
        
        # Let me use a known working LUW construction for k=3.
        # The construction:
        # - d is the parameter, girth = 2d
        # - n = 2 * 4^d
        # - Vertices: (s, x) where s in {0,1}, x in GF(4)^d
        # - Edges: (s, x) connects to (1-s, y) where y = x + c_s * something?
        
        # Actually, the LUW construction is more subtle. Let me use a known implementation.
        # For k=3, the (3, 2d)-cage from LUW has n = 2*4^d vertices.
        # For d=8, n = 2*65536 = 131072. That's too big.
        # For d=4, n = 2*256 = 512. Girth 8. Not enough.
        # For d=8, girth 16, n = 131072. Too big.
        
        # The issue is that LUW gives girth = 2d, so for g=16, d=8, n=2*4^8=131072.
        # This is way larger than 936.
        
        # So LUW is not the right construction for this problem.
        # The (3,16)-cage with 936 vertices is from a different construction.
        
        # Let me think about what construction gives 936 vertices.
        # 936 = 2 * 468 = 2 * 4 * 117 = 8 * 117 = 8 * 9 * 13 = 72 * 13.
        # Or 936 = 2 * 3 * 13 * 12 = 2 * 3 * 13 * 4 * 3 = 24 * 39.
        # 936 = 2 * 468. 468 = 4 * 117 = 4 * 9 * 13.
        # So 936 = 8 * 9 * 13.
        
        # A (3,16)-cage with 936 vertices... this is from a specific construction.
        # It might be a covering of a small graph.
        
        # Alternative approach: Use the known fact that the (3,16)-cage has 936 vertices.
        # We can try to construct it using a different method.
        
        # One approach: Use a random 3-regular graph with n=936 vertices and check if girth >= 16.
        # The probability is low, but we can try many times.
        
        # Another approach: Use a construction based on the Hoffman-Singleton bound or similar.
        
        # Actually, let me reconsider. The task says the smallest known is 936 vertices.
        # The seed program was producing ~4k vertices. We need to beat 4k.
        # 936 is much better than 4k.
        
        # Let me try a different approach: Use a random 3-regular graph with n=936 and check girth.
        # If girth >= 16, we're done. Otherwise, we can try to improve.
        
        # But the probability of a random 3-regular graph on 936 vertices having girth >= 16 is very low.
        # The expected number of cycles of length 16 is about 2^16 / 32 = 2048.
        # So almost all random graphs will have many short cycles.
        
        # We need a smarter construction.
        
        # Let me think about the structure of the (3,16)-cage.
        # 936 = 2 * 468. 
        # One possibility: It's a double cover of a (3,8)-cage or similar.
        
        # Actually, I recall that cages can be constructed using voltage graphs.
        # A voltage graph is a graph with edges labeled by elements of a group.
        # The derived graph has |V| * |G| vertices.
        
        # For k=3, g=16, n=936:
        # 936 = 72 * 13. So maybe a 72-vertex base graph with a 13-element group.
        # Or 936 = 24 * 39. A 24-vertex base with a 39-element group.
        # Or 936 = 8 * 117. An 8-vertex base with a 117-element group.
        # 117 = 9 * 13.
        
        # A voltage graph with a cyclic group of order 117? That's large.
        # Or a product of two groups?
        
        # This is getting too complex. Let me try a different approach.
        
        # Approach: Use a random 3-regular graph with n=936, and use local search to eliminate short cycles.
        # This might work if we're clever about the swaps.
        
        # Let me implement this.
        
        return None  # LUW is not suitable for this problem

    def generate_random_3_regular(n):
        """Generate a random 3-regular graph using the configuration model."""
        if n % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(3)]
        import random
        random.shuffle(stubs)
        edges = []
        seen = set()
        for i in range(0, n * 3, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                return None
            if (u, v) in seen or (v, u) in seen:
                return None
            edges.append((u, v))
            seen.add((u, v))
            seen.add((v, u))
        return edges

    def get_adjacency(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def find_shortest_cycle(adj, n, max_girth_check):
        """Find the shortest cycle in the graph using BFS from each node."""
        min_girth = max_girth_check + 1
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] + 1 >= min_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    def find_one_short_cycle(adj, n, limit):
        """Find one cycle of length < limit."""
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] + 1 >= limit:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < limit:
                            # Reconstruct cycle
                            path_u = []
                            curr = u
                            while curr != -1:
                                path_u.append(curr)
                                curr = parent[curr]
                            path_v = []
                            curr = v
                            while curr != -1:
                                path_v.append(curr)
                                curr = parent[curr]
                            cycle_nodes = path_u + path_v[1:][::-1]
                            return cycle_nodes
        return None

    def try_local_search(edges, n, target_girth, max_iters=5000):
        """Try to improve the girth via edge swaps."""
        import random
        adj = get_adjacency(edges, n)
        current_girth = find_shortest_cycle(adj, n, target_girth)
        if current_girth >= target_girth:
            return edges, True

        seen = set()
        for u, v in edges:
            seen.add((u, v))
            seen.add((v, u))

        for _ in range(max_iters):
            if time_left() < 5.0:
                return edges, False

            cycle_nodes = find_one_short_cycle(adj, n, target_girth)
            if not cycle_nodes:
                return edges, True

            cycle_set = set(cycle_nodes)
            cycle_edges = []
            for i in range(len(cycle_nodes)):
                u = cycle_nodes[i]
                v = cycle_nodes[(i + 1) % len(cycle_nodes)]
                cycle_edges.append((u, v))

            u, v = random.choice(cycle_edges)

            success = False
            for _ in range(30):
                if time_left() < 5.0:
                    return edges, False
                x, y = random.choice(edges)
                if x == u or x == v or y == u or y == v:
                    continue

                # Try swap 1: (u,x) and (v,y)
                if (u, x) not in seen and (v, y) not in seen and u != x and v != y:
                    edges.remove((u, v))
                    edges.remove((x, y))
                    seen.discard((u, v))
                    seen.discard((v, u))
                    seen.discard((x, y))
                    seen.discard((y, x))

                    edges.append((u, x))
                    edges.append((v, y))
                    seen.add((u, x))
                    seen.add((x, u))
                    seen.add((v, y))
                    seen.add((y, v))

                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)

                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)

                    success = True
                    break

                # Try swap 2: (u,y) and (v,x)
                if (u, y) not in seen and (v, x) not in seen and u != y and v != x:
                    edges.remove((u, v))
                    edges.remove((x, y))
                    seen.discard((u, v))
                    seen.discard((v, u))
                    seen.discard((x, y))
                    seen.discard((y, x))

                    edges.append((u, y))
                    edges.append((v, x))
                    seen.add((u, y))
                    seen.add((y, u))
                    seen.add((v, x))
                    seen.add((x, v))

                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)

                    adj[u].append(y)
                    adj[y].append(u)
                    adj[v].append(x)
                    adj[x].append(v)

                    success = True
                    break

            if not success:
                continue

            current_girth = find_shortest_cycle(adj, n, target_girth)
            if current_girth >= target_girth:
                return edges, True

        return edges, False

    # Try to construct a (3,16)-cage with n=936 vertices
    best_n = None
    best_edges = None

    # Try n=936 first
    n_target = 936
    attempts = 20
    for attempt in range(attempts):
        if time_left() < 10.0:
            break

        edges = generate_random_3_regular(n_target)
        if not edges:
            continue

        adj = get_adjacency(edges, n_target)
        girth = find_shortest_cycle(adj, n_target, g)

        if girth >= g:
            best_n = n_target
            best_edges = edges
            break

        # Try local search
        if time_left() > 15.0:
            edges2, success = try_local_search(edges, n_target, g, max_iters=2000)
            if success:
                best_n = n_target
                best_edges = edges2
                break

    # If 936 didn't work, try larger n
    if best_n is None:
        for n_try in [938, 940, 942, 944, 946, 948, 950, 960, 980, 1000, 1024, 1040, 1056, 1072, 1088, 1104, 1120, 1136, 1152, 1200, 1280, 1400, 1600, 2000, 3000, 4000]:
            if time_left() < 15.0:
                break

            edges = generate_random_3_regular(n_try)
            if not edges:
                continue

            adj = get_adjacency(edges, n_try)
            girth = find_shortest_cycle(adj, n_try, g)

            if girth >= g:
                best_n = n_try
                best_edges = edges
                break

            if time_left() > 20.0:
                edges2, success = try_local_search(edges, n_try, g, max_iters=1000)
                if success:
                    best_n = n_try
                    best_edges = edges2
                    break

    # If we still don't have a valid graph, use a larger n and try harder
    if best_n is None:
        for n_try in [4096, 5120, 6144, 8192]:
            if time_left() < 20.0:
                break

            edges = generate_random_3_regular(n_try)
            if not edges:
                continue

            adj = get_adjacency(edges, n_try)
            girth = find_shortest_cycle(adj, n_try, g)

            if girth >= g:
                best_n = n_try
                best_edges = edges
                break

            if time_left() > 30.0:
                edges2, success = try_local_search(edges, n_try, g, max_iters=5000)
                if success:
                    best_n = n_try
                    best_edges = edges2
                    break

    # Fallback: use a very large n where a random graph is likely to have high girth
    if best_n is None:
        # For a random 3-regular graph, the girth is approximately log_2(n).
        # For girth 16, we need n ~ 2^16 = 65536.
        # But that's large. Let's try n=65536.
        # Actually, the expected number of cycles of length l is (k-1)^l / (2l).
        # For l=15, expected cycles = 2^15 / 30 = 109.
        # For l=16, expected cycles = 2^16 / 32 = 2048.
        # So a random graph on 65536 vertices will have many cycles of length 15 and 16.
        # We need to go larger.
        # For girth 16, we need no cycles of length < 16.
        # The probability of no cycles of length 15 is exp(-109) ~ 0.
        # So we need an even larger graph.
        # This approach is not viable.
        
        # Instead, let's use a construction that is known to work.
        # The LUW construction for k=3, d=8 gives n=131072 and girth=16.
        # Let's implement that as a fallback.
        
        def get_luw_graph_fallback(k, g, d_param=8):
            """LUW construction for k=3, girth=2*d_param."""
            q = 4
            d = d_param
            add_tbl = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
            mul_tbl = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            
            def add(x, y):
                return add_tbl[x][y]
            
            def sub(x, y):
                # In GF(4), sub = add since char 2
                return add_tbl[x][y]
            
            def mul(x, y):
                return mul_tbl[x][y]
            
            # LUW rule for k=3
            # The rule is a sequence of pairs (a, b) for c = 2, 3, ..., d
            # This is the standard LUW rule
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r = (c - 5) % 4
                c0 = c - r
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            
            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [0] * (d + 1)
                    y[0] = 0
                    y[1] = first
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        if side == 0:
                            y[c] = add(x[c], mul(y[a], x[b]))
                        else:
                            y[c] = sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y[1:])
            
            start = (0, tuple([0] * d))
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
        
        best_n, best_edges = get_luw_graph_fallback(k, g, 8)

    return best_n, best_edges
