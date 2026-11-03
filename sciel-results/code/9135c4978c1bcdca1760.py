import time
import random
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: LUW Construction
    def luw_construct():
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

    # Helper: Generate random k-regular graph
    def generate_random_regular(n, k):
        while True:
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            valid = True
            edge_set = set()
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                adj[u].append(v)
                adj[v].append(u)
            
            if valid:
                return adj

    # Helper: Check girth
    def check_girth(adj, n, g):
        for start in range(n):
            visited = {start: 0}
            parent = {start: -1}
            queue = collections.deque([(start, 0)])
            while queue:
                u, depth = queue.popleft()
                if depth >= g - 1:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if v in visited:
                        if depth + 1 + visited[v] < g:
                            return False
                    else:
                        visited[v] = depth + 1
                        parent[v] = u
                        queue.append((v, depth + 1))
        return True

    # Helper: Local Search
    def local_search(n, k, g, time_limit):
        adj = generate_random_regular(n, k)
        if not check_girth(adj, n, g):
            # Try to improve
            limit = time.time() + time_limit
            while time.time() < limit:
                # Find a short cycle
                # Simple heuristic: Pick a random edge, check if it's in a short cycle
                # Or just re-check girth and if fail, try random swaps
                if check_girth(adj, n, g):
                    return adj
                
                # Pick random edge to swap
                u = random.randint(0, n-1)
                if not adj[u]: continue
                v = random.choice(adj[u])
                # Pick another edge (w, x)
                w = random.randint(0, n-1)
                if not adj[w]: continue
                x = random.choice(adj[w])
                
                # Check validity of swap (u,v) and (w,x) -> (u,w) and (v,x)
                # Must not create loops or multi-edges
                if u == w or u == x or v == w or v == x:
                    continue
                if x in adj[u] or w in adj[v]:
                    continue
                
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(x)
                adj[x].remove(w)
                
                adj[u].append(w)
                adj[w].append(u)
                adj[v].append(x)
                adj[x].append(v)
                
        return None

    # Try local search for smaller N
    # Moore bound for k=5, g=9 is 426.
    # Try N from 450 to 1000 in steps.
    for target_n in [450, 500, 600, 800, 1000]:
        if time.time() > deadline - 5:
            break
        # Allocate time based on N
        # Smaller N is harder, give more time? Or just fixed time per attempt.
        time_per_attempt = 10
        if time.time() + time_per_attempt > deadline:
            continue
        result_adj = local_search(target_n, k, g, time_per_attempt)
        if result_adj:
            edges = []
            for i in range(target_n):
                for j in result_adj[i]:
                    if i < j:
                        edges.append((i, j))
            return target_n, edges

    # Fallback to LUW
    return luw_construct()
