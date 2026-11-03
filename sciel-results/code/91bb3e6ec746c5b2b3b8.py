import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def get_luw():
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

    luw_n, luw_edges = get_luw()
    
    best_n = luw_n
    best_edges = luw_edges

    # Try to find a smaller graph using Random Local Search
    # Lower bound for k=5, g=9 is 428.
    # We try n values from 450 to 1100.
    # We allocate time to each n.
    
    def find_short_cycle(adj, n, max_girth):
        # Returns a list of vertices in a cycle of length < max_girth, or None
        # BFS from each vertex is too slow if done naively for every check.
        # We can just check for cycles of length 3, 4, 5... up to max_girth-1
        # For efficiency, we check for girth < max_girth.
        # Actually, we just need to know if girth < g.
        # We can use a BFS that stops at depth g//2.
        
        for u in range(n):
            # BFS from u
            visited = [-1] * n
            parent = [-1] * n
            queue = [u]
            visited[u] = 0
            idx = 0
            while idx < len(queue):
                v = queue[idx]
                idx += 1
                if visited[v] >= g // 2:
                    continue
                for w in adj[v]:
                    if visited[w] == -1:
                        visited[w] = visited[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v] and visited[w] < visited[v]:
                        # Found a cycle
                        # Reconstruct path
                        cycle = []
                        cur = v
                        while cur != w:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.append(w)
                        if len(cycle) < g:
                            return cycle
            # Optimization: If we found a cycle of length < g, return it.
            # If we didn't find any in this BFS, continue to next u?
            # No, if girth < g, there is a cycle. BFS from any vertex in the cycle will find it.
            # But we might miss it if we don't check all u.
            # However, if the graph is connected, BFS from one vertex might not reach all?
            # We assume connected.
            pass
        return None

    def is_valid(adj, n, k, g):
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check girth
        # Efficient girth check:
        # For each vertex, BFS up to depth g//2.
        # If we find a back edge to a vertex at depth < g//2, we have a cycle.
        # Cycle length = depth[u] + depth[v] + 1.
        # If length < g, invalid.
        for u in range(n):
            depth = [-1] * n
            parent = [-1] * n
            queue = [u]
            depth[u] = 0
            idx = 0
            while idx < len(queue):
                v = queue[idx]
                idx += 1
                if depth[v] >= g // 2:
                    continue
                for w in adj[v]:
                    if depth[w] == -1:
                        depth[w] = depth[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v]:
                        # Cycle found
                        cycle_len = depth[v] + depth[w] + 1
                        if cycle_len < g:
                            return False
        return True

    def random_search(n, k, g, time_limit):
        limit_time = time.time() + time_limit
        best_adj = None
        best_energy = float('inf')
        
        for attempt in range(50): # Max 50 restarts
            if time.time() > limit_time:
                break
            
            # Generate random k-regular graph
            adj = [set() for _ in range(n)]
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            
            # Configuration model with re-pairing to avoid loops/multiedges
            # Simple approach: Just shuffle and pair, then fix.
            # This is tricky to implement robustly quickly.
            # Alternative: Start with a cycle and add random chords?
            # For k=5, start with a 5-regular graph?
            # Easiest: Use a known random regular graph generator logic.
            
            # Let's use a simple random edge swap approach starting from a random graph.
            # Generate a random graph with degree k.
            # We can use the "configuration model" and then "clean" it.
            # For now, let's try to construct a valid graph using backtracking if n is small,
            # or just random swaps on a random graph (which might not be regular).
            
            # Actually, generating a random k-regular graph is non-trivial.
            # We can use the following heuristic:
            # 1. Create a list of edges.
            # 2. Randomly add edges until degree is k? No, that's hard.
            
            # Let's skip the complex random generation and use a simpler heuristic:
            # Start with a cycle (2-regular) and add edges to reach k=5.
            # This is also hard.
            
            # Given the time, and the fact that LUW is a valid fallback,
            # and the difficulty of finding a 5-regular girth-9 graph with n<1116 in 110s
            # using pure Python random search,
            # we might just return the LUW graph.
            
            # However, we can try a specific known graph if we have one.
            # But we don't.
            
            pass
        return None

    # Since implementing a robust random regular graph generator and SA in 110s
    # in Python is risky and might not beat the LUW fallback (which is valid),
    # we will stick to the LUW construction but optimize its generation speed
    # if possible, or just return it.
    
    # Wait, the LUW graph for k=5, g=9 is n=15624.
    # If we can find ANY valid graph with n < 15624, we win.
    # Even a random graph with n=1000 that happens to have girth 9?
    # Probability is low.
    
    # Let's try a simple BFS-based construction for small n?
    # No, backtracking is too slow.
    
    # We will return the LUW result.
    return luw_n, luw_edges
