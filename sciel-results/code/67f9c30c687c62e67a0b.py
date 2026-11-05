import random
import time
import sys
from collections import deque

def construct(k, g):
    # Fallback: The Lazebnik-Ustimenko-Woldar construction from the seed program
    def l_u_w_construct(k, g):
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

    # Primary Strategy: Greedy Randomized Construction
    # Try to build a k-regular graph of girth >= g with small n
    
    def try_greedy(n, k, g, timeout=5):
        start_time = time.time()
        deg = [0] * n
        adj = [[] for _ in range(n)]
        edges = []
        target_edges = (n * k) // 2
        
        # Helper to check if adding (u, v) creates a cycle < g
        # This is equivalent to checking if dist(u, v) < g - 1 in the current graph
        def is_valid_edge(u, v):
            if u == v:
                return False
            if v in adj[u]:
                return False
            
            # BFS from u to see if v is reachable within g-2 steps
            # If reachable in < g-1 steps, adding edge creates cycle < g
            max_depth = g - 2
            visited = {u: 0}
            queue = deque([(u, 0)])
            
            while queue:
                curr, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for nei in adj[curr]:
                    if nei == v:
                        return False
                    if nei not in visited:
                        visited[nei] = depth + 1
                        queue.append((nei, depth + 1))
            return True

        # We will try to add edges one by one
        # To avoid getting stuck, we pick a random vertex with deg < k
        # and a random neighbor with deg < k
        
        # Optimization: Pre-compute candidates? No, dynamic.
        # We use a list of vertices sorted by degree to pick the one with lowest degree first (greedy)
        # Or random. Random is better for avoiding local minima.
        
        attempts = 0
        max_attempts = 1000000
        
        while len(edges) < target_edges:
            if time.time() - start_time > timeout:
                return None
            
            # Pick a random vertex u with deg < k
            candidates_u = [i for i in range(n) if deg[i] < k]
            if not candidates_u:
                break
            u = random.choice(candidates_u)
            
            # Pick a random vertex v with deg < k, v != u
            candidates_v = [i for i in range(n) if deg[i] < k and i != u]
            if not candidates_v:
                continue
            
            random.shuffle(candidates_v)
            added = False
            for v in candidates_v:
                if is_valid_edge(u, v):
                    # Add edge
                    adj[u].append(v)
                    adj[v].append(u)
                    deg[u] += 1
                    deg[v] += 1
                    edges.append((u, v))
                    added = True
                    break
            
            if not added:
                # If we couldn't add an edge for u, we might be stuck.
                # In a proper greedy algo, we'd backtrack. Here we just restart or fail.
                # To increase success rate, we can try to "relax" by removing a recent edge?
                # For simplicity, we just fail this attempt.
                return None
                
        # Verify
        if len(edges) != target_edges:
            return None
        for i in range(n):
            if deg[i] != k:
                return None
        
        # Final girth check
        # We already ensured no short cycles during construction, but let's be safe.
        # Actually, the construction ensures that NO cycle of length < g is formed.
        # So the girth is >= g.
        
        return n, edges

    # Try different n values
    # Moore bound for k=5, g=9 is 425.
    # Known best is 1116.
    # Let's try n around 1000-1200.
    
    # If k and g are such that the problem is very hard, we might just use the fallback.
    # But for k=5, g=9, we can try.
    
    # Time budget management
    total_time_limit = 110  # Leave 10s for fallback
    
    # Try a few n values
    # Start from a reasonable guess.
    # For k=5, g=9, n=1000 is a good guess.
    
    # If the parameters are different, we should adapt.
    # For now, assume k=5, g=9 is the main case.
    
    if k == 5 and g == 9:
        n_trials = [1000, 1050, 1100, 1150, 1200]
        for n in n_trials:
            res = try_greedy(n, k, g, timeout=15)
            if res:
                return res
    else:
        # For other parameters, we might not have a tuned strategy.
        # We can still try a generic greedy approach with a small n.
        # But to be safe, we fall back to L-U-W if it's not the specific case.
        # Or try a small n.
        moore_bound = 0
        if g % 2 == 1:
            r = (g - 3) // 2
            for i in range(r):
                moore_bound += (k - 1) ** i
            moore_bound *= k
        else:
            r = g // 2 - 1
            for i in range(r + 1):
                moore_bound += (k - 1) ** i
            moore_bound *= 2
        
        n_start = max(10, moore_bound)
        for n in [n_start, n_start + 50, n_start + 100]:
            res = try_greedy(n, k, g, timeout=10)
            if res:
                return res

    # Fallback
    return l_u_w_construct(k, g)
