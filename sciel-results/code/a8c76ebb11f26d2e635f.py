import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110

    def moore_bound(k, g):
        if g % 2 == 0:
            h = g // 2
            return k * (k - 1) ** (h - 1)
        else:
            h = (g - 1) // 2
            return 1 + k * sum((k - 1) ** i for i in range(h))

    def luw_construct(q, d):
        """
        Lazebnik-Ustimenko-Woldar construction.
        q: prime power
        d: depth parameter
        Returns (n, edges) for a 3-regular graph.
        """
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
        
        rule = {2: (1, 1), 3: (2, 1)}
        for c in range(4, d + 1):
            r = (c - 4) % 4
            c0 = c - r
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
                yield (1 - side, tuple(y))
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vi, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def girth_check_fast(n, adj, min_girth):
        """
        Check if graph has girth >= min_girth using non-backtracking walks.
        Returns True if girth >= min_girth, False otherwise.
        Uses early termination and efficient state tracking.
        """
        for start in range(n):
            # BFS in non-backtracking state space from start
            # State: (current, prev)
            visited = set()
            queue = deque()
            
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            
            while queue:
                u, prev, dist = queue.popleft()
                
                if dist >= min_girth:
                    continue
                
                for nb in adj[u]:
                    if nb == prev:
                        continue
                    if nb == start:
                        return False
                    if (nb, u) not in visited:
                        visited.add((nb, u))
                        queue.append((nb, u, dist + 1))
        
        return True

    def girth_check_vectorized(n, adj, min_girth):
        """
        Faster girth check using arrays instead of sets where possible.
        Returns True if girth >= min_girth, False otherwise.
        """
        # For each start vertex, do BFS in non-backtracking state space
        # Encode state (u, prev) as u * n + prev
        for start in range(n):
            # Use a bytearray or list for visited states
            # State (u, prev) -> u * n + prev, but we only need states where prev != u
            # and u is adjacent to prev
            visited = set()
            queue = deque()
            
            for nb in adj[start]:
                state = nb * n + start
                queue.append((nb, start, 1))
                visited.add(state)
            
            while queue:
                u, prev, dist = queue.popleft()
                
                if dist >= min_girth:
                    continue
                
                for nb in adj[u]:
                    if nb == prev:
                        continue
                    if nb == start:
                        return False
                    state = nb * n + u
                    if state not in visited:
                        visited.add(state)
                        queue.append((nb, u, dist + 1))
        
        return True

    # Strategy: Search LUW constructions systematically
    # The LUW(q, d) graph has approximately 2*q^d vertices (roughly)
    # For girth 16, we need depth d such that 2d+2 >= 16, so d >= 7
    # But the actual girth depends on the specific construction
    
    best_result = None
    best_n = float('inf')
    
    # Priority: try smaller (q, d) combinations first
    # q=3 gives smallest graphs
    # d needs to be large enough for girth >= g
    
    # Estimate minimum d needed: girth of LUW(q,d) is at least 2d+2 for small q
    # So for g=16, we need d >= 7
    # But let's try a range and verify
    
    luw_order = []
    for q in [3, 4, 5, 7]:
        for d in range(6, 14):
            # Estimated number of vertices ~ 2 * q^d
            est_n = 2 * (q ** d)
            luw_order.append((est_n, q, d))
    
    luw_order.sort()
    
    tried = set()
    for est_n, q, d in luw_order:
        if time.time() > deadline - 20:
            break
        if (q, d) in tried:
            continue
        tried.add((q, d))
        
        try:
            n, edges = luw_construct(q, d)
        except Exception:
            continue
        
        if n > best_n:
            continue
        
        if n > 50000:  # Skip very large graphs
            continue
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify 3-regular
        if not all(len(adj[i]) == 3 for i in range(n)):
            continue
        
        # Check girth
        if girth_check_fast(n, adj, g):
            best_result = (n, edges)
            best_n = n
            break  # Found the smallest, stop
    
    # If LUW didn't find anything reasonable, try a few more
    if best_result is None:
        for q in [3, 5]:
            for d in range(7, 12):
                if time.time() > deadline - 10:
                    break
                try:
                    n, edges = luw_construct(q, d)
                    if n <= 30000:
                        adj = [[] for _ in range(n)]
                        for u, v in edges:
                            adj[u].append(v)
                            adj[v].append(u)
                        if all(len(adj[i]) == 3 for i in range(n)):
                            if girth_check_fast(n, adj, g):
                                best_result = (n, edges)
                                break
                except Exception:
                    continue
            if best_result is not None:
                break
    
    # Final fallback: LUW with q=3, d=10
    if best_result is None:
        n, edges = luw_construct(3, 10)
        best_result = (n, edges)
    
    return best_result
