import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def _deadline_check():
        return time.time() > deadline

    def verify_girth(adj, g):
        """Verify girth >= g using non-backtracking walks from every vertex."""
        n = len(adj)
        if n == 0:
            return True
        # For girth g, we need no cycle of length < g
        # Non-backtracking: walk of length L with no immediate backtracking
        # A cycle of length L exists iff there's a non-backtracking walk of length L
        # that returns to start (with the constraint that it's a simple cycle)
        # Simpler: BFS with parent tracking up to depth (g-1)//2
        # If any node is reached at depth d from u where d < g, and it's not the parent,
        # we have a cycle.
        
        radius = (g - 1) // 2
        for u in range(n):
            # BFS from u, tracking parent to avoid backtracking
            visited = {}
            visited[u] = (-1, 0)  # (parent, depth)
            queue = deque([(u, -1, 0)])
            while queue:
                node, parent, dist = queue.popleft()
                if dist == radius:
                    continue
                for nei in adj[node]:
                    if nei == parent:
                        continue  # non-backtracking
                    if nei in visited:
                        # Found a cycle
                        return False
                    visited[nei] = (node, dist + 1)
                    queue.append((nei, node, dist + 1))
        return True

    def luw_construct(k, g):
        """LUW construction for 3-regular graphs."""
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

    # Get LUW baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # Moore bound for k=3, g=16: 2 * (3 * (2^7 - 1)/(2-1)) + 2 = 2*384 + 2 = 770
    # Actually: for k-regular, girth g even: n >= 2*k*(k-1)^((g/2)-1)
    # For k=3, g=16: n >= 2*3*2^7 = 1536? No wait.
    # Moore bound: n >= k*(k-1)^(d-1) + 1 where d = g/2 - 1 for even girth
    # For g=16, d=7: n >= 3*2^6 + 1 = 193? That's not right either.
    # Correct Moore bound for k-regular graph with girth g:
    # If g is even: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * ((k-1)^(g/2) - 1)/(k-2)
    # For k=3, g=16: n >= 2 * (2^8 - 1) = 2 * 255 = 510
    # So Moore bound is 510.
    
    moore_bound = 2 * ((k - 1) ** (g // 2) - 1) // (k - 2) if k > 2 else 1
    search_start = min(best_n - 1, 935)
    search_end = max(moore_bound, 510)

    if search_start < search_end:
        # Try algebraic constructions: use voltage graphs
        # A voltage graph on a base graph with voltage group Z_m gives a cover
        # of size m * |V_base|
        
        # Strategy: try various base graphs and voltage assignments
        # For 3-regular, base must be 3-regular
        
        # Simpler approach: try to construct using known algebraic methods
        # For now, focus on optimizing the random search with better time allocation
        
        # Allocate time: 60% to algebraic/structured, 40% to random
        time_for_structured = 60
        time_for_random = 45
        
        # Structured search: try specific n values with algebraic constructions
        # Try n values that are multiples of small numbers (for voltage graph covers)
        candidate_ns = []
        for n in range(search_start, search_end - 1, -1):
            candidate_ns.append(n)
            if len(candidate_ns) >= 50:
                break
        
        # Try each candidate with a few random graphs, prioritizing smaller n
        for n in candidate_ns:
            if _deadline_check():
                break
            if time.time() > start_time + time_for_structured + 10:
                break
            
            # Try up to 10 random graphs for this n
            max_trials = 10
            for trial in range(max_trials):
                if _deadline_check():
                    break
                if time.time() > start_time + time_for_structured + 10:
                    break
                    
                adj = _generate_random_3_regular(n)
                if adj is None:
                    continue
                if verify_girth(adj, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges

        # Random search for remaining time
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            # Try random n values between search_end and search_start
            for _ in range(200):
                if _deadline_check():
                    break
                n = random.randint(search_end, search_start)
                adj = _generate_random_3_regular(n)
                if adj is None:
                    continue
                if verify_girth(adj, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return best_n, best_edges

def _generate_random_3_regular(n):
    """Generate a random 3-regular graph using configuration model."""
    if n < 3 or (3 * n) % 2 != 0:
        return None
    
    # Configuration model
    half_edges = list(range(3 * n))
    random.shuffle(half_edges)
    
    adj = [[] for _ in range(n)]
    valid = True
    for i in range(0, 3 * n, 2):
        u = half_edges[i] // 3
        v = half_edges[i + 1] // 3
        
        if u == v:
            valid = False
            break
        
        if v in adj[u]:
            valid = False
            break
        
        adj[u].append(v)
        adj[v].append(u)
    
    if not valid:
        return None
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
        if len(set(adj[i])) != 3:
            return None
    
    return adj
