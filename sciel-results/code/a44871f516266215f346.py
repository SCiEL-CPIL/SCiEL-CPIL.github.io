"""Rewritten program: construct k-regular graph of girth g using random Cayley graphs."""
# EVOLVE-BLOCK-START
import random
import itertools
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Try to build a Cayley graph on a non-abelian group using involutory generators.
    If that fails within time, fall back to the Lazebnik-Ustimenko-Woldar construction.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    timeout = 115  # leave 5s buffer
    
    # Moore bound calculation
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # L-U-W construction as reliable fallback
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    # Verify girth using non-backtracking walks
    def verify_girth(n, edges, g):
        """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do non-backtracking BFS up to depth (g-1)//2
        # If we find a vertex reachable in fewer steps than expected, girth is violated
        max_depth = (g - 1) // 2
        
        for start in range(n):
            # State: (current_vertex, previous_vertex, depth)
            # Use BFS with non-backtracking constraint
            visited = set()
            queue = deque([(start, -1, 0)])
            visited.add((start, -1))
            
            while queue:
                u, prev, d = queue.popleft()
                if d >= max_depth:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue  # non-backtracking
                    if (v, u) in visited:
                        continue
                    visited.add((v, u))
                    queue.append((v, u, d + 1))
        
        # Now check for cycles shorter than g by looking at distances
        # Actually, let's use a more direct approach: check all pairs
        # For each vertex, find shortest cycle through it
        for start in range(n):
            # BFS from start, tracking parent to detect cycles
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        
        return True
    
    # Try random Cayley graph on (Z/2Z)^m for various m
    # For k=5, we need 5 involutions. In (Z/2Z)^m, every element is an involution.
    # The Cayley graph will be k-regular if we pick k distinct non-identity elements.
    # But we need to ensure girth >= g.
    
    def try_cayley_z2m(k, g, m, num_trials=5000):
        """Try to find a good set of k generators in (Z/2Z)^m."""
        n = 1 << m
        # Represent group elements as integers 0..n-1
        # XOR is the group operation
        
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                return None
            
            # Pick k random non-identity elements
            gens = set()
            while len(gens) < k:
                x = random.randint(1, n - 1)
                gens.add(x)
            gens = list(gens)
            
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for i in range(n):
                for gen in gens:
                    j = i ^ gen
                    adj[i].append(j)
            
            # Check girth
            # Quick check: verify no short cycles
            # For bipartite check first
            # Actually, let's do a proper girth check
            # Use BFS from each vertex to find shortest cycle
            
            # Optimized girth check
            max_depth = (g - 1) // 2
            valid = True
            
            for start in range(n):
                # Non-backtracking BFS
                # State: (vertex, prev)
                dist = {}
                queue = deque()
                for gen in gens:
                    v = start ^ gen
                    if v == start:
                        continue
                    dist[(v, start)] = 1
                    queue.append((v, start))
                
                while queue:
                    u, prev = queue.popleft()
                    d = dist[(u, prev)]
                    if d >= max_depth:
                        continue
                    for gen in gens:
                        v = u ^ gen
                        if v == prev:
                            continue
                        if (v, u) in dist:
                            continue
                        dist[(v, u)] = d + 1
                        queue.append((v, u))
            
            # Now check for actual short cycles
            # A cycle of length L < g means there's a non-backtracking walk of length L
            # that returns to start
            # Let's check by doing BFS and looking for back-edges
            
            for start in range(n):
                # Track visited states (vertex, prev) and their distance
                visited = {}
                queue = deque()
                
                # Initial: from start, go to each neighbor
                for gen in gens:
                    v = start ^ gen
                    state = (v, start)
                    if state not in visited:
                        visited[state] = 1
                        queue.append((v, start, 1))
                
                while queue:
                    u, prev, d = queue.popleft()
                    if d >= max_depth:
                        continue
                    for gen in gens:
                        v = u ^ gen
                        if v == prev:
                            continue
                        if v == start and d + 1 < g:
                            # Found a cycle of length d+1 < g
                            valid = False
                            break
                        state = (v, u)
                        if state in visited:
                            continue
                        visited[state] = d + 1
                        queue.append((v, u, d + 1))
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Build edge list
                edges = []
                for i in range(n):
                    for gen in gens:
                        j = i ^ gen
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        return None
    
    # Try Cayley graphs on (Z/2Z)^m for increasing m
    # For k=5, g=9, Moore bound is 1 + 5*(1+4+16+64) = 1 + 5*85 = 426
    # So we need at least 426 vertices, meaning m >= 9 (2^9 = 512)
    
    # But random Cayley graphs on (Z/2Z)^m may not have high girth
    # Let's try various m values
    
    best_result = None
    best_n = float('inf')
    
    # Try (Z/2Z)^m for m from 9 to 14
    for m in range(9, 15):
        if time.time() - start_time > timeout:
            break
        
        result = try_cayley_z2m(k, g, m, num_trials=2000)
        if result is not None:
            n, edges = result
            if n < best_n:
                best_n = n
                best_result = result
        
        if time.time() - start_time > timeout:
            break
    
    if best_result is not None:
        return best_result
    
    # Fall back to L-U-W construction
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
