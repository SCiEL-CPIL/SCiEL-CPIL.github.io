import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    if k == 4 and g == 10:
        # Try to construct a smaller graph using random regular graph + 2-switches
        # Moore bound for k=4, g=10 is 243.
        # Known best is 320.
        # We try N from 243 to 400.
        for n_attempt in range(243, 401):
            if time.time() > deadline - 10:
                break
            try:
                n, edges = _random_regular_high_girth(k, g, n_attempt, deadline)
                if n is not None:
                    return n, edges
            except Exception:
                pass
        
        # If random construction failed or took too long, fall back to LUW
        return _luw_graph(k, g)
    else:
        # For other parameters, use LUW as a safe fallback
        return _luw_graph(k, g)

def _luw_graph(k, g):
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

def _random_regular_high_girth(k, g, n, deadline):
    # Generate a random k-regular graph and remove short cycles
    # 1. Create half-edges
    half_edges = []
    for i in range(n):
        for j in range(k):
            half_edges.append(i)
    
    # Shuffle and pair
    random.shuffle(half_edges)
    adj = [set() for _ in range(n)]
    # We will use a list of lists for adjacency to allow duplicates initially, 
    # then convert to set or check for multi-edges.
    # Actually, for 2-switches, it's easier to track the specific edges.
    # Let's represent the graph as a list of lists of neighbors, allowing duplicates.
    # But we need simple graph eventually.
    
    # Better approach: 
    # 1. Random pairing.
    # 2. Check for multi-edges and loops.
    # 3. Use 2-switches to fix.
    
    # Initialize edges as list of (u, v)
    edges = []
    for i in range(0, len(half_edges), 2):
        u = half_edges[i]
        v = half_edges[i+1]
        edges.append((u, v))
        
    # Convert to adjacency list (multiset)
    # We need to be careful with 2-switches on multisets.
    # Let's try to get a simple graph first.
    
    # Function to check if graph is simple and has girth >= g
    def is_valid_simple_graph(edges, n, g):
        # Check loops
        for u, v in edges:
            if u == v:
                return False
        # Check multi-edges
        seen = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            if (u, v) in seen:
                return False
            seen.add((u, v))
        # Check girth
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # BFS from each vertex to find shortest cycle
        # For girth g, we only need to check up to depth g//2 - 1? 
        # Actually, BFS from v, if we find a neighbor of v at depth d, cycle length is d+1.
        # Max depth to check is (g-1)//2.
        max_depth = (g - 1) // 2
        for start in range(n):
            visited = {start: 0}
            queue = deque([(start, 0)])
            while queue:
                u, d = queue.popleft()
                if d == max_depth:
                    continue
                for v in adj[u]:
                    if v == start and d + 1 < g:
                        return False
                    if v not in visited:
                        visited[v] = d + 1
                        queue.append((v, d + 1))
                    else:
                        # If we see a visited node, it's a back-edge or cross-edge.
                        # In undirected BFS tree, if v is visited and v != parent(u), 
                        # cycle length is d + 1 + (dist[start][v] + 1)? No.
                        # Cycle length = dist(start, u) + dist(start, v) + 1.
                        # Since we are doing BFS from start, dist(start, u) = d.
                        # dist(start, v) is visited[v].
                        # If v is in visited and v != start, and the edge (u,v) is not a tree edge?
                        # In simple graph, if we encounter a visited node, it forms a cycle.
                        # Cycle length = d + 1 + visited[v]
                        # Wait, BFS tree edge is (u, parent). If v is parent, it's not a cycle.
                        # We don't track parent here.
                        # Let's just check if d + 1 + visited[v] < g.
                        # But we need to ensure (u,v) is not the edge we came from.
                        # This simple BFS check is tricky without parent pointers.
                        # Alternative: Just run BFS and if we reach depth max_depth without finding a cycle back to start?
                        # No, we need to detect ANY cycle < g.
                        
                        # Let's use a simpler check: 
                        # If we find a node v that is already in visited, and the path length to v is less than g?
                        # Actually, standard girth check:
                        # For each edge (u,v), if dist(u,v) in graph without (u,v) is < g-1? No.
                        
                        # Let's stick to: 
                        # For each start, BFS. If we find a neighbor w of u such that w is visited 
                        # and w is not the parent of u, then cycle length = dist[u] + dist[w] + 1.
                        # We need to store parent.
                        
                        pass # Logic too complex for inline, let's use a helper
                
        # Let's use a dedicated girth checker
        if _girth_check(adj, g) < g:
            return False
        return True

    def _girth_check(adj, g):
        n = len(adj)
        min_cycle = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] * 2 >= min_cycle: # Pruning
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < g:
                                return min_cycle
        return min_cycle

    # Main loop: Random restarts and 2-switches
    # We will try to fix the graph iteratively.
    # If we find a simple graph with girth >= g, return it.
    
    max_iter = 5000
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            return None, None
        
        # Check if current graph is valid
        # First, check for loops and multi-edges
        is_simple = True
        edge_set = set()
        for u, v in edges:
            if u == v:
                is_simple = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                is_simple = False
                break
            edge_set.add((u, v))
        
        if not is_simple:
            # Try to fix with 2-switches
            # Find a "bad" edge pair
            # Loop: (u, u). Swap with (v, w) -> (u, v), (u, w)
            # Multi-edge: (u, v) and (u, v). Swap one with (x, y) -> (u, x), (v, y)
            
            # Find a loop
            loop_idx = -1
            for i, (u, v) in enumerate(edges):
                if u == v:
                    loop_idx = i
                    break
            
            if loop_idx != -1:
                u = edges[loop_idx][0]
                # Find another edge (a, b) to swap with
                # We need (a, b) such that a != u and b != u (to avoid new loops/multiedges easily)
                # Actually, swapping (u, u) with (a, b) gives (u, a) and (u, b).
                # If a==u or b==u, we might create multi-edges or loops.
                # Let's just pick a random other edge.
                other_idx = random.choice([i for i in range(len(edges)) if i != loop_idx])
                a, b = edges[other_idx]
                # Perform swap
                edges[loop_idx] = (u, a)
                edges[other_idx] = (u, b)
                continue
            
            # Find multi-edge
            edge_count = {}
            for u, v in edges:
                if u > v: u, v = v, u
                edge_count[(u, v)] = edge_count.get((u, v), 0) + 1
            
            multi_edge = None
            for e, count in edge_count.items():
                if count > 1:
                    multi_edge = e
                    break
            
            if multi_edge:
                u, v = multi_edge
                # Find two edges (u, v) in the list
                idx1, idx2 = -1, -1
                for i, (a, b) in enumerate(edges):
                    if (a, b) == (u, v) or (a, b) == (v, u):
                        if idx1 == -1:
                            idx1 = i
                        else:
                            idx2 = i
                            break
                
                # Pick a random other edge (x, y)
                other_idx = random.choice([i for i in range(len(edges)) if i not in (idx1, idx2)])
                x, y = edges[other_idx]
                
                # Swap: (u, v) and (x, y) -> (u, x) and (v, y)
                # We need to be careful with orientation.
                # Original: (u, v) at idx1, (v, u) at idx2 (or similar)
                # Let's just replace idx1 with (u, x) and other_idx with (v, y)?
                # No, 2-switch: (u, v) and (x, y) becomes (u, x) and (v, y).
                # We are removing one (u, v) and one (x, y).
                # We add (u, x) and (v, y).
                # The other (u, v) remains.
                
                edges[idx1] = (u, x)
                edges[other_idx] = (v, y)
                continue
            
            # If no loops or multi-edges, check girth
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            girth = _girth_check(adj, g)
            if girth >= g:
                # Return list of edges
                return n, edges
            
            # If girth < g, find a short cycle and break it
            # Find a cycle of length < g
            # We can use the _girth_check logic to return the cycle?
            # For now, let's just do a random 2-switch.
            i = random.randint(0, len(edges) - 1)
            j = random.randint(0, len(edges) - 1)
            if i == j:
                continue
            u1, v1 = edges[i]
            u2, v2 = edges[j]
            
            # Avoid creating loops
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
            
            # Perform switch: (u1, v1) and (u2, v2) -> (u1, u2) and (v1, v2)
            edges[i] = (u1, u2)
            edges[j] = (v1, v2)
            
    return None, None
