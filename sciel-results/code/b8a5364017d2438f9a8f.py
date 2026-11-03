import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    def luw_construction(q, d):
        # Define field operations
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

    def verify_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        for L in range(3, g):
            for start in range(n):
                visited = set()
                stack = [(start, -1, 0)]
                while stack:
                    v, prev, l = stack.pop()
                    if l == L:
                        if v == start:
                            return False
                        continue
                    for w in adj[v]:
                        if w != prev:
                            if (v, prev, l) not in visited:
                                visited.add((v, prev, l))
                                stack.append((w, v, l + 1))
        return True

    def find_short_cycle(n, adj, max_len):
        """Find a cycle of length <= max_len using BFS from each vertex."""
        for start in range(n):
            # BFS to find shortest cycle through start
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
                    elif v != parent[u] and dist[v] <= dist[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len <= max_len and cycle_len >= 3:
                            # Reconstruct cycle
                            cycle = [u]
                            while cycle[-1] != v:
                                cycle.append(parent[cycle[-1]])
                            cycle.reverse()
                            return cycle
        return None

    def random_regular_graph(n, degree, seed=42):
        """Generate a random d-regular graph using configuration model with rejection."""
        random.seed(seed)
        if n * degree % 2 != 0:
            return None
        
        # Create stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * degree)
        
        max_attempts = 100
        for _ in range(max_attempts):
            random.shuffle(stubs)
            edges = []
            seen = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append((u, v))
            
            if valid:
                return edges
        
        return None

    def edge_swap_improve(n, edges, g, time_limit):
        """Iteratively swap edges to eliminate short cycles."""
        end_time = time.time() + time_limit
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u, v), max(u, v)))
        
        # Find initial short cycle
        cycle = find_short_cycle(n, adj, g - 1)
        if cycle is None:
            return n, edges, True  # Already girth >= g
        
        # Try to break the cycle by swapping an edge
        # For a cycle v0-v1-...-v_{L-1}-v0, we can swap edge (vi, vi+1)
        # with some other edge to break the cycle
        L = len(cycle)
        
        # Collect all edges not in the cycle
        cycle_edges = set()
        for i in range(L):
            u, v = cycle[i], cycle[(i+1) % L]
            cycle_edges.add((min(u, v), max(u, v)))
        
        non_cycle_edges = [(u, v) for (u, v) in edge_set if (u, v) not in cycle_edges]
        
        # Try swapping each edge in the cycle with a non-cycle edge
        for i in range(L):
            u, v = cycle[i], cycle[(i+1) % L]
            # Try swapping (u, v) with some (a, b) where a,b are not adjacent to u or v
            # This is complex; instead, try a simpler approach:
            # Remove edge (u,v) and add edges (u,a) and (v,b) where (a,b) was an edge
            # and a,b are chosen to avoid creating new short cycles
            
            # For simplicity, try a few random swaps
            if time.time() > end_time:
                break
                
            for _ in range(min(10, len(non_cycle_edges))):
                if time.time() > end_time:
                    break
                a, b = random.choice(non_cycle_edges)
                
                # Check if swap is valid:
                # 1. u != a, u != b, v != a, v != b
                # 2. (u,a) and (v,b) are not already edges
                # 3. No loops
                if u == a or u == b or v == a or v == b:
                    continue
                if (min(u, a), max(u, a)) in edge_set:
                    continue
                if (min(v, b), max(v, b)) in edge_set:
                    continue
                
                # Perform swap: remove (u,v) and (a,b), add (u,a) and (v,b)
                # But we need to be careful: we're swapping one edge for two?
                # No, edge swap means: remove (u,v) and (a,b), add (u,a) and (v,b)
                # This maintains degree if we're careful.
                
                # Actually, a proper 2-switch: remove (u,v) and (a,b), add (u,a) and (v,b)
                # or remove (u,v) and (a,b), add (u,b) and (v,a)
                
                # Try both options
                for option in range(2):
                    if option == 0:
                        new_edges = [(u, a), (v, b)]
                    else:
                        new_edges = [(u, b), (v, a)]
                    
                    # Check validity
                    valid = True
                    for (x, y) in new_edges:
                        if x == y:
                            valid = False
                            break
                        if (min(x, y), max(x, y)) in edge_set and (min(x, y), max(x, y)) != (min(u, v), max(u, v)) and (min(x, y), max(x, y)) != (min(a, b), max(a, b)):
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Apply swap and check girth
                    old_edges = [(u, v), (a, b)]
                    new_edge_set = edge_set.copy()
                    for (x, y) in old_edges:
                        new_edge_set.discard((min(x, y), max(x, y)))
                    for (x, y) in new_edges:
                        new_edge_set.add((min(x, y), max(x, y)))
                    
                    # Build new adjacency
                    new_adj = [[] for _ in range(n)]
                    for (x, y) in new_edge_set:
                        new_adj[x].append(y)
                        new_adj[y].append(x)
                    
                    # Check if girth improved
                    new_cycle = find_short_cycle(n, new_adj, g - 1)
                    if new_cycle is None or (cycle is not None and len(new_cycle) > L):
                        # Accept the swap
                        edge_set = new_edge_set
                        adj = new_adj
                        # Rebuild edges list
                        edges = list(edge_set)
                        cycle = new_cycle
                        break
                else:
                    continue
                break
        
        # Continue until no short cycle or time runs out
        while cycle is not None and time.time() < end_time:
            L = len(cycle)
            cycle_edges = set()
            for i in range(L):
                u, v = cycle[i], cycle[(i+1) % L]
                cycle_edges.add((min(u, v), max(u, v)))
            
            non_cycle_edges = [(u, v) for (u, v) in edge_set if (u, v) not in cycle_edges]
            
            found_swap = False
            for i in range(L):
                u, v = cycle[i], cycle[(i+1) % L]
                
                for _ in range(min(5, len(non_cycle_edges))):
                    if time.time() > end_time:
                        return n, edges, False
                    a, b = random.choice(non_cycle_edges)
                    
                    if u == a or u == b or v == a or v == b:
                        continue
                    if (min(u, a), max(u, a)) in edge_set:
                        continue
                    if (min(v, b), max(v, b)) in edge_set:
                        continue
                    
                    for option in range(2):
                        if option == 0:
                            new_edges = [(u, a), (v, b)]
                        else:
                            new_edges = [(u, b), (v, a)]
                        
                        valid = True
                        for (x, y) in new_edges:
                            if x == y:
                                valid = False
                                break
                            key = (min(x, y), max(x, y))
                            if key in edge_set and key != (min(u, v), max(u, v)) and key != (min(a, b), max(a, b)):
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        old_edges = [(u, v), (a, b)]
                        new_edge_set = edge_set.copy()
                        for (x, y) in old_edges:
                            new_edge_set.discard((min(x, y), max(x, y)))
                        for (x, y) in new_edges:
                            new_edge_set.add((min(x, y), max(x, y)))
                        
                        new_adj = [[] for _ in range(n)]
                        for (x, y) in new_edge_set:
                            new_adj[x].append(y)
                            new_adj[y].append(x)
                        
                        new_cycle = find_short_cycle(n, new_adj, g - 1)
                        if new_cycle is None or len(new_cycle) > L:
                            edge_set = new_edge_set
                            adj = new_adj
                            edges = list(edge_set)
                            cycle = new_cycle
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    break
            
            if not found_swap:
                break
        
        return n, edges, (cycle is None)

    best_n = float('inf')
    best_edges = None
    
    # First try LUW construction
    candidates = []
    for d in [4, 5, 6]:
        for q in [3, 5, 7, 4]:
            candidates.append((q, d))
    candidates.sort(key=lambda x: 2 * (x[0] ** x[1]))
    
    for q, d in candidates:
        if time.time() > deadline - 30:
            break
        try:
            n, edges = luw_construction(q, d)
            if n < best_n:
                if verify_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
        except Exception:
            continue
    
    # Now try random walk extension to potentially find smaller graphs
    # Start from near Moore bound
    # Moore bound for k=4, g=10: 1 + k * (1 + (k-1) + ... + (k-1)^{g/2-1}) / 2
    # For even g: 1 + k * sum_{i=0}^{g/2-1} (k-1)^i
    # = 1 + 4 * (1 + 3 + 9 + 27 + 81) = 1 + 4 * 121 = 485? 
    # Wait, let me recalculate: g/2 = 5, so sum from i=0 to 4: 1+3+9+27+81 = 121
    # 1 + 4*121 = 485. But the problem says lower bound is 243.
    # Actually for even girth g, Moore bound is: 1 + k * sum_{i=0}^{g/2-1} (k-1)^i
    # For g=10: g/2-1 = 4, sum = 1+3+9+27+81 = 121, 1+4*121 = 485
    # But known lower bound is 243, so maybe I'm misremembering.
    
    # Let's try n values from 243 up to best_n
    if best_n == float('inf'):
        best_n = 2048  # LUW with q=4, d=5
        best_edges = luw_construction(4, 5)[1]
    
    # Try random graphs with n from 243 to best_n-1
    for n in range(243, min(best_n, 400)):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initial graphs
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            edges = random_regular_graph(n, k, seed=seed + 1000)
            if edges is None:
                continue
            
            # Check if it's already valid
            if verify_girth(n, edges, g):
                best_n = n
                best_edges = edges
                break
            
            # Try to improve with edge swaps
            n2, edges2, success = edge_swap_improve(n, edges, g, time_limit=5)
            if success and verify_girth(n, edges2, g):
                best_n = n
                best_edges = edges2
                break
    
    if best_edges is None:
        n, edges = luw_construction(4, 5)
        return n, edges
        
    return best_n, best_edges
