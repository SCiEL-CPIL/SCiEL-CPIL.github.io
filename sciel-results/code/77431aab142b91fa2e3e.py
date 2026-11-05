import random
import time
from collections import deque

def construct(k, g):
    # Local search: start from random k-regular graph, eliminate short cycles
    
    def find_short_cycles(adj, n, max_len):
        """Find all cycles of length < max_len"""
        cycles = []
        for start in range(n):
            # BFS to find short cycles
            visited = [[False] * n for _ in range(max_len)]
            queue = deque([(start, start, 0, [start])])
            while queue:
                prev, curr, depth, path = queue.popleft()
                if depth >= max_len:
                    continue
                for nei in adj[curr]:
                    if nei == prev:
                        continue
                    if nei == start and depth + 1 >= 3:
                        # Found a cycle
                        cycle = path + [start]
                        # Normalize: rotate so smallest vertex is first
                        min_idx = cycle.index(min(cycle))
                        norm = tuple(cycle[min_idx:] + cycle[:min_idx])
                        if norm not in [tuple(c) for c in cycles]:
                            cycles.append(norm)
                    elif not visited[depth + 1][nei]:
                        visited[depth + 1][nei] = True
                        queue.append((curr, nei, depth + 1, path + [nei]))
        return cycles
    
    def has_short_cycle(adj, n, g):
        """Check if graph has any cycle shorter than g"""
        for start in range(n):
            visited = [[False] * n for _ in range(g - 1)]
            queue = deque([(start, start, 0)])
            while queue:
                prev, curr, depth = queue.popleft()
                if depth >= g - 1:
                    continue
                for nei in adj[curr]:
                    if nei == prev:
                        continue
                    if nei == start and depth + 1 >= 3:
                        return True
                    if not visited[depth + 1][nei]:
                        visited[depth + 1][nei] = True
                        queue.append((curr, nei, depth + 1))
        return False
    
    def random_regular_graph(n, k):
        """Generate a random k-regular graph"""
        if n * k % 2 != 0:
            return None
        # Use configuration model
        half_edges = list(range(n * k))
        for i in range(n):
            half_edges.extend([i] * k)
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                return None  # Self-loop, retry
            if v in adj[u]:
                return None  # Multi-edge, retry
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def try_local_search(n, k, g, timeout=10):
        start_time = time.time()
        max_iter = 50000
        
        for attempt in range(50):
            if time.time() - start_time > timeout:
                break
            
            adj = random_regular_graph(n, k)
            if adj is None:
                continue
            
            # Check initial girth
            if not has_short_cycle(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            
            # Local search: try to eliminate short cycles via edge swaps
            for it in range(max_iter):
                if time.time() - start_time > timeout:
                    break
                
                # Find a short cycle
                short_cycle = None
                for start in range(n):
                    visited = [[False] * n for _ in range(g - 1)]
                    queue = deque([(start, start, 0, [start])])
                    while queue:
                        prev, curr, depth, path = queue.popleft()
                        if depth >= g - 1:
                            continue
                        for nei in adj[curr]:
                            if nei == prev:
                                continue
                            if nei == start and depth + 1 >= 3:
                                short_cycle = path + [start]
                                break
                            if not visited[depth + 1][nei]:
                                visited[depth + 1][nei] = True
                                queue.append((curr, nei, depth + 1, path + [nei]))
                        if short_cycle:
                            break
                    if short_cycle:
                        break
                
                if short_cycle is None:
                    # No short cycles found
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                
                # Try to break the cycle via edge swap
                # Pick two edges (a,b) and (c,d) not in the cycle, swap to (a,c) and (b,d)
                # This is complex; simpler: pick a random edge in the cycle and try to rewire one endpoint
                cycle_edges = set()
                for i in range(len(short_cycle) - 1):
                    u, v = short_cycle[i], short_cycle[i + 1]
                    cycle_edges.add((min(u, v), max(u, v)))
                
                # Pick a random edge in the cycle to rewire
                edge_list = list(cycle_edges)
                if not edge_list:
                    continue
                u, v = random.choice(edge_list)
                
                # Try to find a new partner for u
                candidates = [x for x in adj[u] if x != v and x not in adj[v]]
                if not candidates:
                    continue
                new_v = random.choice(candidates)
                
                # Check if rewiring maintains girth (simplified: just do it and check later)
                # Remove old edge, add new edge
                adj[u].remove(v)
                adj[v].remove(u)
                adj[u].append(new_v)
                adj[new_v].append(u)
        
        return None
    
    # For k=5, g=9, try local search with various n
    if k == 5 and g == 9:
        n_trials = [500, 600, 700, 800, 900, 1000]
        for n in n_trials:
            res = try_local_search(n, k, g, timeout=15)
            if res:
                return res
    
    # Fallback: simple construction
    # Use a known small construction or fail gracefully
    # For now, return a valid but possibly larger graph
    # We'll use a simple cyclic construction
    
    # Cyclic construction: vertices 0..n-1, edges (i, i+d) mod n for various d
    # This won't have high girth, so we need a better fallback
    
    # Use a random regular graph and hope for the best (unlikely to work)
    # Better: return None to indicate failure? No, must return valid graph.
    
    # Last resort: construct a graph that definitely has girth >= g
    # A tree has infinite girth, but we need k-regular. 
    # A k-regular tree doesn't exist as a finite graph.
    
    # Use the L-U-W construction as ultimate fallback
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
    
    return l_u_w_construct(k, g)
