"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def construct(k, g):
    # Petersen graph: 10 vertices, 3-regular, girth 5
    # We use a random lift by Z_m and then local search to eliminate short cycles
    
    # Petersen graph edges (standard labeling)
    base_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),  # outer cycle
        (5,7), (7,9), (9,6), (6,8), (8,5),  # inner star
        (0,5), (1,6), (2,7), (3,8), (4,9)   # spokes
    ]
    base_adj = [[] for _ in range(10)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
    
    edge_keys = [(min(u,v), max(u,v)) for u, v in base_edges]
    
    # Find all cycles of length < g in Petersen graph (these are the base cycles that can lift to short cycles)
    def find_all_cycles_up_to_len(adj, max_len):
        cycles = set()
        n_base = len(adj)
        for start in range(n_base):
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) > max_len:
                    continue
                for nb in adj[node]:
                    if nb == start and len(path) >= 3:
                        cycle = list(path)
                        min_idx = cycle.index(min(cycle))
                        norm = tuple(cycle[min_idx:] + cycle[:min_idx])
                        rev = list(reversed(norm))
                        rev_min_idx = rev.index(min(rev))
                        rev_norm = tuple(rev[rev_min_idx:] + rev[:rev_min_idx])
                        canon = min(norm, rev_norm)
                        cycles.add(canon)
                    elif nb not in path:
                        stack.append((nb, path + [nb]))
        return list(cycles)
    
    short_cycles = find_all_cycles_up_to_len(base_adj, g - 1)
    
    def cycle_edge_keys(cycle):
        keys = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            keys.append((min(u,v), max(u,v)))
        return keys
    
    cycle_keys_list = [cycle_edge_keys(c) for c in short_cycles]
    
    def build_lift(m, twists):
        """Build the lifted graph. twists is a dict mapping (u,v) to twist value."""
        n = 10 * m
        edges = []
        for u, v in base_edges:
            key = (min(u,v), max(u,v))
            t = twists.get(key, 0)
            for i in range(m):
                u_lift = u * m + i
                v_lift = v * m + (i + t) % m
                edges.append((min(u_lift, v_lift), max(u_lift, v_lift)))
        return n, edges
    
    def find_short_cycles(n, edges, max_len):
        """Find all cycles of length < max_len using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        short_cycles = []
        for start in range(n):
            # BFS with parent tracking
            dist = [-1] * n
            parent = [-1] * n
            queue = deque([start])
            dist[start] = 0
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < max_len:
                            # Reconstruct the cycle
                            path1 = []
                            cur = u
                            while cur != -1:
                                path1.append(cur)
                                cur = parent[cur]
                            path1.reverse()
                            
                            path2 = []
                            cur = v
                            while cur != -1:
                                path2.append(cur)
                                cur = parent[cur]
                            path2.reverse()
                            
                            # Find LCA
                            set1 = set(path1)
                            lca = path2[-1]
                            for node in path2:
                                if node in set1:
                                    lca = node
                                    break
                            
                            cycle = path1[path1.index(lca):] + path2[path2.index(lca):][::-1]
                            # Normalize
                            min_idx = cycle.index(min(cycle))
                            norm = tuple(cycle[min_idx:] + cycle[:min_idx])
                            rev = list(reversed(norm))
                            rev_min_idx = rev.index(min(rev))
                            rev_norm = tuple(rev[rev_min_idx:] + rev[:rev_min_idx])
                            canon = min(norm, rev_norm)
                            short_cycles.append(canon)
        
        return list(set(short_cycles))
    
    def check_girth(n, edges, target_g):
        """Check if girth >= target_g."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = deque([start])
            dist[start] = 0
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_g:
                            return False
        return True
    
    # Try different m values and random twist assignments
    best_result = None
    best_n = float('inf')
    
    # Start with reasonable m values
    for m in range(50, 100, 1):
        n = 10 * m
        if n >= best_n:
            continue
        
        # Try random twist assignments
        for attempt in range(100):
            random.seed(42 + m * 1000 + attempt)
            twists = {key: random.randint(0, m-1) for key in edge_keys}
            
            # Check constraints: for each short base cycle, sum of twists != 0 mod m
            valid = True
            for keys in cycle_keys_list:
                s = sum(twists[key] for key in keys) % m
                if s == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Build and check girth
            n_lift, edges = build_lift(m, twists)
            if check_girth(n_lift, edges, g):
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW-like construction
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
# EVOLVE-BLOCK-END
