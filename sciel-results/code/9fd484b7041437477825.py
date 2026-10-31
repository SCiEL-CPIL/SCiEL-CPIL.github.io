"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def construct(k, g):
    # Construct 3-regular graph with girth >= g using Petersen graph lift + local search
    
    if k != 3:
        # Fallback for non-3-regular: use LUW-like construction
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
    
    # Petersen graph: 10 vertices, 3-regular, girth 5
    # Outer cycle: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
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
    
    # Find all cycles of length < g in Petersen graph (up to length g-1)
    def find_all_cycles_up_to_len(adj, max_len):
        cycles = set()
        n_base = len(adj)
        for start in range(n_base):
            # DFS for cycles
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) > max_len:
                    continue
                for nb in adj[node]:
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        cycle = list(path)
                        # Normalize: start from min vertex
                        min_idx = cycle.index(min(cycle))
                        norm = cycle[min_idx:] + cycle[:min_idx]
                        rev = list(reversed(norm))
                        rev_min_idx = rev.index(min(rev))
                        rev_norm = rev[rev_min_idx:] + rev[:rev_min_idx]
                        canon = tuple(min(norm, rev_norm))
                        cycles.add(canon)
                    elif nb not in path:
                        stack.append((nb, path + [nb]))
        return list(cycles)
    
    short_cycles = find_all_cycles_up_to_len(base_adj, g - 1)
    
    # For each cycle, get the sequence of edge keys with signs
    # We need to track the direction of traversal
    def cycle_edge_sequence(cycle):
        """Return list of (edge_key, sign) where sign is +1 or -1 based on direction."""
        seq = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            key = (min(u,v), max(u,v))
            # Sign: +1 if u < v in the key order, -1 otherwise
            sign = 1 if u == key[0] else -1
            seq.append((key, sign))
        return seq
    
    cycle_seqs = [cycle_edge_sequence(c) for c in short_cycles]
    
    def build_lift(m, twists):
        """Build the lifted graph."""
        n = 10 * m
        edges = []
        for u, v in base_edges:
            key = (min(u,v), max(u,v))
            t = twists[key]
            for i in range(m):
                u_lift = u * m + i
                v_lift = v * m + (i + t) % m
                edges.append((min(u_lift, v_lift), max(u_lift, v_lift)))
        return n, edges
    
    def compute_girth(n, edges, target_g):
        """Check if girth >= target_g using non-backtracking BFS."""
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
                            return False, cycle_len
        return True, float('inf')
    
    def find_short_cycles(n, edges, max_len):
        """Find all cycles of length <= max_len."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        found = []
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
                        if cycle_len <= max_len:
                            # Reconstruct cycle
                            cycle = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                cycle.append(cur)
                            cycle.append(v)
                            # Normalize
                            min_idx = cycle.index(min(cycle))
                            norm = cycle[min_idx:] + cycle[:min_idx]
                            rev = list(reversed(norm))
                            rev_min_idx = rev.index(min(rev))
                            rev_norm = rev[rev_min_idx:] + rev[:rev_min_idx]
                            canon = tuple(min(norm, rev_norm))
                            if canon not in found:
                                found.append(canon)
        return found
    
    # Strategy: try different m values, find twist assignments that avoid all short cycles
    # Then use local search to fix any remaining issues
    
    best_result = None
    best_n = float('inf')
    
    # Try m values that give reasonable graph sizes
    # Known best is 936, so try m around 90-100
    for m in [80, 85, 90, 95, 100, 94, 96, 98, 102, 105]:
        n = 10 * m
        if n >= best_n:
            continue
        
        # Try multiple random twist assignments
        found_valid = False
        for attempt in range(200):
            random.seed(42 + m * 1000 + attempt)
            twists = {key: random.randint(0, m-1) for key in edge_keys}
            
            # Check constraints: for each short cycle, signed sum of twists != 0 mod m
            valid = True
            for seq in cycle_seqs:
                s = sum(sign * twists[key] for key, sign in seq) % m
                if s == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Build and verify girth
            n_lift, edges = build_lift(m, twists)
            ok, girth_val = compute_girth(n_lift, edges, g)
            if ok:
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                found_valid = True
                break
        
        if found_valid:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW construction
    q = 3
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
