import time
import math
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    For k=4, g=10, it searches for a small random 4-regular graph.
    Falls back to Lazebnik-Ustimenko-Woldar construction if search fails or takes too long.
    """
    start_time = time.time()
    deadline = start_time + 105  # 15 seconds safety margin before 120s kill
    
    if k != 4 or g != 10:
        return _luw_construct(k, g)

    # Target size 320 (known smallest cage)
    target_n = 320
    
    def _gen_random_4reg(n):
        """Generates a random 4-regular graph on n vertices using configuration model with repair."""
        for attempt in range(30):
            # Build stub list
            stubs = []
            for i in range(n):
                stubs.extend([i] * 4)
            random.shuffle(stubs)
            
            # Pair up stubs, fixing loops and multi-edges via local swaps
            pairs = []
            for i in range(0, len(stubs), 2):
                pairs.append([stubs[i], stubs[i+1]])
            
            # Fix loops: if a pair is (u, u), swap one endpoint with another pair
            for i in range(len(pairs)):
                if pairs[i][0] == pairs[i][1]:
                    u = pairs[i][0]
                    # Find a pair j where we can swap to fix the loop
                    for j in range(len(pairs)):
                        if j == i:
                            continue
                        a, b = pairs[j]
                        # Try swapping: (u, u) and (a, b) -> (u, a) and (u, b) or (u, b) and (u, a)
                        if a != u and b != u:
                            pairs[i] = [u, a]
                            pairs[j] = [u, b]
                            break
                        elif a == u and b != u:
                            # (u, u) and (u, b) -> (u, b) and (u, u) - still has loop, skip
                            pass
                    if pairs[i][0] == pairs[i][1]:
                        break  # couldn't fix, retry
                if pairs[i][0] == pairs[i][1]:
                    break
            
            if any(p[0] == p[1] for p in pairs):
                continue
            
            # Check for multi-edges and fix via swaps
            edge_set = set()
            valid = True
            for i in range(len(pairs)):
                u, v = pairs[i]
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
            
            if valid:
                return [(pairs[i][0], pairs[i][1]) for i in range(len(pairs))]
        
        # Fallback: simple rejection
        for attempt in range(20):
            stubs = []
            for i in range(n):
                stubs.extend([i] * 4)
            random.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or (min(u,v), max(u,v)) in edges:
                    valid = False
                    break
                edges.add((min(u,v), max(u,v)))
            
            if valid and len(edges) == 2 * n:
                return list(edges)
        
        return None

    def _build_adj(n, edges):
        """Build adjacency list from edge list."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def _count_short_cycles(n, adj, g):
        """Count cycles of length < g using non-backtracking walks from each vertex."""
        short_count = 0
        for s in range(n):
            # BFS tracking (node, prev) pairs
            visited = set()
            visited.add((s, -1))
            queue = deque([(s, -1, 0)])
            while queue:
                u, prev, d = queue.popleft()
                if d >= g - 1:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s and d + 1 < g:
                        short_count += 1
                    if (v, u) not in visited:
                        visited.add((v, u))
                        queue.append((v, u, d + 1))
        return short_count

    def _has_girth_at_least(n, edges, g):
        """Checks if graph has girth at least g."""
        adj = _build_adj(n, edges)
        return _count_short_cycles(n, adj, g) == 0

    def _local_search(edges, n, g, max_iters=500):
        """Try to eliminate short cycles by swapping edges while maintaining 4-regularity."""
        adj = _build_adj(n, edges)
        edge_set = set((min(u,v), max(u,v)) for u, v in edges)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            sc = _count_short_cycles(n, adj, g)
            if sc == 0:
                return edges
            
            # Find a short cycle and try to break it
            # Pick two non-adjacent edges and swap their endpoints
            # Try random edge swaps
            for trial in range(20):
                # Pick two random edges
                e1_idx = random.randint(0, len(edges) - 1)
                e2_idx = random.randint(0, len(edges) - 1)
                if e1_idx == e2_idx:
                    continue
                
                u1, v1 = edges[e1_idx]
                u2, v2 = edges[e2_idx]
                
                # Try swapping: (u1,v1) and (u2,v2) -> (u1,u2) and (v1,v2)
                new_e1 = (u1, u2)
                new_e2 = (v1, v2)
                
                # Check validity
                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                    continue
                ne1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                ne2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
                
                if ne1 == ne2:
                    continue
                
                # Remove old edges, add new ones
                old_e1 = (min(u1,v1), max(u1,v1))
                old_e2 = (min(u2,v2), max(u2,v2))
                
                if old_e1 in edge_set and old_e2 in edge_set:
                    edge_set.discard(old_e1)
                    edge_set.discard(old_e2)
                    if ne1 not in edge_set and ne2 not in edge_set:
                        edge_set.add(ne1)
                        edge_set.add(ne2)
                        edges[e1_idx] = new_e1
                        edges[e2_idx] = new_e2
                        adj = _build_adj(n, edges)
                        new_sc = _count_short_cycles(n, adj, g)
                        if new_sc <= sc:
                            sc = new_sc
                            if sc == 0:
                                return edges
                            continue
                        else:
                            # Revert
                            edge_set.discard(ne1)
                            edge_set.discard(ne2)
                            edge_set.add(old_e1)
                            edge_set.add(old_e2)
                            edges[e1_idx] = (u1, v1)
                            edges[e2_idx] = (u2, v2)
        
        return edges

    # Search loop
    while time.time() < deadline:
        edges = _gen_random_4reg(target_n)
        if edges is None:
            continue
        
        if _has_girth_at_least(target_n, edges, g):
            return target_n, edges
        
        # Try local search to fix short cycles
        if time.time() < deadline - 10:
            edges = _local_search(edges, target_n, g)
            if _has_girth_at_least(target_n, edges, g):
                return target_n, edges
            
    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original LUW code
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
