import time
import math
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    For k=4, g=10, it uses Simulated Annealing to search for a 320-vertex graph.
    Falls back to Lazebnik-Ustimenko-Woldar construction if search fails.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin

    if k != 4 or g != 10:
        return _luw_construct(k, g)

    # Target size 320 (known smallest cage)
    target_n = 320

    def _gen_random_4reg(n):
        """Generates a random 4-regular graph on n vertices."""
        for attempt in range(50):
            stubs = []
            for i in range(n):
                stubs.extend([i] * 4)
            random.shuffle(stubs)
            
            pairs = []
            for i in range(0, len(stubs), 2):
                pairs.append([stubs[i], stubs[i+1]])
            
            # Fix loops
            valid = True
            for i in range(len(pairs)):
                if pairs[i][0] == pairs[i][1]:
                    u = pairs[i][0]
                    fixed = False
                    for j in range(len(pairs)):
                        if j == i: continue
                        a, b = pairs[j]
                        if a != u and b != u:
                            pairs[i] = [u, a]
                            pairs[j] = [u, b]
                            fixed = True
                            break
                    if not fixed:
                        valid = False
                        break
            if not valid:
                continue
            
            # Check multi-edges
            edge_set = set()
            valid = True
            for u, v in pairs:
                if u > v: u, v = v, u
                if (u, v) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
            
            if valid:
                return [(pairs[i][0], pairs[i][1]) for i in range(len(pairs))]
        return None

    def _build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def _count_short_cycles(n, adj, g):
        """Count cycles of length < g."""
        short_count = 0
        for s in range(n):
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

    def _apply_swap(edges, adj, edge_set, i1, i2, e1_new, e2_new, e1_old, e2_old):
        """Applies a swap of two edges, updating adj and edge_set."""
        # Remove old
        u1, v1 = e1_old
        u2, v2 = e2_old
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        edge_set.discard(e1_old)
        edge_set.discard(e2_old)
        
        # Add new
        w1, x1 = e1_new
        w2, x2 = e2_new
        adj[w1].append(x1)
        adj[x1].append(w1)
        adj[w2].append(x2)
        adj[x2].append(w2)
        edge_set.add(e1_new)
        edge_set.add(e2_new)
        
        edges[i1] = e1_new
        edges[i2] = e2_new

    edges = _gen_random_4reg(target_n)
    if edges is None:
        return _luw_construct(k, g)

    adj = _build_adj(target_n, edges)
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    current_energy = _count_short_cycles(target_n, adj, g)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    best_energy = current_energy
    best_edges = edges[:]
    best_adj = [lst[:] for lst in adj]
    best_edge_set = set(edge_set)
    
    while time.time() < deadline and T > min_T:
        # Pick two random edges
        i1 = random.randint(0, len(edges) - 1)
        i2 = random.randint(0, len(edges) - 1)
        if i1 == i2:
            continue
            
        u1, v1 = edges[i1]
        u2, v2 = edges[i2]
        
        # Try swap: (u1,v1) and (u2,v2) -> (u1,u2) and (v1,v2)
        new_e1 = (u1, u2)
        new_e2 = (v1, v2)
        
        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            continue
            
        ne1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
        ne2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
        
        if ne1 == ne2:
            continue
            
        old_e1 = (min(u1,v1), max(u1,v1))
        old_e2 = (min(u2,v2), max(u2,v2))
        
        if old_e1 in edge_set and old_e2 in edge_set:
            if ne1 not in edge_set and ne2 not in edge_set:
                # Apply
                _apply_swap(edges, adj, edge_set, i1, i2, ne1, ne2, old_e1, old_e2)
                
                new_energy = _count_short_cycles(target_n, adj, g)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_edges = edges[:]
                    best_adj = [lst[:] for lst in adj]
                    best_edge_set = set(edge_set)
                    if new_energy == 0:
                        return target_n, edges
                        
                delta_E = new_energy - current_energy
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    current_energy = new_energy
                else:
                    # Revert
                    _apply_swap(edges, adj, edge_set, i1, i2, old_e1, old_e2, ne1, ne2)
                    
                T *= cooling

    if best_energy == 0:
        return target_n, best_edges

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
