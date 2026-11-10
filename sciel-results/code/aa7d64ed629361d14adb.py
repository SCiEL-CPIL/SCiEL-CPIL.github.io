import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # Leave 10s for safety
    
    if k != 5:
        return _construct_luw(k, g)
        
    for q in [13, 17, 19, 23]:
        if time.time() > deadline:
            break
            
        n = q * (q * q - 1) // 2
        if n < 1066:
            continue
            
        elements, idx_map, inverses, orders, mul_table, ident_idx = _generate_psl2_fast(q)
        
        involutions = []
        non_involutions = []
        for i in range(n):
            if orders[i] == 2:
                involutions.append(i)
            elif orders[i] > 2:
                non_involutions.append(i)
                
        if not involutions or not non_involutions:
            continue
            
        found = False
        attempts = 0
        max_attempts = 10000
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            
            s1_idx = random.choice(non_involutions)
            s2_idx = random.choice(non_involutions)
            s3_idx = random.choice(involutions)
            
            s1_inv_idx = inverses[s1_idx]
            s2_inv_idx = inverses[s2_idx]
            
            candidates = [s1_idx, s1_inv_idx, s2_idx, s2_inv_idx, s3_idx]
            
            if len(set(candidates)) != 5:
                continue
                
            if _check_girth_cayley(n, candidates, mul_table, ident_idx, g):
                edges = []
                for i in range(n):
                    for s in candidates:
                        j = mul_table[i][s]
                        if i < j:
                            edges.append((i, j))
                
                if len(edges) == n * 5 // 2:
                    return n, edges
                    
    return _construct_luw(k, g)

def _generate_psl2_fast(q):
    elements = []
    idx_map = {}
    
    inv = [0] * q
    for i in range(1, q):
        inv[i] = pow(i, q - 2, q)
        
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0:
                    if b == 0:
                        continue
                    if (b * c) % q != (q - 1) % q:
                        continue
                    d = 0
                else:
                    d = (1 + b * c) % q
                    d = (d * inv[a]) % q
                
                if (a * d - b * c) % q != 1:
                    continue
                
                m = (a, b, c, d)
                first_non_zero = None
                for val in m:
                    if val != 0:
                        first_non_zero = val
                        break
                        
                if first_non_zero is not None and first_non_zero > (q - 1) // 2:
                    m = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
                    
                if m not in idx_map:
                    idx_map[m] = len(elements)
                    elements.append(m)
                    
    n = len(elements)
    
    mul_table = [[0] * n for _ in range(n)]
    inverses = [0] * n
    orders = [0] * n
    
    ident = (1, 0, 0, 1)
    ident_idx = idx_map[ident]
    
    for i in range(n):
        a1, b1, c1, d1 = elements[i]
        for j in range(n):
            a2, b2, c2, d2 = elements[j]
            a = (a1 * a2 + b1 * c2) % q
            b = (a1 * b2 + b1 * d2) % q
            c = (c1 * a2 + d1 * c2) % q
            d = (c1 * b2 + d1 * d2) % q
            prod = (a, b, c, d)
            first_non_zero = None
            for val in prod:
                if val != 0:
                    first_non_zero = val
                    break
            if first_non_zero is not None and first_non_zero > (q - 1) // 2:
                prod = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
            
            mul_table[i][j] = idx_map[prod]
            
    for i in range(n):
        a, b, c, d = elements[i]
        inv_m = (d, (q - b) % q, (q - c) % q, a)
        first_non_zero = None
        for val in inv_m:
            if val != 0:
                first_non_zero = val
                break
        if first_non_zero is not None and first_non_zero > (q - 1) // 2:
            inv_m = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
        inverses[i] = idx_map[inv_m]
        
        cur = i
        order = 1
        while cur != ident_idx:
            cur = mul_table[cur][i]
            order += 1
            if order > q + 2:
                break
        orders[i] = order
        
    return elements, idx_map, inverses, orders, mul_table, ident_idx

def _check_girth_cayley(n, candidates, mul_table, ident_idx, g):
    dist = [-1] * n
    dist[ident_idx] = 0
    queue = deque([ident_idx])
    
    while queue:
        u = queue.popleft()
        du = dist[u]
        
        # If we are at depth du, neighbors are at du+1.
        # A cycle of length L is found if a neighbor v has dist[v] <= du-1?
        # Specifically, if v is ident_idx and du > 0, cycle length is du+1.
        # If v is visited and dist[v] < du, we have a cycle.
        # But we only care about cycles < g.
        
        for s in candidates:
            v = mul_table[u][s]
            
            if v == ident_idx:
                if du > 0:
                    cycle_len = du + 1
                    if cycle_len < g:
                        return False
            else:
                if dist[v] == -1:
                    dist[v] = du + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # The cycle length is du + 1 + dist[v] ? No.
                    # In BFS, if we find an edge (u, v) where v is visited,
                    # and v is not parent of u, we have a cycle.
                    # The length of the cycle is dist[u] + dist[v] + 1.
                    # This is valid if the paths are disjoint?
                    # In BFS tree, paths from root to u and root to v are unique.
                    # The union forms a cycle of length dist[u] + dist[v] + 1.
                    # This cycle passes through the root? No, it passes through LCA.
                    # Actually, this formula gives the length of the cycle formed by the two tree paths and the edge.
                    # This cycle might not be simple if they share nodes other than LCA?
                    # In a tree, paths to u and v share nodes up to LCA.
                    # The cycle is simple.
                    # So cycle_len = dist[u] + dist[v] + 1.
                    # However, we must ensure this is not a backtracking edge (u -> parent).
                    # If v is parent of u, dist[v] = dist[u] - 1.
                    # Then cycle_len = dist[u] + dist[u] - 1 + 1 = 2*dist[u].
                    # This corresponds to going back and forth?
                    # In a simple graph, (u, v) is an edge.
                    # If v is parent, the "cycle" is u->v->u, length 2.
                    # But graph is simple, so no multiple edges.
                    # However, in Cayley graph, we might have s and s^-1.
                    # If we go u -> s -> v, and v is parent, it means we went back.
                    # This is a "cycle" of length 2 in the walk, but not a simple cycle in the graph (unless multigraph).
                    # Since graph is simple, we should ignore backtracking.
                    # So if dist[v] == dist[u] - 1, skip?
                    # Or just check if cycle_len < g.
                    # If cycle_len = 2, and g=9, 2 < 9, so it would fail.
                    # But girth is defined on simple cycles.
                    # So we must ensure we don't count backtracking.
                    # Condition: dist[v] != dist[u] - 1.
                    
                    if dist[v] != dist[u] - 1:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
                            
    return True

def _construct_luw(k, g):
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
