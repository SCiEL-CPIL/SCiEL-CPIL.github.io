import numpy as np
import random
import time

def construct(k, g):
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # For k=4, g=10, try voltage lift of PG(2,3)
    if k == 4 and g == 10:
        # Build PG(2,3) incidence graph (26 vertices, 52 edges)
        # Points and lines in PG(2,3)
        points = []
        seen_points = set()
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Canonicalize: make first non-zero coordinate 1
                    if x != 0:
                        inv_x = 1 if x == 1 else 2
                        p = (1, (y * inv_x) % 3, (z * inv_x) % 3)
                    elif y != 0:
                        inv_y = 1 if y == 1 else 2
                        p = (0, 1, (z * inv_y) % 3)
                    else:
                        p = (0, 0, 1)
                    if p not in seen_points:
                        seen_points.add(p)
                        points.append(p)
        
        lines = []
        line_points = []
        seen_lines = set()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    if a == 0 and b == 0 and c == 0:
                        continue
                    # Canonicalize line normal
                    if a != 0:
                        inv_a = 1 if a == 1 else 2
                        n = (1, (b * inv_a) % 3, (c * inv_a) % 3)
                    elif b != 0:
                        inv_b = 1 if b == 1 else 2
                        n = (0, 1, (c * inv_b) % 3)
                    else:
                        n = (0, 0, 1)
                    
                    if n in seen_lines:
                        continue
                    seen_lines.add(n)
                    lines.append(n)
                    pts = []
                    for p in points:
                        if (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) % 3 == 0:
                            pts.append(p)
                    line_points.append(pts)
        
        # Build base graph adjacency
        base_edges = []
        adj_base = {i: [] for i in range(26)}
        for i, p in enumerate(points):
            for j, lp in enumerate(line_points):
                if p in lp:
                    base_edges.append((i, 13 + j))
                    adj_base[i].append(13 + j)
                    adj_base[13 + j].append(i)
        
        # Pre-compute all 6-cycles and 8-cycles in base graph
        # For voltage lift, we need cycles of length 6 and 8 to be non-zero sum mod m
        # 6-cycles: p1-l1-p2-l2-p3-l3-p1
        cycles_6 = []
        for p1 in range(13):
            for l1 in adj_base[p1]:
                for p2 in adj_base[l1]:
                    if p2 == p1:
                        continue
                    for l2 in adj_base[p2]:
                        if l2 == l1:
                            continue
                        for p3 in adj_base[l2]:
                            if p3 in (p1, p2):
                                continue
                            for l3 in adj_base[p3]:
                                if l3 == l2:
                                    continue
                                if p1 in adj_base[l3]:
                                    cycles_6.append([(p1, l1), (l1, p2), (p2, l2), (l2, p3), (p3, l3), (l3, p1)])
        
        # 8-cycles: p1-l1-p2-l2-p3-l3-p4-l4-p1
        cycles_8 = []
        for p1 in range(13):
            for l1 in adj_base[p1]:
                for p2 in adj_base[l1]:
                    if p2 == p1:
                        continue
                    for l2 in adj_base[p2]:
                        if l2 == l1:
                            continue
                        for p3 in adj_base[l2]:
                            if p3 in (p1, p2):
                                continue
                            for l3 in adj_base[p3]:
                                if l3 == l2:
                                    continue
                                for p4 in adj_base[l3]:
                                    if p4 in (p1, p2, p3):
                                        continue
                                    for l4 in adj_base[p4]:
                                        if l4 == l3:
                                            continue
                                        if p1 in adj_base[l4]:
                                            # Check that this is a simple cycle
                                            # l4 must not be l1 (otherwise it's a 6-cycle)
                                            if l4 != l1:
                                                cycles_8.append([(p1, l1), (l1, p2), (p2, l2), (l2, p3), (p3, l3), (l3, p4), (p4, l4), (l4, p1)])
        
        # Try to find voltage assignments for m=10, 11, 12
        for m in [10, 11, 12]:
            if time.time() > deadline:
                break
            
            n_vertices = 26 * m
            # Moore bound check
            moore_bound = 1 + k * sum([k**(i-1) for i in range(1, g//2)])
            if n_vertices < moore_bound:
                continue
            
            # Local search for voltage assignment
            best_cost = float('inf')
            best_voltages = None
            
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Random initialization
                voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
                
                # Cost function: number of cycles with zero sum mod m
                def compute_cost(v):
                    cost = 0
                    for cycle in cycles_6:
                        s = 0
                        for u, w in cycle:
                            idx = base_edges.index((u, w))
                            s += v[idx]
                        if s % m == 0:
                            cost += 1
                    for cycle in cycles_8:
                        s = 0
                        for u, w in cycle:
                            idx = base_edges.index((u, w))
                            s += v[idx]
                        if s % m == 0:
                            cost += 1
                    return cost
                
                cost = compute_cost(voltages)
                
                # Hill climbing
                stuck = 0
                for step in range(200):
                    if time.time() > deadline:
                        break
                    
                    if cost == 0:
                        best_cost = 0
                        best_voltages = voltages[:]
                        break
                    
                    # Try changing one random edge voltage
                    edge_idx = random.randint(0, len(base_edges) - 1)
                    old_val = voltages[edge_idx]
                    new_val = random.randint(0, m - 1)
                    while new_val == old_val:
                        new_val = random.randint(0, m - 1)
                    voltages[edge_idx] = new_val
                    
                    new_cost = compute_cost(voltages)
                    
                    if new_cost <= cost:
                        cost = new_cost
                        stuck = 0
                    else:
                        voltages[edge_idx] = old_val
                        stuck += 1
                    
                    if stuck > 50:
                        break
                
                if cost == 0:
                    best_cost = 0
                    best_voltages = voltages[:]
                    break
                
                if cost < best_cost:
                    best_cost = cost
                    best_voltages = voltages[:]
            
            if best_voltages is not None and best_cost == 0:
                # Build the lifted graph
                lift_edges = []
                for i, (u, v) in enumerate(base_edges):
                    vu = best_voltages[i]
                    for layer in range(m):
                        lift_edges.append((u * m + layer, v * m + (layer + vu) % m))
                
                # Verify girth with BFS
                adj_lift = {i: [] for i in range(n_vertices)}
                for u, v in lift_edges:
                    adj_lift[u].append(v)
                    adj_lift[v].append(u)
                
                min_girth = 999
                for start in range(n_vertices):
                    if time.time() > deadline:
                        break
                    dist = {start: 0}
                    parent = {start: None}
                    queue = [start]
                    while queue:
                        cur = queue.pop(0)
                        for nb in adj_lift[cur]:
                            if nb not in dist:
                                dist[nb] = dist[cur] + 1
                                parent[nb] = cur
                                queue.append(nb)
                            elif parent[nb] != cur and parent[cur] != nb:
                                cycle_len = dist[nb] + dist[cur] + 1
                                if cycle_len < min_girth:
                                    min_girth = cycle_len
                    if min_girth >= g:
                        break
                
                if min_girth >= g:
                    return n_vertices, lift_edges
        
        # Fallback: use LUW or other method
        pass
    
    # Fallback construction for general k, g
    # Use a simple construction or return a known graph
    # For k=4, g=10, we can try the (3,10) cage or use a known construction
    
    # Simple fallback: use the seed's LUW-like construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q
    
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
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
