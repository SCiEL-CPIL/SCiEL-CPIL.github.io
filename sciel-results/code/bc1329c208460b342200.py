import numpy as np
import random
import time
from itertools import combinations

def construct(k, g):
    # Construct PG(2,3) incidence graph
    # Points: 13, Lines: 13
    # Total 26 vertices.
    
    # Generate points
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0: continue
                # Canonicalize: first non-zero is 1
                if x != 0:
                    inv_x = 1 if x == 1 else 2
                    p = (1, (y*inv_x)%3, (z*inv_x)%3)
                elif y != 0:
                    inv_y = 1 if y == 1 else 2
                    p = (0, 1, (z*inv_y)%3)
                else:
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)
    
    # Generate lines
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0: continue
                # Canonicalize normal
                if a != 0:
                    inv_a = 1 if a == 1 else 2
                    n = (1, (b*inv_a)%3, (c*inv_a)%3)
                elif b != 0:
                    inv_b = 1 if b == 1 else 2
                    n = (0, 1, (c*inv_b)%3)
                else:
                    n = (0, 0, 1)
                
                if n in [tuple(l) for l in lines]: continue
                lines.append(n)
                pts = []
                for p in points:
                    if (n[0]*p[0] + n[1]*p[1] + n[2]*p[2]) % 3 == 0:
                        pts.append(p)
                line_points.append(pts)
    
    # Build base graph
    # Vertices 0..12 are points, 13..25 are lines.
    base_edges = []
    adj = {i: [] for i in range(26)}
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
                adj[i].append(13+j)
                adj[13+j].append(i)
                
    # Enumerate cycles of length 6 and 8 in the base graph
    # We need to orient them.
    # Store as list of (u, v) directed edges.
    
    def get_directed_cycles(length):
        cycles = []
        seen = set()
        for start in range(26):
            stack = [(start, [start])]
            while stack:
                cur, path = stack.pop()
                if len(path) == length:
                    if start in adj[cur]:
                        # Check if this cycle is new
                        # Canonical form: rotate so smallest vertex is first, and second smallest is second?
                        # Or just store the set of directed edges?
                        # A cycle is defined by the sequence of vertices.
                        # Two sequences can represent the same cycle (rotation/reversal).
                        # We can store the frozenset of undirected edges to deduplicate.
                        # But we need the directed sequence for the voltage sum.
                        # If we find a cycle, we can store one valid orientation.
                        edges_in_cycle = set()
                        for i in range(length):
                            u = path[i]
                            v = path[(i+1)%length]
                            edges_in_cycle.add((min(u,v), max(u,v)))
                        key = frozenset(edges_in_cycle)
                        if key not in seen:
                            seen.add(key)
                            # Store the path as is (it's a valid orientation)
                            cycles.append(path)
                    continue
                for nb in adj[cur]:
                    if nb not in path:
                        # Pruning: ensure we don't go back to start until the end
                        if len(path) < length - 1:
                            if nb == start:
                                continue
                        stack.append((nb, path + [nb]))
        return cycles

    cycles_6 = get_directed_cycles(6)
    cycles_8 = get_directed_cycles(8)
    
    # Search for voltage assignment
    # Try m=10, 11, 13
    best_n = None
    best_edges = None
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    for m in [10, 11, 13]:
        if time.time() > deadline:
            break
            
        n_vertices = 26 * m
        
        # Random restarts with local search
        for restart in range(10000):
            if time.time() > deadline:
                break
                
            voltages = {}
            for u, v in base_edges:
                val = random.randint(0, m-1)
                voltages[(u, v)] = val
                voltages[(v, u)] = (-val) % m
            
            # Check validity
            def is_valid(voltages):
                for path in cycles_6:
                    s = 0
                    for i in range(len(path)):
                        u = path[i]
                        v = path[(i+1)%len(path)]
                        s += voltages[(u, v)]
                    if s % m == 0:
                        return False
                for path in cycles_8:
                    s = 0
                    for i in range(len(path)):
                        u = path[i]
                        v = path[(i+1)%len(path)]
                        s += voltages[(u, v)]
                    if s % m == 0:
                        return False
                return True
            
            if is_valid(voltages):
                # Build lift
                lift_edges = []
                for u, v in base_edges:
                    vu = voltages[(u, v)]
                    for i in range(m):
                        lift_edges.append((u*m + i, v*m + (i + vu) % m))
                
                # Verify girth
                adj_lift = {i: [] for i in range(n_vertices)}
                for u, v in lift_edges:
                    adj_lift[u].append(v)
                    adj_lift[v].append(u)
                    
                min_girth = 999
                for start in range(n_vertices):
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
                    return n_vertices, lift_edges

    # Fallback to LUW
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
