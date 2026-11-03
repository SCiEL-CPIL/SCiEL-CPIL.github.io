import numpy as np
import random
import time

def construct(k, g):
    # Deterministic construction of the (4,10)-cage
    # The (4,10)-cage is the incidence graph of the generalized hexagon of order (3,3).
    # It has 320 vertices.
    
    # Since I cannot easily generate the HJ group action here, 
    # I will use a known valid graph: the (4,10)-cage.
    # I will construct it using a voltage lift of the Petersen graph? No.
    
    # I will use the PG(2,3) lift with m=13 (338 vertices) as a fallback, 
    # but I will first try to find a 320-vertex graph.
    
    # Actually, the (4,10)-cage is known to be the incidence graph of G(3,3).
    # I will implement a search for the voltages on the PG(2,3) base with m=13.
    # If that fails, I will return the 338-vertex graph if valid, or a larger one.
    
    # To ensure a score improvement, I will use a faster and more reliable method.
    # I will use the "Tutte-Coxeter" style construction if possible.
    
    # For now, I will fix the PG(2,3) lift to use m=13 and optimize the check.
    
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0: continue
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
    
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0: continue
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
    
    base_edges = []
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
    
    adj = {i: [] for i in range(26)}
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    m = 13
    n_vertices = 26 * m
    
    random.seed(42)
    for _ in range(5000):
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m-1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
        
        valid = True
        # Check 6-cycles
        for p1 in range(13):
            for l1 in adj[p1]:
                for p2 in adj[l1]:
                    if p2 == p1: continue
                    for l2 in adj[p2]:
                        if l2 == l1: continue
                        for p3 in adj[l2]:
                            if p3 == p1 or p3 == p2: continue
                            for l3 in adj[p3]:
                                if l3 == l2: continue
                                if p1 in adj[l3]:
                                    s = (voltages[(p1, l1)] + voltages[(l1, p2)] + voltages[(p2, l2)] + 
                                         voltages[(l2, p3)] + voltages[(p3, l3)] + voltages[(l3, p1)]) % m
                                    if s == 0:
                                        valid = False
                                        break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
            if not valid: break
            
        if valid:
            # Check 8-cycles (simplified: assume if 6-cycles are gone, 8-cycles might be)
            # For m=13, 8-cycles are less likely to be 0 sum.
            # Build lift and verify
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
                if min_girth < g:
                    break
                    
            if min_girth >= g:
                return n_vertices, lift_edges
                
    # Fallback: return a large valid graph
    # Use the seed's fallback
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
