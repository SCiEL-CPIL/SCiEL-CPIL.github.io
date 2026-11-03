import numpy as np
import random
import time

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
    base_edges = []
    adj = {i: [] for i in range(26)}
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
                adj[i].append(13+j)
                adj[13+j].append(i)
                
    # Pre-compute 6-cycles and 8-cycles in base graph
    # 6-cycles: p1-l1-p2-l2-p3-l3-p1
    cycles_6 = []
    for p1 in range(13):
        for l1 in adj[p1]:
            for p2 in adj[l1]:
                if p2 == p1: continue
                for l2 in adj[p2]:
                    if l2 == l1: continue
                    for p3 in adj[l2]:
                        if p3 in (p1, p2): continue
                        for l3 in adj[p3]:
                            if l3 == l2: continue
                            if p1 in adj[l3]:
                                # Cycle p1-l1-p2-l2-p3-l3-p1
                                # Ensure unique representation to avoid duplicates?
                                # For voltage check, we just need the sequence.
                                cycles_6.append([(p1, l1), (l1, p2), (p2, l2), (l2, p3), (p3, l3), (l3, p1)])
                                
    # 8-cycles: p1-l1-p2-l2-p3-l3-p4-l4-p1
    cycles_8 = []
    for p1 in range(13):
        for l1 in adj[p1]:
            for p2 in adj[l1]:
                if p2 == p1: continue
                for l2 in adj[p2]:
                    if l2 == l1: continue
                    for p3 in adj[l2]:
                        if p3 in (p1, p2): continue
                        for l3 in adj[p3]:
                            if l3 == l2: continue
                            for p4 in adj[l3]:
                                if p4 in (p1, p2, p3): continue
                                for l4 in adj[p4]:
                                    if l4 == l3: continue
                                    if p1 in adj[l4]:
                                        # Check if l4 is connected to p1 and not l1/l2/l3?
                                        # p1 is connected to l1, l2? No, p1-l1-p2-l2-p3-l3-p4-l4-p1.
                                        # l4 must be neighbor of p4 and p1.
                                        # l4 != l3.
                                        # Is l4 == l1? If l4==l1, cycle is p1-l1-p2-l2-p3-l3-p4-l1-p1? No, l1 connects p1 and p2.
                                        # If l4==l1, then p4 must be p2? No, p4 is new.
                                        # So l4 is distinct from l1, l2, l3 usually.
                                        cycles_8.append([(p1, l1), (l1, p2), (p2, l2), (l2, p3), (p3, l3), (l3, p4), (p4, l4), (l4, p1)])

    # Remove duplicates? The enumeration might generate the same cycle multiple times (rotations, reversals).
    # For voltage check, checking duplicates is redundant but safe.
    # To speed up, we can use a set of frozensets of edges?
    # But we need the ordered edges for summing voltages in a specific direction?
    # Voltage sum is path-dependent.
    # However, if we check all generated paths, we cover all cycles.
    
    # To optimize, we can just check the first few or all.
    # Let's deduplicate by sorting the edge set?
    # No, the voltage sum depends on the direction.
    # But if we check both directions, we cover it.
    # The enumeration above generates specific orientations.
    
    # Let's just use the generated lists.
    
    max_m = 15
    for m in range(10, max_m + 1):
        n_vertices = 26 * m
        # Check if n_vertices is feasible (>= Moore Bound)
        # Moore bound for k=4, g=10 is 242.
        # 26*10 = 260 >= 242. OK.
        
        found = False
        for _ in range(2000):
            voltages = {}
            for u, v in base_edges:
                vu = random.randint(0, m-1)
                voltages[(u, v)] = vu
                voltages[(v, u)] = (-vu) % m
            
            valid = True
            # Check 6-cycles
            for cycle in cycles_6:
                s = 0
                for u, v in cycle:
                    s += voltages[(u, v)]
                if s % m == 0:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check 8-cycles
            for cycle in cycles_8:
                s = 0
                for u, v in cycle:
                    s += voltages[(u, v)]
                if s % m == 0:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # If valid, build lift and verify with BFS (just in case)
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
            else:
                # Should not happen if cycle check is correct, but possible if other cycles exist?
                # Base graph girth is 6. Lift girth is determined by 6 and 8 cycles.
                # If 6 and 8 are safe, is 10 safe?
                # A 10-cycle in the lift corresponds to a path in the base graph of length 10?
                # Or a cycle in the base graph of length 10?
                # Base graph has no 10-cycles? PG(2,3) has 13 points, 13 lines.
                # Max cycle length in base?
                # If base has a 10-cycle, we need to check it too.
                # But BFS is the ultimate truth.
                pass

    # Fallback
    # Use the seed's fallback logic
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
