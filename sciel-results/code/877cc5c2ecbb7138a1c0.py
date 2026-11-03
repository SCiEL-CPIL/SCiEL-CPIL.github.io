import numpy as np
import random
import time

def construct(k, g):
    # Construct PG(2,3) incidence graph
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
    adj = {i: [] for i in range(26)}
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
                adj[i].append(13+j)
                adj[13+j].append(i)
    
    # Pre-compute all 6-cycles and 8-cycles with canonical form
    cycles6 = set()
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
                                cyc = (p1, l1, p2, l2, p3, l3)
                                # Canonicalize: rotate so min vertex first
                                mn = min(cyc)
                                for r in range(6):
                                    if cyc[r] == mn:
                                        rot = tuple(cyc[r:] + cyc[:r])
                                        # Also try reverse
                                        rev = tuple(reversed(cyc))
                                        for r2 in range(6):
                                            if rev[r2] == mn:
                                                rot2 = tuple(rev[r2:] + rev[:r2])
                                                cycles6.add(min(rot, rot2))
                                        break
                                break
    cycles6 = list(cycles6)
    
    cycles8 = set()
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
                            for p4 in adj[l3]:
                                if p4 == p1 or p4 == p2 or p4 == p3: continue
                                for l4 in adj[p4]:
                                    if l4 == l3: continue
                                    if p1 in adj[l4]:
                                        cyc = (p1, l1, p2, l2, p3, l3, p4, l4)
                                        mn = min(cyc)
                                        for r in range(8):
                                            if cyc[r] == mn:
                                                rot = tuple(cyc[r:] + cyc[:r])
                                                rev = tuple(reversed(cyc))
                                                for r2 in range(8):
                                                    if rev[r2] == mn:
                                                        rot2 = tuple(rev[r2:] + rev[:r2])
                                                        cycles8.add(min(rot, rot2))
                                                break
                                        break
    cycles8 = list(cycles8)
    
    m = 10
    n_vertices = 26 * m
    edge_to_idx = {}
    for idx, (u, v) in enumerate(base_edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx
    
    # Fast random search with pre-computed cycles
    for attempt in range(20000):
        random.seed(attempt)
        voltages = [random.randint(0, m-1) for _ in range(len(base_edges))]
        
        valid = True
        for cyc in cycles6:
            s = 0
            for i in range(6):
                u, v = cyc[i], cyc[(i+1) % 6]
                s += voltages[edge_to_idx[(u, v)]]
            if s % m == 0:
                valid = False
                break
        if not valid:
            continue
        for cyc in cycles8:
            s = 0
            for i in range(8):
                u, v = cyc[i], cyc[(i+1) % 8]
                s += voltages[edge_to_idx[(u, v)]]
            if s % m == 0:
                valid = False
                break
        if not valid:
            continue
        
        # Build lift and verify
        lift_edges = []
        for idx, (u, v) in enumerate(base_edges):
            vu = voltages[idx]
            for i in range(m):
                lift_edges.append((u*m + i, v*m + (i + vu) % m))
        
        adj_lift = [[] for _ in range(n_vertices)]
        for u, v in lift_edges:
            adj_lift[u].append(v)
            adj_lift[v].append(u)
        
        min_girth = 999
        for start in range(n_vertices):
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                cur = queue[head]
                head += 1
                for nb in adj_lift[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        parent[nb] = cur
                        queue.append(nb)
                    elif parent[nb] != cur and parent[cur] != nb:
                        cl = dist[nb] + dist[cur] + 1
                        if cl < min_girth:
                            min_girth = cl
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
