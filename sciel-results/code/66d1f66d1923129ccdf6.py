import numpy as np
import random
import time

def construct(k, g):
    # PG(3,3) incidence graph: 19 points + 19 planes = 38 vertices, 4-regular, girth 8
    # Voltage lift over Z5 -> 190 vertices.
    # Need to verify girth >= 10 (no 8-cycles in lift).
    
    if k != 4 or g != 10:
        # Fallback for other parameters
        return _fallback_luw(k, g)
    
    # Generate points of PG(3,3): nonzero vectors in F3^4 up to scalar
    points = []
    seen = set()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    if a == 0 and b == 0 and c == 0 and d == 0:
                        continue
                    # Canonicalize: first nonzero coordinate = 1
                    if a != 0:
                        inv = 1 if a == 1 else 2
                        p = (1, (b*inv)%3, (c*inv)%3, (d*inv)%3)
                    elif b != 0:
                        inv = 1 if b == 1 else 2
                        p = (0, 1, (c*inv)%3, (d*inv)%3)
                    elif c != 0:
                        inv = 1 if c == 1 else 2
                        p = (0, 0, 1, (d*inv)%3)
                    else:
                        p = (0, 0, 0, 1)
                    if p not in seen:
                        seen.add(p)
                        points.append(p)
    
    # Generate planes: nonzero linear forms up to scalar
    planes = []
    seen_p = set()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    if a == 0 and b == 0 and c == 0 and d == 0:
                        continue
                    if a != 0:
                        inv = 1 if a == 1 else 2
                        n = (1, (b*inv)%3, (c*inv)%3, (d*inv)%3)
                    elif b != 0:
                        inv = 1 if b == 1 else 2
                        n = (0, 1, (c*inv)%3, (d*inv)%3)
                    elif c != 0:
                        inv = 1 if c == 1 else 2
                        n = (0, 0, 1, (d*inv)%3)
                    else:
                        n = (0, 0, 0, 1)
                    if n not in seen_p:
                        seen_p.add(n)
                        planes.append(n)
    
    n_points = len(points)  # 19
    n_planes = len(planes)  # 19
    
    # Build incidence: point i is on plane j if dot product = 0
    base_adj = {}
    for i in range(n_points):
        base_adj[i] = []
    for j in range(n_planes):
        base_adj[n_points + j] = []
    
    for i, p in enumerate(points):
        for j, n in enumerate(planes):
            if (n[0]*p[0] + n[1]*p[1] + n[2]*p[2] + n[3]*p[3]) % 3 == 0:
                base_adj[i].append(n_points + j)
                base_adj[n_points + j].append(i)
    
    # Each vertex should have degree 4
    # Verify
    for v in range(n_points + n_planes):
        if len(base_adj[v]) != 4:
            # Fallback
            return _fallback_luw(k, g)
    
    # Edge list (undirected, canonical order)
    base_edges = []
    for i in range(n_points):
        for j in base_adj[i]:
            if i < j:
                base_edges.append((i, j))
    # Plane-plane edges (none in incidence graph)
    # Point-point edges (none)
    # So base_edges has only point-plane edges
    
    m = 5  # voltage group size
    n_vertices = (n_points + n_planes) * m
    
    # Search for voltages that break all 8-cycles
    # 8-cycles in base graph: P1-L1-P2-L2-P3-L3-P4-L4-P1
    # where Pi are distinct points, Lj are distinct planes
    # Voltage sum around cycle must be nonzero mod 5
    
    # Enumerate all 8-cycles in base graph
    # For efficiency, precompute
    cycles_8 = []
    for p1 in range(n_points):
        for l1 in base_adj[p1]:
            for p2 in base_adj[l1]:
                if p2 == p1:
                    continue
                for l2 in base_adj[p2]:
                    if l2 == l1:
                        continue
                    for p3 in base_adj[l2]:
                        if p3 == p1 or p3 == p2:
                            continue
                        for l3 in base_adj[p3]:
                            if l3 == l2:
                                continue
                            for p4 in base_adj[l3]:
                                if p4 == p1 or p4 == p2 or p4 == p3:
                                    continue
                                for l4 in base_adj[p4]:
                                    if l4 == l3:
                                        continue
                                    if p1 in base_adj[l4]:
                                        # Cycle: p1-l1-p2-l2-p3-l3-p4-l4-p1
                                        cycle = [p1, l1, p2, l2, p3, l3, p4, l4]
                                        cycles_8.append(cycle)
    
    # Also check 4-cycles (should be none in girth-8 graph, but verify)
    # And 6-cycles (should be none)
    
    # Try deterministic voltage assignments first
    # Then random search
    start_time = time.time()
    max_time = 100  # seconds
    
    def check_voltages(voltages):
        """Check if voltage assignment breaks all 8-cycles."""
        for cycle in cycles_8:
            s = 0
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                s += voltages.get((u,v), voltages.get((v,u), 0))
                s %= m
            if s == 0:
                return False
        return True
    
    # Deterministic attempts
    for attempt in range(200):
        if time.time() - start_time > max_time:
            break
        voltages = {}
        for u, v in base_edges:
            if attempt < 10:
                # Structured assignments
                voltages[(u,v)] = (u + v + attempt) % m
            else:
                voltages[(u,v)] = random.randint(0, m-1)
            voltages[(v,u)] = (-voltages[(u,v)]) % m
        
        if check_voltages(voltages):
            # Build lift graph
            lift_edges = []
            for u, v in base_edges:
                vu = voltages[(u,v)]
                for i in range(m):
                    lift_edges.append((u*m + i, v*m + (i + vu) % m))
            
            # Verify girth via BFS
            adj_lift = [[] for _ in range(n_vertices)]
            for u, v in lift_edges:
                adj_lift[u].append(v)
                adj_lift[v].append(u)
            
            min_girth = 999
            for start_v in range(n_vertices):
                dist = [-1] * n_vertices
                parent = [-1] * n_vertices
                dist[start_v] = 0
                queue = [start_v]
                qi = 0
                while qi < len(queue):
                    cur = queue[qi]
                    qi += 1
                    for nb in adj_lift[cur]:
                        if dist[nb] == -1:
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
    
    # Fallback
    return _fallback_luw(k, g)

def _fallback_luw(k, g):
    """LUW construction fallback."""
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
