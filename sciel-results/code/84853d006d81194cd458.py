import numpy as np
import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    # Build PG(2,3) incidence graph (26 vertices, 4-regular, girth 6)
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = 1 if x == 1 else 2
                    p = (1, (y * inv_x) % 3, (z * inv_x) % 3)
                elif y != 0:
                    inv_y = 1 if y == 1 else 2
                    p = (0, 1, (z * inv_y) % 3)
                else:
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)

    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0:
                    continue
                if a != 0:
                    inv_a = 1 if a == 1 else 2
                    n = (1, (b * inv_a) % 3, (c * inv_a) % 3)
                elif b != 0:
                    inv_b = 1 if b == 1 else 2
                    n = (0, 1, (c * inv_b) % 3)
                else:
                    n = (0, 0, 1)
                if n in [tuple(l) for l in lines]:
                    continue
                lines.append(n)
                pts = []
                for p in points:
                    if (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) % 3 == 0:
                        pts.append(p)
                line_points.append(pts)

    # Base edges and adjacency
    base_edges = []
    adj = {i: [] for i in range(26)}
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13 + j))
                adj[i].append(13 + j)
                adj[13 + j].append(i)

    edge_to_idx = {}
    for idx, (u, v) in enumerate(base_edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx

    n_edges = len(base_edges)  # 52

    # Pre-compute all 6-cycles and 8-cycles as lists of edge indices
    cycles6 = []
    seen6 = set()
    for p1 in range(13):
        for l1 in adj[p1]:
            for p2 in adj[l1]:
                if p2 == p1:
                    continue
                for l2 in adj[p2]:
                    if l2 == l1:
                        continue
                    for p3 in adj[l2]:
                        if p3 == p1 or p3 == p2:
                            continue
                        for l3 in adj[p3]:
                            if l3 == l2:
                                continue
                            if p1 in adj[l3]:
                                # Cycle: p1-l1-p2-l2-p3-l3-p1
                                cyc_edges = [
                                    edge_to_idx[(p1, l1)],
                                    edge_to_idx[(l1, p2)],
                                    edge_to_idx[(p2, l2)],
                                    edge_to_idx[(l2, p3)],
                                    edge_to_idx[(p3, l3)],
                                    edge_to_idx[(l3, p1)],
                                ]
                                # Canonical form: sort the cycle representation
                                cyc = tuple(sorted(cyc_edges))
                                if cyc not in seen6:
                                    seen6.add(cyc)
                                    cycles6.append(cyc_edges)

    cycles8 = []
    seen8 = set()
    for p1 in range(13):
        for l1 in adj[p1]:
            for p2 in adj[l1]:
                if p2 == p1:
                    continue
                for l2 in adj[p2]:
                    if l2 == l1:
                        continue
                    for p3 in adj[l2]:
                        if p3 == p1 or p3 == p2:
                            continue
                        for l3 in adj[p3]:
                            if l3 == l2:
                                continue
                            for p4 in adj[l3]:
                                if p4 == p1 or p4 == p2 or p4 == p3:
                                    continue
                                for l4 in adj[p4]:
                                    if l4 == l3:
                                        continue
                                    if p1 in adj[l4]:
                                        cyc_edges = [
                                            edge_to_idx[(p1, l1)],
                                            edge_to_idx[(l1, p2)],
                                            edge_to_idx[(p2, l2)],
                                            edge_to_idx[(l2, p3)],
                                            edge_to_idx[(p3, l3)],
                                            edge_to_idx[(l3, p4)],
                                            edge_to_idx[(p4, l4)],
                                            edge_to_idx[(l4, p1)],
                                        ]
                                        cyc = tuple(sorted(cyc_edges))
                                        if cyc not in seen8:
                                            seen8.add(cyc)
                                            cycles8.append(cyc_edges)

    # Convert to numpy arrays for fast checking
    cycles6_arr = np.array(cycles6, dtype=np.int32) if cycles6 else np.empty((0, 6), dtype=np.int32)
    cycles8_arr = np.array(cycles8, dtype=np.int32) if cycles8 else np.empty((0, 8), dtype=np.int32)

    # Try voltage lifts with various moduli
    moduli_to_try = [10, 12, 15, 20, 25, 30, 16, 18, 24]

    def check_and_build(voltages, m):
        """Check if voltage assignment works and build lift if so."""
        # Check 6-cycles
        if len(cycles6_arr) > 0:
            sums6 = (cycles6_arr @ voltages) % m
            if np.any(sums6 == 0):
                return None
        # Check 8-cycles
        if len(cycles8_arr) > 0:
            sums8 = (cycles8_arr @ voltages) % m
            if np.any(sums8 == 0):
                return None

        # Build lift
        n_vertices = 26 * m
        lift_edges = []
        for idx, (u, v) in enumerate(base_edges):
            vu = voltages[idx]
            for i in range(m):
                lift_edges.append((u * m + i, v * m + (i + vu) % m))

        # Verify girth via BFS (quick check on a few vertices)
        adj_lift = [[] for _ in range(n_vertices)]
        for u, v in lift_edges:
            adj_lift[u].append(v)
            adj_lift[v].append(u)

        min_girth = 999
        # Check girth from a subset of vertices for speed
        check_vertices = list(range(0, n_vertices, max(1, n_vertices // 20)))
        for start in check_vertices:
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                cur = queue[head]
                head += 1
                if dist[cur] >= g:
                    break
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
        return None

    # Random search with multiple moduli
    for m in moduli_to_try:
        if time.time() > deadline:
            break
        n_attempts = 5000 if m <= 12 else 2000
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            random.seed(attempt * 31 + m)
            voltages = np.array([random.randint(0, m - 1) for _ in range(n_edges)], dtype=np.int32)
            result = check_and_build(voltages, m)
            if result is not None:
                return result

    # Fallback: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
