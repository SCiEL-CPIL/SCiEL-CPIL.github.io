import numpy as np
import random
import time
import sys
import itertools

def construct(k, g):
    deadline = time.time() + 110  # safety margin before 120s kill
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
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
                
    m = 10
    n_vertices = 26 * m
    
    # Build adjacency and cycle info
    adj = {i: [] for i in range(26)}
    edge_index = {}
    for idx, (u, v) in enumerate(base_edges):
        adj[u].append(v)
        adj[v].append(u)
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
        
    # Find all 6-cycles
    cycles_6 = []
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
                                # Cycle: p1-l1-p2-l2-p3-l3-p1
                                # Edges: (p1,l1), (l1,p2), (p2,l2), (l2,p3), (p3,l3), (l3,p1)
                                e1 = edge_index[(p1, l1)]
                                e2 = edge_index[(l1, p2)]
                                e3 = edge_index[(p2, l2)]
                                e4 = edge_index[(l2, p3)]
                                e5 = edge_index[(p3, l3)]
                                e6 = edge_index[(l3, p1)]
                                cycle_edges = (e1, e2, e3, e4, e5, e6)
                                # Avoid duplicates? 
                                # For backtracking, we just need the list of constraints.
                                # We can store unique cycles to reduce checks.
                                if cycle_edges not in cycles_6:
                                    cycles_6.append(cycle_edges)
                                    
    # Backtracking search for voltages
    # Voltages are assigned to edges 0..51
    # Constraint: sum of voltages on cycle edges != 0 mod 10
    # Note: The sum direction matters? 
    # In the lift, a cycle in the base graph corresponds to a closed walk in the lift if the sum of voltages along the walk is 0 mod m.
    # The voltage of an edge (u,v) is $v_{uv}$. The voltage of (v,u) is $-v_{uv}$.
    # In the cycle enumeration above, we follow the path $p_1 \to l_1 \to p_2 \dots$
    # So we sum $v_{p_1 l_1} + v_{l_1 p_2} + \dots$
    # This is consistent.
    
    def solve_voltages():
        voltages = [-1] * len(base_edges)
        
        # Precompute for each edge, which cycles it belongs to
        edge_to_cycles = {i: [] for i in range(len(base_edges))}
        for i, cyc in enumerate(cycles_6):
            for e in cyc:
                edge_to_cycles[e].append(i)
        
        # Precompute cycle edge lists for faster summing
        cycle_edges_list = [list(cyc) for cyc in cycles_6]
        
        def backtrack(edge_idx):
            if edge_idx == len(base_edges):
                return True
            
            # Randomize value order for diversity
            vals = list(range(m))
            random.shuffle(vals)
            
            for val in vals:
                if time.time() > deadline:
                    return False
                voltages[edge_idx] = val
                valid = True
                # Check all cycles containing this edge
                for ci in edge_to_cycles[edge_idx]:
                    cyc = cycle_edges_list[ci]
                    s = 0
                    complete = True
                    for e in cyc:
                        if voltages[e] == -1:
                            complete = False
                            break
                        s += voltages[e]
                    if complete and s % m == 0:
                        valid = False
                        break
                if valid:
                    if backtrack(edge_idx + 1):
                        return True
            voltages[edge_idx] = -1
            return False
            
        if backtrack(0):
            return voltages
        return None

    # Try multiple random restarts
    voltages = None
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        voltages = solve_voltages()
        if voltages:
            break
    
    if voltages:
        # Build lift graph
        lift_edges = []
        for u, v in base_edges:
            # Get voltage for this directed edge
            # base_edges stores (u,v). voltages is indexed by edge index.
            # We need to be careful with direction.
            # In solve_voltages, we treated edge_index[(u,v)] and (v,u) as same index.
            # The voltage value `v` is assigned to the edge.
            # If we traverse u->v, we add `v`. If v->u, we add `-v`.
            # In the cycle check, we summed voltages in the direction of the walk.
            # So for edge index `idx` corresponding to undirected edge {u,v}, 
            # if the walk goes u->v, we add voltages[idx].
            # if the walk goes v->u, we add -voltages[idx].
            
            # In lift_edges construction:
            # We add edges (u*m + i, v*m + (i + vu) % m).
            # This implies traversing u->v adds `vu`.
            # So we need to map the base edge to its voltage.
            e_idx = edge_index[(u, v)]
            vu = voltages[e_idx]
            for i in range(m):
                lift_edges.append((u*m + i, v*m + (i + vu) % m))
                
        # Verify girth
        adj_lift = {i: [] for i in range(n_vertices)}
        for u, v in lift_edges:
            adj_lift[u].append(v)
            adj_lift[v].append(u)
            
        min_girth = 999
        # BFS from each vertex is O(N * (N+E)). N=260, E=520.
        # 260 * 780 ~ 200k ops. Very fast.
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
            
    # If all fails, use LUW fallback from original code
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
