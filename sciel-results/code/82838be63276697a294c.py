import numpy as np
import random
import time
from itertools import product

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
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
    
    # Voltage Lift over Z10
    m = 10
    n_vertices = 26 * m
    
    # Build adjacency for base graph
    adj = {i: [] for i in range(26)}
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Enumerate all 6-cycles and 8-cycles in base graph
    # Base graph is bipartite with girth 6
    
    # Get all 6-cycles: p1-l1-p2-l2-p3-l3-p1
    # Fix p1 to avoid duplicates, but we need all for constraint checking
    six_cycles = []
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
                                cycle = [(p1, l1), (l1, p2), (p2, l2), (l2, p3), (p3, l3), (l3, p1)]
                                six_cycles.append(cycle)
    
    # Get all 8-cycles: p1-l1-p2-l2-p3-l3-p4-l4-p1
    eight_cycles = []
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
                                        cycle = [(p1, l1), (l1, p2), (p2, l2), (l2, p3), (p3, l3), (l3, p4), (p4, l4), (l4, p1)]
                                        eight_cycles.append(cycle)
    
    # Deterministic search for voltages using backtracking
    # Assign voltages to edges one by one, checking constraints as we go
    
    edge_list = list(base_edges)
    edge_idx = {e: i for i, e in enumerate(edge_list)}
    
    # For each cycle, precompute the edge indices and directions
    def get_cycle_constraints(cycle):
        """Return list of (edge_idx, sign) for each edge in cycle"""
        constraints = []
        for (u, v) in cycle:
            e = (u, v)
            if e in edge_idx:
                constraints.append((edge_idx[e], 1))
            else:
                e = (v, u)
                constraints.append((edge_idx[e], -1))
        return constraints
    
    six_constraints = [get_cycle_constraints(c) for c in six_cycles]
    eight_constraints = [get_cycle_constraints(c) for c in eight_cycles]
    
    # Backtracking search
    n_edges = len(edge_list)
    voltages = [None] * n_edges
    
    def backtrack(idx):
        if idx == n_edges:
            # Verify all cycles
            for con in six_constraints:
                s = 0
                for ei, sign in con:
                    s += sign * voltages[ei]
                if s % m == 0:
                    return False
            for con in eight_constraints:
                s = 0
                for ei, sign in con:
                    s += sign * voltages[ei]
                if s % m == 0:
                    return False
            return True
        
        for val in range(m):
            voltages[idx] = val
            # Early check: for any constraint involving only edges up to idx, check if satisfiable
            valid = True
            for con in six_constraints + eight_constraints:
                # Check if all edges in this constraint are assigned
                all_assigned = all(voltages[ei] is not None for ei, _ in con)
                if not all_assigned:
                    continue
                s = 0
                for ei, sign in con:
                    if voltages[ei] is None:
                        s = None
                        break
                    s += sign * voltages[ei]
                if s is not None and s % m == 0:
                    valid = False
                    break
            if valid:
                if backtrack(idx + 1):
                    return True
        voltages[idx] = None
        return False
    
    if backtrack(0):
        # Build lift graph
        lift_edges = []
        for u, v in base_edges:
            vu = voltages[edge_idx[(u, v)]]
            for i in range(m):
                lift_edges.append((u*m + i, v*m + (i + vu) % m))
        
        # Verify girth via BFS
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
    
    # Fallback: random search
    for _ in range(500):
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m-1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
        
        valid = True
        for con in six_constraints:
            s = 0
            for ei, sign in con:
                e = edge_list[ei]
                if sign == 1:
                    s += voltages[e]
                else:
                    s -= voltages[e]
            if s % m == 0:
                valid = False
                break
        if not valid:
            continue
        
        for con in eight_constraints:
            s = 0
            for ei, sign in con:
                e = edge_list[ei]
                if sign == 1:
                    s += voltages[e]
                else:
                    s -= voltages[e]
            if s % m == 0:
                valid = False
                break
        if not valid:
            continue
        
        # Build lift graph
        lift_edges = []
        for u, v in base_edges:
            vu = voltages[(u, v)]
            for i in range(m):
                lift_edges.append((u*m + i, v*m + (i + vu) % m))
        
        # Verify girth via BFS
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
