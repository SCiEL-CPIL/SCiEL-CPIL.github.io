import numpy as np
import random
import time
import sys
from collections import defaultdict

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
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
                
    m = 10
    n_vertices = 26 * m
    
    # Pre-compute 6-cycles
    # adj list
    adj = {i: [] for i in range(26)}
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    six_cycles = []
    # P1-L1-P2-L2-P3-L3-P1
    # P1, P2, P3 are points (0-12)
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
                                # Note: voltages are stored on directed edges or unordered?
                                # Let's use unordered edges in the cycle list for summing.
                                # We need consistent orientation.
                                # Let's just store the list of (u,v) pairs as they appear in the cycle.
                                cycle_edges = [(p1, l1), (l1, p2), (p2, l2), (l2, p3), (p3, l3), (l3, p1)]
                                six_cycles.append(cycle_edges)
    
    # Backtracking search for voltages
    # Variables: base_edges
    # Values: 0..9
    # Constraints: Sum of voltages on cycle != 0
    
    # To make backtracking efficient, we can use a simple recursive search.
    # Given the small size, it should be fast.
    
    def solve_voltages():
        voltages = {}
        # Order edges
        edges_to_assign = list(base_edges)
        # Randomize order to avoid bias
        random.shuffle(edges_to_assign)
        
        def backtrack(idx):
            if idx == len(edges_to_assign):
                # Check all 6-cycles
                for cycle in six_cycles:
                    s = 0
                    for u, v in cycle:
                        # Voltage is stored as (u,v) -> val
                        # We need to ensure we get the voltage for the edge regardless of direction
                        # In the lift, (u,v) with voltage w means u*m+i -> v*m+(i+w)%m
                        # The cycle sum is sum of voltages in the direction of the cycle.
                        # So we need voltages[(u,v)] for each (u,v) in the cycle sequence.
                        # Our voltages dict has entries for (u,v) and (v,u).
                        s += voltages.get((u, v), 0)
                    if s % m == 0:
                        return False
                return True
                
            u, v = edges_to_assign[idx]
            for val in range(m):
                voltages[(u, v)] = val
                voltages[(v, u)] = (-val) % m
                
                # Forward checking: Check if any 6-cycle containing this edge is now "decided" to be 0?
                # This is complex. Just do full check at end or partial check.
                # Given 234 cycles, full check at each step is 234 * 6 ops = 1400 ops.
                # 52 steps. 72800 ops per path.
                # Branching factor 10.
                # This might be slow if we go deep.
                
                if backtrack(idx + 1):
                    return True
                
                del voltages[(u, v)]
                del voltages[(v, u)]
            return False
        
        return backtrack(0), voltages

    start_time = time.time()
    found = False
    while time.time() - start_time < 100:
        random.seed(int(time.time() * 1000) % 10000)
        sol, voltages = solve_voltages()
        if sol:
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
                                if min_girth < g:
                                    break
                if min_girth < g:
                    break
            
            if min_girth >= g:
                return n_vertices, lift_edges

    # Fallback
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
