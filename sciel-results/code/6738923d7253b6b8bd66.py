import time
import random
import sys
import math
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(d, q):
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

    def has_girth_at_least(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (v, u) in edge_set:
                return False
            edge_set.add((u, v))
            
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        # Try to construct the 320-vertex cage using voltage graphs
        # Base graph: K5,5 minus perfect matching (10 vertices, 4-regular)
        # Voltage group: Z2^5 (32 elements)
        
        def construct_320_cage():
            # Pre-compute edge index mapping
            base_edges = []
            edge_idx = {}
            for i in range(5):
                for j in range(5):
                    if i != j:
                        idx = len(base_edges)
                        base_edges.append((i, 5 + j))
                        edge_idx[(i, 5 + j)] = idx
                        edge_idx[(5 + j, i)] = idx  # reverse direction same edge
            
            # Pre-compute all 4-cycles in the base graph
            # A 4-cycle: i1 - j1 - i2 - j2 - i1 where i1,i2 in {0..4}, j1,j2 in {0..4}
            # Edges: (i1, 5+j1), (i2, 5+j1), (i2, 5+j2), (i1, 5+j2)
            # Conditions: i1 != j1, i2 != j1, i2 != j2, i1 != j2, i1 != i2, j1 != j2
            four_cycles = []
            for i1 in range(5):
                for i2 in range(5):
                    if i1 == i2:
                        continue
                    for j1 in range(5):
                        if j1 == i1 or j1 == i2:
                            continue
                        for j2 in range(5):
                            if j2 == i1 or j2 == i2 or j2 == j1:
                                continue
                            e1 = edge_idx[(i1, 5 + j1)]
                            e2 = edge_idx[(i2, 5 + j1)]
                            e3 = edge_idx[(i2, 5 + j2)]
                            e4 = edge_idx[(i1, 5 + j2)]
                            four_cycles.append((e1, e2, e3, e4))
            
            # Pre-compute all 6-cycles in the base graph
            # A 6-cycle alternates between A and B: i1-j1-i2-j2-i3-j3-i1
            # Edges: (i1,5+j1), (i2,5+j1), (i2,5+j2), (i3,5+j2), (i3,5+j3), (i1,5+j3)
            # Conditions: all i's distinct, all j's distinct, i_p != j_p for each p
            six_cycles = []
            for i1 in range(5):
                for i2 in range(5):
                    if i2 == i1: continue
                    for i3 in range(5):
                        if i3 == i1 or i3 == i2: continue
                        for j1 in range(5):
                            if j1 == i1 or j1 == i2 or j1 == i3: continue
                            for j2 in range(5):
                                if j2 == i1 or j2 == i2 or j2 == i3 or j2 == j1: continue
                                for j3 in range(5):
                                    if j3 == i1 or j3 == i2 or j3 == i3 or j3 == j1 or j3 == j2: continue
                                    # Check edge existence
                                    e1 = edge_idx.get((i1, 5 + j1))
                                    e2 = edge_idx.get((i2, 5 + j1))
                                    e3 = edge_idx.get((i2, 5 + j2))
                                    e4 = edge_idx.get((i3, 5 + j2))
                                    e5 = edge_idx.get((i3, 5 + j3))
                                    e6 = edge_idx.get((i1, 5 + j3))
                                    if e1 is not None and e2 is not None and e3 is not None and \
                                       e4 is not None and e5 is not None and e6 is not None:
                                        six_cycles.append((e1, e2, e3, e4, e5, e6))
            
            # Pre-compute all 8-cycles in the base graph
            # This is more complex. For now, let's focus on 4 and 6 cycles first.
            # If we avoid 4 and 6 cycles, we still need to check 8 cycles in the lift.
            # But checking 8-cycles in the base is expensive. We'll verify in the lift.
            
            num_attempts = 500
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    return None
                
                # Assign random 5-bit vectors to edges
                voltages = [random.getrandbits(5) for _ in base_edges]
                
                # Check all 4-cycles
                valid = True
                for (e1, e2, e3, e4) in four_cycles:
                    s = voltages[e1] ^ voltages[e2] ^ voltages[e3] ^ voltages[e4]
                    if s == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check all 6-cycles
                for (e1, e2, e3, e4, e5, e6) in six_cycles:
                    s = voltages[e1] ^ voltages[e2] ^ voltages[e3] ^ voltages[e4] ^ voltages[e5] ^ voltages[e6]
                    if s == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Build the lift
                n_lift = 10 * 32
                lift_edges = []
                
                for idx, (u, v) in enumerate(base_edges):
                    w = voltages[idx]
                    for x in range(32):
                        u_lift = u * 32 + x
                        v_lift = v * 32 + ((x + w) % 32)
                        lift_edges.append((u_lift, v_lift))
                
                # Verify girth (this is the expensive part)
                if has_girth_at_least(n_lift, lift_edges, 10):
                    return n_lift, lift_edges
            
            return None

        result = construct_320_cage()
        if result:
            return result

        # Fall back to LUW
        d_try = 5
        n, edges = luw_construct(d_try, 4)
        if has_girth_at_least(n, edges, 10):
            return n, edges
            
        d_try = 6
        n, edges = luw_construct(d_try, 4)
        if has_girth_at_least(n, edges, 10):
            return n, edges

    # General case
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = luw_construct(d, k)
    return n, edges
