import time
import random
import sys
from itertools import product

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    def luw_construct(k, g):
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

    if k == 4 and g == 10:
        # Try Voltage Graph Construction on K4,4
        # Group: Z_10 x Z_4
        G1, G2 = 10, 4
        def add_g(v1, v2):
            return ((v1[0] + v2[0]) % G1, (v1[1] + v2[1]) % G2)
        def neg_g(v):
            return ((-v[0]) % G1, (-v[1]) % G2)
        def eq_g(v1, v2):
            return v1[0] == v2[0] and v1[1] == v2[1]
        
        edges_base = [(i, j+4) for i in range(4) for j in range(4)]
        edge_index = {e: i for i, e in enumerate(edges_base)}
        
        # Generate cycles
        cycles = []
        # 4-cycles
        for i1 in range(4):
            for i2 in range(4):
                if i1 == i2: continue
                for j1 in range(4):
                    for j2 in range(4):
                        if j1 == j2: continue
                        e1 = (i1, j1+4)
                        e2 = (i2, j1+4)
                        e3 = (i2, j2+4)
                        e4 = (i1, j2+4)
                        cycles.append([(edge_index[e1], 0), (edge_index[e2], 1), (edge_index[e3], 0), (edge_index[e4], 1)])
                        
        # 6-cycles
        for i1 in range(4):
            for i2 in range(4):
                if i1 == i2: continue
                for i3 in range(4):
                    if i1 == i3 or i2 == i3: continue
                    for j1 in range(4):
                        for j2 in range(4):
                            if j1 == j2: continue
                            for j3 in range(4):
                                if j1 == j3 or j2 == j3: continue
                                e1 = (i1, j1+4)
                                e2 = (i2, j1+4)
                                e3 = (i2, j2+4)
                                e4 = (i3, j2+4)
                                e5 = (i3, j3+4)
                                e6 = (i1, j3+4)
                                cycles.append([(edge_index[e1], 0), (edge_index[e2], 1), (edge_index[e3], 0), (edge_index[e4], 1), (edge_index[e5], 0), (edge_index[e6], 1)])
                                
        # 8-cycles
        import itertools
        for i_perm in itertools.permutations(range(4)):
            for j_perm in itertools.permutations(range(4)):
                e1 = (i_perm[0], j_perm[0]+4)
                e2 = (i_perm[1], j_perm[0]+4)
                e3 = (i_perm[1], j_perm[1]+4)
                e4 = (i_perm[2], j_perm[1]+4)
                e5 = (i_perm[2], j_perm[2]+4)
                e6 = (i_perm[3], j_perm[2]+4)
                e7 = (i_perm[3], j_perm[3]+4)
                e8 = (i_perm[0], j_perm[3]+4)
                cycles.append([(edge_index[e1], 0), (edge_index[e2], 1), (edge_index[e3], 0), (edge_index[e4], 1), (edge_index[e5], 0), (edge_index[e6], 1), (edge_index[e7], 0), (edge_index[e8], 1)])

        # Backtracking search
        voltages = [None] * 16
        assigned = [False] * 16
        
        # Map edge index to list of cycles containing it
        edge_cycles = [[] for _ in range(16)]
        for i, cycle in enumerate(cycles):
            for e_idx, _ in cycle:
                edge_cycles[e_idx].append(i)
                
        def check_constraints(e_idx):
            # Check all cycles containing e_idx that are now fully assigned
            for c_idx in edge_cycles[e_idx]:
                cycle = cycles[c_idx]
                if all(assigned[ce] for ce, _ in cycle):
                    sum_v = (0, 0)
                    for ce, dir in cycle:
                        v = voltages[ce]
                        if dir == 0:
                            sum_v = add_g(sum_v, v)
                        else:
                            sum_v = add_g(sum_v, neg_g(v))
                    if eq_g(sum_v, (0, 0)):
                        return False
        return True

        def backtrack(i):
            if time.time() > deadline:
                return False
            if i == 16:
                return True
            # Try all possible voltages for edge i
            # Shuffle for randomization
            vals = [(a, b) for a in range(G1) for b in range(G2)]
            random.shuffle(vals)
            for v in vals:
                voltages[i] = v
                assigned[i] = True
                if check_constraints(i):
                    if backtrack(i + 1):
                        return True
                assigned[i] = False
                voltages[i] = None
            return False

        if backtrack(0):
            # Construct lift
            n = 320
            edges = []
            for e_idx, (u, v) in enumerate(edges_base):
                voltage = voltages[e_idx]
                for a in range(G1):
                    for b in range(G2):
                        g_elem = (a, b)
                        u_id = u * 40 + a * 4 + b
                        g_next = add_g(g_elem, voltage)
                        v_id = v * 40 + g_next[0] * 4 + g_next[1]
                        edges.append((u_id, v_id))
            return n, edges

    return luw_construct(k, g)
