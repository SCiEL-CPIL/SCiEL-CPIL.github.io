import time
import numpy as np
from collections import deque
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # LUW graph construction (Fallback)
    def luw_construct(k, g):
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

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def check_degree(adj, k):
        return all(len(a) == k for a in adj)

    def girth_of_graph(n, adj, k):
        if not check_degree(adj, k):
            return 0
        best = float('inf')
        for s in range(n):
            if time.time() > deadline:
                break
            dist = {}
            dq = deque()
            for nb in adj[s]:
                state = (nb, s)
                dist[state] = 1
                dq.append(state)
            while dq:
                curr, prev = dq.popleft()
                d = dist[(curr, prev)]
                if curr == s:
                    cycle_len = d + 1
                    if cycle_len < best:
                        best = cycle_len
                    if best <= 3:
                        return best
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    state = (nb, curr)
                    if state not in dist:
                        dist[state] = d + 1
                        dq.append(state)
        return best

    # Voltage graph lift: K_{3,3} base with Z_m x Z_m voltages
    # K_{3,3} edges:
    # U: 0,1,2 ; V: 3,4,5
    # Edges: (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)
    # Index mapping for voltage array:
    # 0: (0,3), 1: (0,4), 2: (0,5)
    # 3: (1,3), 4: (1,4), 5: (1,5)
    # 6: (2,3), 7: (2,4), 8: (2,5)
    
    # Cycles in K_{3,3}
    # 4-cycles: (u1, v1, u2, v2)
    # Edges: (u1, v1), (u2, v1), (u2, v2), (u1, v2)
    # Indices: 
    # u1=0, u2=1: v1=3, v2=4 -> 0, 3, 4, 1 (Sum: v0 - v3 - v4 + v1? Orientation matters)
    # Let's define the voltage sum for a cycle as sum of voltages along the path.
    # For a 4-cycle u1-v1-u2-v2-u1:
    # Path: u1 -> v1 (vol e1), v1 -> u2 (vol -e2? No, edge is undirected in base, but voltage is directed in lift)
    # In the lift, edge (u, v) with voltage w connects (u, i) to (v, i+w).
    # So going u->v adds w. Going v->u adds -w.
    # Cycle u1->v1->u2->v2->u1:
    # u1->v1: +v_{u1,v1}
    # v1->u2: -v_{u2,v1}
    # u2->v2: +v_{u2,v2}
    # v2->u1: -v_{u1,v2}
    # Sum: v_{u1,v1} - v_{u2,v1} + v_{u2,v2} - v_{u1,v2}
    
    # We want this sum to be non-zero in Z_m x Z_m.
    
    def voltage_lift(m, voltages):
        n = 6 * m
        edges = []
        # voltages is a 9x2 array or list of 9 tuples
        edge_map = [
            (0, 3, voltages[0]), (0, 4, voltages[1]), (0, 5, voltages[2]),
            (1, 3, voltages[3]), (1, 4, voltages[4]), (1, 5, voltages[5]),
            (2, 3, voltages[6]), (2, 4, voltages[7]), (2, 5, voltages[8])
        ]
        for bu, bv, vol in edge_map:
            v1, v2 = vol
            for j in range(m):
                u = bu * m + j
                v = bv * m + (j + v1) % m
                # Wait, the second coordinate of the group element also changes?
                # If group is Z_m x Z_m, and voltage is (a,b), then (u, j) -> (v, j+a) is not enough.
                # The vertex in the lift is (base_vertex, group_element).
                # So u is (bu, g). v is (bv, g + vol).
                # We need to encode the group element.
                # Let's use a single index for the group element: 0..m^2-1.
                # This makes n = 6 * m^2.
                # For m=10, n=600. For m=7, n=294.
                # This is a better strategy: use Z_m x Z_m.
                pass
        # Revised voltage lift for Z_m x Z_m
        return n, edges

    def voltage_lift_2d(m, voltages):
        n = 6 * m * m
        edges = []
        # Vertices: (base_u, i, j) where i,j in 0..m-1
        # Index: ((base_u * m + i) * m + j)
        # Edge (bu, bv) with voltage (a, b):
        # (bu, i, j) connected to (bv, (i+a)%m, (j+b)%m)
        
        edge_map = [
            (0, 3, voltages[0]), (0, 4, voltages[1]), (0, 5, voltages[2]),
            (1, 3, voltages[3]), (1, 4, voltages[4]), (1, 5, voltages[5]),
            (2, 3, voltages[6]), (2, 4, voltages[7]), (2, 5, voltages[8])
        ]
        
        for bu, bv, vol in edge_map:
            a, b = vol
            for i in range(m):
                for j in range(m):
                    u_idx = (bu * m + i) * m + j
                    v_i = (i + a) % m
                    v_j = (j + b) % m
                    v_idx = (bv * m + v_i) * m + v_j
                    edges.append((u_idx, v_idx))
        return n, edges

    def check_voltage_constraints_2d(voltages, m):
        # voltages: list of 9 tuples (a,b)
        # Check 4-cycles
        # u1, u2 in {0,1,2}; v1, v2 in {3,4,5}
        # Edges: (u1,v1) idx e1, (u2,v1) idx e2, (u2,v2) idx e3, (u1,v2) idx e4
        # Sum: v_e1 - v_e2 + v_e3 - v_e4
        # In Z_m x Z_m, sum is ((a1-a2+a3-a4)%m, (b1-b2+b3-b4)%m)
        # Must not be (0,0)
        
        # Map (u, v) to edge index
        # u in 0..2, v in 0..2 (local index for V)
        # v_global = v + 3
        # Edge index:
        # (0,0)->0, (0,1)->1, (0,2)->2
        # (1,0)->3, (1,1)->4, (1,2)->5
        # (2,0)->6, (2,1)->7, (2,2)->8
        
        def get_edge_idx(u, v_local):
            return u * 3 + v_local
            
        for u1 in range(3):
            for u2 in range(u1+1, 3):
                for v1 in range(3):
                    for v2 in range(v1+1, 3):
                        e1 = get_edge_idx(u1, v1)
                        e2 = get_edge_idx(u2, v1)
                        e3 = get_edge_idx(u2, v2)
                        e4 = get_edge_idx(u1, v2)
                        
                        sum_a = (voltages[e1][0] - voltages[e2][0] + voltages[e3][0] - voltages[e4][0]) % m
                        sum_b = (voltages[e1][1] - voltages[e2][1] + voltages[e3][1] - voltages[e4][1]) % m
                        if sum_a == 0 and sum_b == 0:
                            return False
        
        # Check 6-cycles
        # u1, u2, u3 distinct; v1, v2, v3 distinct
        # Cycle: u1-v1-u2-v2-u3-v3-u1
        # Edges: (u1,v1), (u2,v1), (u2,v2), (u3,v2), (u3,v3), (u1,v3)
        # Sum: v1 - v2 + v3 - v4 + v5 - v6
        # Actually:
        # u1->v1: +v(u1,v1)
        # v1->u2: -v(u2,v1)
        # u2->v2: +v(u2,v2)
        # v2->u3: -v(u3,v2)
        # u3->v3: +v(u3,v3)
        # v3->u1: -v(u1,v3)
        
        for u1 in range(3):
            for u2 in range(3):
                if u1 == u2: continue
                for u3 in range(3):
                    if u1 == u3 or u2 == u3: continue
                    for v1 in range(3):
                        for v2 in range(3):
                            if v1 == v2: continue
                            for v3 in range(3):
                                if v1 == v3 or v2 == v3: continue
                                
                                e1 = get_edge_idx(u1, v1)
                                e2 = get_edge_idx(u2, v1)
                                e3 = get_edge_idx(u2, v2)
                                e4 = get_edge_idx(u3, v2)
                                e5 = get_edge_idx(u3, v3)
                                e6 = get_edge_idx(u1, v3)
                                
                                sum_a = (voltages[e1][0] - voltages[e2][0] + voltages[e3][0] - voltages[e4][0] + voltages[e5][0] - voltages[e6][0]) % m
                                sum_b = (voltages[e1][1] - voltages[e2][1] + voltages[e3][1] - voltages[e4][1] + voltages[e5][1] - voltages[e6][1]) % m
                                if sum_a == 0 and sum_b == 0:
                                    return False
        return True

    # Try voltage lifts
    best_result = None
    best_n = float('inf')

    # LUW baseline
    n_luw, edges_luw = luw_construct(k, g)
    adj_luw = build_adj(n_luw, edges_luw)
    if check_degree(adj_luw, k):
        girth_luw = girth_of_graph(n_luw, adj_luw, k)
        if girth_luw >= g:
            best_n = n_luw
            best_result = (n_luw, edges_luw)

    # Try voltage lifts for k=4, g=10
    if k == 4 and g == 10:
        # n = 6 * m^2. 
        # Moore bound 243. 6*m^2 >= 243 => m^2 >= 40.5 => m >= 7.
        # Try m from 7 to 15.
        for m in range(7, 16):
            if time.time() > deadline - 5:
                break
            if 6 * m * m >= best_n:
                break

            found = False
            # Use local search to find valid voltages
            # Start with random voltages
            for restart in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Random initialization
                voltages = [(random.randint(0, m-1), random.randint(0, m-1)) for _ in range(9)]
                
                # Local search: minimize number of short cycles
                # We can just check if it's valid. If not, try to fix.
                # For simplicity, just random restarts with a few mutations.
                valid = check_voltage_constraints_2d(voltages, m)
                
                if not valid:
                    # Try to improve by random mutation
                    for _ in range(50):
                        if time.time() > deadline - 5:
                            break
                        # Pick a random edge and change its voltage
                        idx = random.randint(0, 8)
                        voltages[idx] = (random.randint(0, m-1), random.randint(0, m-1))
                        if check_voltage_constraints_2d(voltages, m):
                            valid = True
                            break
                
                if valid:
                    n_lift, edges_lift = voltage_lift_2d(m, voltages)
                    adj_lift = build_adj(n_lift, edges_lift)
                    if check_degree(adj_lift, k):
                        girth_lift = girth_of_graph(n_lift, adj_lift, k)
                        if girth_lift >= g:
                            if n_lift < best_n:
                                best_n = n_lift
                                best_result = (n_lift, edges_lift)
                                found = True
                    if found:
                        break

    if best_result is not None:
        return best_result

    # Fallback
    return n_luw, edges_luw
