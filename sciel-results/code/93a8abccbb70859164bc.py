import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110

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

    # Try voltage graph lift on Clebsch graph
    if k == 5 and g == 9:
        base_n = 16
        base_edges = []
        edge_idx = {}
        for u in range(base_n):
            for v in range(u + 1, base_n):
                diff = bin(u ^ v).count('1')
                if diff == 1 or diff == 4:
                    edge_idx[(u, v)] = len(base_edges)
                    base_edges.append((u, v))
        
        num_edges = len(base_edges)
        
        def get_edge_id(u, v):
            if u > v:
                u, v = v, u
            return edge_idx.get((u, v), -1)
        
        # Build adjacency for base graph
        adj = [set() for _ in range(base_n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Generate non-backtracking closed walks of length 5..8
        constraints = []
        
        def gen_walks(length):
            def backtrack(v_idx, current, prev, path):
                if v_idx == length:
                    if current != 0 and prev != 0:
                        if path[1] != path[length - 1]:
                            coeffs = {}
                            for i in range(length):
                                u = path[i]
                                v = path[(i + 1) % length]
                                eid = get_edge_id(u, v)
                                if eid == -1:
                                    continue
                                sign = 1 if u < v else -1
                                if eid in coeffs:
                                    coeffs[eid] += sign
                                else:
                                    coeffs[eid] = sign
                            c_list = {}
                            for eid, s in coeffs.items():
                                if s != 0:
                                    c_list[eid] = s
                            if c_list:
                                constraints.append(c_list)
                    return
                
                if v_idx == length - 1:
                    if current != 0 and prev != 0:
                        path.append(0)
                        if path[1] != path[length - 1]:
                            coeffs = {}
                            for i in range(length):
                                u = path[i]
                                v = path[(i + 1) % length]
                                eid = get_edge_id(u, v)
                                if eid == -1:
                                    continue
                                sign = 1 if u < v else -1
                                if eid in coeffs:
                                    coeffs[eid] += sign
                                else:
                                    coeffs[eid] = sign
                            c_list = {}
                            for eid, s in coeffs.items():
                                if s != 0:
                                    c_list[eid] = s
                            if c_list:
                                constraints.append(c_list)
                        path.pop()
                    return

                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    path.append(next_v)
                    backtrack(v_idx + 1, next_v, current, path)
                    path.pop()
            
            for v1 in adj[0]:
                backtrack(1, v1, 0, [0, v1])

        for L in range(5, 9):
            gen_walks(L)
            if time.time() > deadline - 60:
                break
            
        num_constraints = len(constraints)
        
        # Build constraint matrix: C[num_constraints][num_edges] with coefficients
        C = np.zeros((num_constraints, num_edges), dtype=np.int32)
        for i, c_list in enumerate(constraints):
            for eid, coeff in c_list.items():
                C[i, eid] = coeff
        
        # Precompute for fast delta evaluation: for each edge e, store (constraint_indices, coeffs)
        edge_to_cons = []
        for e in range(num_edges):
            col = C[:, e]
            nz = np.nonzero(col)[0]
            edge_to_cons.append((nz, col[nz]))
        
        # Try m from small values up
        for m in range(2, 100):
            if time.time() > deadline - 30:
                break
            
            def solve_m(m):
                C_mod = C % m
                edge_to_cons_mod = []
                for e in range(num_edges):
                    col = C_mod[:, e]
                    nz = np.nonzero(col)[0]
                    edge_to_cons_mod.append((nz, col[nz]))
                
                best_volts = None
                best_hard = num_constraints + 1
                
                num_restarts = max(10, int((deadline - time.time() - 15) * 2))
                for restart in range(num_restarts):
                    if time.time() > deadline - 10:
                        break
                    
                    # Random init
                    volts = np.random.randint(0, m, size=num_edges, dtype=np.int32)
                    # sums = C_mod @ volts mod m
                    sums = (C_mod @ volts) % m
                    
                    h_score = int(np.sum(sums == 0))
                    if h_score == 0:
                        return volts.tolist()
                    
                    if h_score < best_hard:
                        best_hard = h_score
                        best_volts = volts.copy()
                    
                    # Local search with simulated annealing
                    temp = 2.0
                    no_improve = 0
                    max_no_improve = 50
                    
                    while no_improve < max_no_improve and time.time() < deadline - 10:
                        improved_this_pass = False
                        # Randomize edge order for diversity
                        edge_order = np.random.permutation(num_edges)
                        
                        for e in edge_order:
                            if time.time() > deadline - 10:
                                break
                            old_v = volts[e]
                            nz, coeffs = edge_to_cons_mod[e]
                            if len(nz) == 0:
                                continue
                            
                            best_new_v = old_v
                            best_new_score = h_score
                            best_new_soft = float('inf')
                            
                            # Evaluate all possible new values
                            for new_v in range(m):
                                if new_v == old_v:
                                    continue
                                delta = (new_v - old_v) % m
                                if delta == 0:
                                    continue
                                
                                # Compute new sums for affected constraints
                                new_sums_affected = (sums[nz] + delta * coeffs) % m
                                old_affected = sums[nz]
                                
                                # Hard score change
                                old_zeros = int(np.sum(old_affected == 0))
                                new_zeros = int(np.sum(new_sums_affected == 0))
                                new_score = h_score - old_zeros + new_zeros
                                
                                # Soft score: sum of min(s, m-s) for all constraints (smooth surrogate)
                                # Only recompute affected ones; keep a running soft score
                                # For simplicity, compute full soft score occasionally
                                if new_score < best_new_score:
                                    best_new_score = new_score
                                    best_new_v = new_v
                            
                            if best_new_v != old_v:
                                delta = (best_new_v - old_v) % m
                                sums[edge_to_cons_mod[e][0]] = (sums[edge_to_cons_mod[e][0]] + delta * edge_to_cons_mod[e][1]) % m
                                volts[e] = best_new_v
                                h_score = best_new_score
                                improved_this_pass = True
                                no_improve = 0
                                if h_score == 0:
                                    return volts.tolist()
                            else:
                                no_improve += 1
                                # Simulated annealing: occasionally accept worse moves
                                if no_improve > 20 and temp > 0.1:
                                    # Pick a random edge and random value
                                    e_rand = int(np.random.randint(num_edges))
                                    nz, coeffs = edge_to_cons_mod[e_rand]
                                    if len(nz) > 0:
                                        old_v = volts[e_rand]
                                        new_v = int(np.random.randint(0, m))
                                        if new_v != old_v:
                                            delta = (new_v - old_v) % m
                                            old_sums_affected = sums[nz].copy()
                                            sums[nz] = (sums[nz] + delta * coeffs) % m
                                            new_zeros = int(np.sum(sums[nz] == 0))
                                            old_zeros = int(np.sum(old_sums_affected == 0))
                                            new_score = h_score - old_zeros + new_zeros
                                            # Accept with probability exp(-(new_score - h_score) / temp)
                                            if new_score <= h_score or random.random() < math.exp(-(new_score - h_score) / temp):
                                                volts[e_rand] = new_v
                                                h_score = new_score
                                            else:
                                                sums[nz] = old_sums_affected
                                    temp *= 0.95
                        
                        if not improved_this_pass:
                            no_improve += 5
                            temp = max(0.1, temp * 0.9)
                    
                    if h_score < best_hard:
                        best_hard = h_score
                        best_volts = volts.copy()
                    
                    if best_hard == 0:
                        return best_volts.tolist()
                
                return None
            
            sol = solve_m(m)
            if sol:
                n = base_n * m
                edges = []
                for u in range(base_n):
                    for i in range(m):
                        for v in adj[u]:
                            eid = get_edge_id(u, v)
                            if eid == -1:
                                continue
                            if u < v:
                                phi = sol[eid]
                            else:
                                phi = -sol[eid]
                            
                            j = (i + phi) % m
                            id1 = u * m + i
                            id2 = v * m + j
                            if id1 < id2:
                                edges.append((id1, id2))
                
                return n, edges

        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)

# EVOLVE-BLOCK-END
