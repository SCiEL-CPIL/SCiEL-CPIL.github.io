import time
import random
import math

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

    if k == 5 and g == 9:
        # Clebsch graph: 16 vertices, 5-regular, girth 5
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
            
        # Precompute edge-to-constraint mapping
        edge_to_cons = [[] for _ in range(num_edges)]
        for i, c_list in enumerate(constraints):
            for eid, coeff in c_list.items():
                edge_to_cons[eid].append((i, coeff))
        
        num_constraints = len(constraints)
        
        # Try m from small values up
        for m in range(2, 100):
            if time.time() > deadline - 30:
                break
            
            def solve_m(m):
                # Use a smooth objective: maximize sum(s * (m - s)) for all constraint sums s
                # This is a surrogate for the hard constraint s != 0.
                
                def compute_objective(volts):
                    sums = [0] * num_constraints
                    for e in range(num_edges):
                        v = volts[e]
                        if v == 0:
                            continue
                        for c_idx, coeff in edge_to_cons[e]:
                            sums[c_idx] = (sums[c_idx] + v * coeff) % m
                    obj = 0
                    for s in sums:
                        obj += s * (m - s)
                    return obj, sums
                
                def is_valid(sums):
                    return all(s != 0 for s in sums)
                
                # Random restarts
                for restart in range(50):
                    if time.time() > deadline - 10:
                        return None
                    
                    # Random init
                    volts = [random.randint(0, m - 1) for _ in range(num_edges)]
                    obj, sums = compute_objective(volts)
                    if is_valid(sums):
                        return volts
                    
                    # Local search: try changing each edge to each value, pick the one that maximizes obj
                    improved = True
                    while improved and time.time() < deadline - 10:
                        improved = False
                        # Shuffle edges to avoid bias
                        edges_order = list(range(num_edges))
                        random.shuffle(edges_order)
                        
                        for e in edges_order:
                            old_v = volts[e]
                            best_new_v = old_v
                            best_new_obj = obj
                            
                            # Try all possible values for this edge
                            for new_v in range(m):
                                if new_v == old_v:
                                    continue
                                # Compute delta
                                delta = (new_v - old_v) % m
                                if delta == 0:
                                    continue
                                
                                # Compute new objective
                                new_obj = obj
                                for c_idx, coeff in edge_to_cons[e]:
                                    old_s = sums[c_idx]
                                    new_s = (old_s + delta * coeff) % m
                                    new_obj -= old_s * (m - old_s)
                                    new_obj += new_s * (m - new_s)
                                    sums[c_idx] = new_s # Temporarily update
                                
                                if new_obj > best_new_obj:
                                    best_new_obj = new_obj
                                    best_new_v = new_v
                                
                                # Revert sums
                                for c_idx, coeff in edge_to_cons[e]:
                                    new_s = sums[c_idx]
                                    old_s = (new_s - delta * coeff) % m
                                    sums[c_idx] = old_s
                            
                            if best_new_v != old_v:
                                # Apply change
                                delta = (best_new_v - old_v) % m
                                for c_idx, coeff in edge_to_cons[e]:
                                    sums[c_idx] = (sums[c_idx] + delta * coeff) % m
                                volts[e] = best_new_v
                                obj = best_new_obj
                                improved = True
                                if is_valid(sums):
                                    return volts
                    
                    if is_valid(sums):
                        return volts
                
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
