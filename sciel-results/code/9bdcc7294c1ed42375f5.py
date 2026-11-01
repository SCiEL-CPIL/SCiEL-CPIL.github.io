import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Specialized for k=5, g=9
    if k == 5 and g == 9:
        deadline = time.time() + 110
        
        # Base graph K_6
        base_n = 6
        base_edges = []
        edge_idx = {}
        for u in range(base_n):
            for v in range(u + 1, base_n):
                edge_idx[(u, v)] = len(base_edges)
                base_edges.append((u, v))
        
        num_edges = len(base_edges)
        
        def get_edge_id(u, v):
            if u > v: u, v = v, u
            return edge_idx.get((u, v), -1)
        
        # Generate non-backtracking closed walks of length 3..8
        constraints = []
        
        def gen_walks(length):
            def backtrack(v_idx, current, prev, path):
                if v_idx == length:
                    if current != 0 and prev != 0:
                        if path[1] != path[length-1]:
                            coeffs = {}
                            for i in range(length):
                                u = path[i]
                                v = path[(i+1)%length]
                                eid = get_edge_id(u, v)
                                if eid == -1: continue
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
                        if path[1] != path[length-1]:
                            coeffs = {}
                            for i in range(length):
                                u = path[i]
                                v = path[(i+1)%length]
                                eid = get_edge_id(u, v)
                                if eid == -1: continue
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

                for next_v in range(base_n):
                    if next_v == current or next_v == prev:
                        continue
                    path.append(next_v)
                    backtrack(v_idx + 1, next_v, current, path)
                    path.pop()
            
            for v1 in range(1, base_n):
                backtrack(1, v1, 0, [0, v1])

        for L in range(3, 9):
            gen_walks(L)
            
        # Precompute edge-to-constraint mapping for fast updates
        # constraint_sums[i] is the sum for constraint i
        # We store for each edge, the list of (constraint_index, coefficient)
        edge_to_cons = [[] for _ in range(num_edges)]
        for i, c_list in enumerate(constraints):
            for eid, coeff in c_list.items():
                edge_to_cons[eid].append((i, coeff))
        
        num_constraints = len(constraints)
        
        best_n = None
        best_edges = None
        
        # Try m from 1 upwards
        # We want smallest n = 6*m
        for m in range(1, 500):
            if time.time() > deadline:
                break
            
            # Local search for voltages
            # We maintain current voltage assignment and constraint sums
            # score = number of constraints with sum % m == 0
            
            def solve_m(m):
                # Random init
                volts = [random.randint(0, m-1) for _ in range(num_edges)]
                sums = [0] * num_constraints
                
                # Calculate initial sums
                for e in range(num_edges):
                    v = volts[e]
                    if v == 0: continue
                    for c_idx, coeff in edge_to_cons[e]:
                        sums[c_idx] = (sums[c_idx] + v * coeff) % m
                
                # Calculate initial score
                score = 0
                for s in sums:
                    if s == 0:
                        score += 1
                
                # Local search
                # We want to minimize score (target 0)
                # Try to flip variables to reduce score
                
                # For small m, we can be more exhaustive
                # For large m, random steps are fine
                
                # Simulated Annealing / Hill Climbing
                # Since we just need score 0, hill climbing is sufficient if landscape is not too rugged.
                # To avoid local minima, we can do random restarts or random jumps.
                
                for restart in range(50): # 50 restarts per m
                    if time.time() > deadline:
                        return None
                    
                    if restart > 0:
                        volts = [random.randint(0, m-1) for _ in range(num_edges)]
                        sums = [0] * num_constraints
                        for e in range(num_edges):
                            v = volts[e]
                            if v == 0: continue
                            for c_idx, coeff in edge_to_cons[e]:
                                sums[c_idx] = (sums[c_idx] + v * coeff) % m
                        score = sum(1 for s in sums if s == 0)
                    
                    if score == 0:
                        return volts
                    
                    # Hill climbing
                    # Iterate until no improvement or time limit
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        # Try changing each variable to all possible values?
                        # Or just random values?
                        # For small m, try all values.
                        # For large m, try random values.
                        
                        if m <= 20:
                            # Exhaustive per variable
                            for e in range(num_edges):
                                old_v = volts[e]
                                for new_v in range(m):
                                    if new_v == old_v: continue
                                    delta = (new_v - old_v) % m
                                    if delta == 0: continue
                                    
                                    # Check if this reduces score
                                    # We need to update sums and check score change
                                    # This is O(num_cons_for_e)
                                    
                                    # Temporarily apply
                                    old_sums = []
                                    for c_idx, coeff in edge_to_cons[e]:
                                        old_s = sums[c_idx]
                                        new_s = (old_s + delta * coeff) % m
                                        old_sums.append((c_idx, old_s, new_s))
                                        sums[c_idx] = new_s
                                    
                                    new_score = 0
                                    for c_idx, old_s, new_s in old_sums:
                                        if old_s == 0:
                                            if new_s != 0:
                                                new_score -= 1 # Improvement
                                            else:
                                                new_score += 1 # Worse
                                        else:
                                            if new_s == 0:
                                                new_score += 1 # Worse
                                            else:
                                                new_score -= 1 # Improvement? No, 0->non0 is improvement, non0->0 is worse.
                                                # Wait.
                                                # Score counts zeros.
                                                # Old score contribution: 1 if old_s==0 else 0.
                                                # New score contribution: 1 if new_s==0 else 0.
                                                # Delta score = New - Old.
                                                # We want Delta < 0.
                                                pass
                                    
                                    # Let's just recalc score for affected constraints
                                    delta_score = 0
                                    for c_idx, old_s, new_s in old_sums:
                                        if old_s == 0 and new_s != 0:
                                            delta_score -= 1
                                        elif old_s != 0 and new_s == 0:
                                            delta_score += 1
                                        
                                    if delta_score < 0:
                                        score += delta_score
                                        volts[e] = new_v
                                        improved = True
                                        break # Move to next variable
                                    elif delta_score == 0 and random.random() < 0.1:
                                        # Accept neutral move to escape local minima?
                                        score += delta_score
                                        volts[e] = new_v
                                        improved = True
                                        break
                                    else:
                                        # Revert
                                        for c_idx, old_s, new_s in old_sums:
                                            sums[c_idx] = old_s
                        else:
                            # Random moves
                            for _ in range(100):
                                e = random.randint(0, num_edges - 1)
                                old_v = volts[e]
                                new_v = random.randint(0, m-1)
                                if new_v == old_v: continue
                                delta = (new_v - old_v) % m
                                if delta == 0: continue
                                
                                old_sums = []
                                for c_idx, coeff in edge_to_cons[e]:
                                    old_s = sums[c_idx]
                                    new_s = (old_s + delta * coeff) % m
                                    old_sums.append((c_idx, old_s, new_s))
                                    sums[c_idx] = new_s
                                    
                                delta_score = 0
                                for c_idx, old_s, new_s in old_sums:
                                    if old_s == 0 and new_s != 0:
                                        delta_score -= 1
                                    elif old_s != 0 and new_s == 0:
                                        delta_score += 1
                                        
                                if delta_score < 0:
                                    score += delta_score
                                    volts[e] = new_v
                                    improved = True
                                    break
                                else:
                                    for c_idx, old_s, new_s in old_sums:
                                        sums[c_idx] = old_s
                    if score == 0:
                        return volts
                        
                return None
            
            sol = solve_m(m)
            if sol:
                n = 6 * m
                edges = []
                for u in range(base_n):
                    for i in range(m):
                        for v in range(base_n):
                            if u == v: continue
                            eid = get_edge_id(u, v)
                            if eid == -1: continue
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

        # Fallback
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)

# EVOLVE-BLOCK-END
