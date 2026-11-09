import time
import math
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for safety within 90s limit
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict edges
    conflicts = [[] for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i].append(j)
                conflicts[j].append(i)
                edges.append((i, j))
    
    if not edges:
        return valid_nodes
    
    # Try MILP for Minimum Vertex Cover
    milp_result = None
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
        
        m = n
        c = np.ones(m)
        num_edges = len(edges)
        
        row_indices = []
        col_indices = []
        data = []
        
        for idx, (u, v) in enumerate(edges):
            row_indices.extend([idx, idx])
            col_indices.extend([u, v])
            data.extend([-1, -1])
        
        A = csr_matrix((data, (row_indices, col_indices)), shape=(num_edges, m))
        
        constraints = LinearConstraint(A, lb=-np.inf, ub=-np.ones(num_edges))
        integrality = np.ones(m)
        bounds = Bounds(lb=0, ub=1)
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 75, "presolve": True, "mip_rel_gap": 1e-6}
        )
        
        if res.status == 0:
            cover_mask = res.x
            independent_set_indices = []
            for i in range(m):
                if cover_mask[i] < 0.5:
                    independent_set_indices.append(valid_nodes[i])
            
            if independent_set_indices:
                # Verify
                valid = True
                for i in range(len(independent_set_indices)):
                    for j in range(i, len(independent_set_indices)):
                        s = independent_set_indices[i] + independent_set_indices[j]
                        r = math.isqrt(s)
                        if r * r == s:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    milp_result = independent_set_indices
    except Exception:
        pass
    
    if milp_result is not None:
        # Try to improve with local search
        result_set = set(milp_result)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try adding each excluded node
            for v in valid_nodes:
                if v not in result_set:
                    # Check if v conflicts with any in result_set
                    can_add = True
                    for u in result_set:
                        if u + v in squares:
                            can_add = False
                            break
                    if can_add:
                        result_set.add(v)
                        improved = True
            # Try swapping: remove one, add two
            if not improved:
                for v in list(result_set):
                    for u1 in valid_nodes:
                        if u1 in result_set or u1 + v in squares:
                            continue
                        for u2 in valid_nodes:
                            if u2 in result_set or u2 == u1 or u2 + v in squares or u1 + u2 in squares:
                                continue
                            # Check u1 and u2 don't conflict with rest of set
                            ok = True
                            for w in result_set:
                                if w != v:
                                    if u1 + w in squares or u2 + w in squares:
                                        ok = False
                                        break
                            if ok:
                                result_set.remove(v)
                                result_set.add(u1)
                                result_set.add(u2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        result = list(result_set)
        if result:
            return result
    
    # Fallback: B&B with fast bit operations
    # Build conflict masks as Python ints
    conflict_masks = [0] * n
    for i in range(n):
        for j in conflicts[i]:
            conflict_masks[i] |= (1 << j)
    
    def greedy_mis():
        available = (1 << n) - 1
        chosen = 0
        while available:
            best_v = -1
            best_deg = float('inf')
            temp = available
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (conflict_masks[v] & available).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            chosen |= (1 << best_v)
            available &= ~(1 << best_v | conflict_masks[best_v])
        return chosen
    
    best_mask = greedy_mis()
    best_size = best_mask.bit_count()
    
    # Local search: try adding nodes
    def local_search(initial_mask):
        current = initial_mask
        current_size = current.bit_count()
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for v in range(n):
                if not (current & (1 << v)):
                    if not (conflict_masks[v] & current):
                        current |= (1 << v)
                        current_size += 1
                        improved = True
        return current, current.bit_count()
    
    best_mask, best_size = local_search(best_mask)
    
    # B&B with tight pruning
    def bnb(available_mask, current_size, chosen_mask, best):
        if time.time() > deadline:
            return
        # Upper bound: current + remaining available
        if current_size + available_mask.bit_count() <= best[0]:
            return
        if available_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
                best[1] = chosen_mask
            return
        
        # Pick vertex with highest degree in available
        best_v = -1
        best_deg = -1
        temp = available_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = (conflict_masks[v] & available_mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            return
        
        v_bit = 1 << best_v
        
        # Include (try first since we want to maximize)
        new_avail = available_mask & ~(v_bit | conflict_masks[best_v])
        bnb(new_avail, current_size + 1, chosen_mask | v_bit, best)
        
        # Exclude
        new_avail2 = available_mask & ~v_bit
        bnb(new_avail2, current_size, chosen_mask, best)
    
    best = [best_size, best_mask]
    all_mask = (1 << n) - 1
    bnb(all_mask, 0, 0, best)
    
    best_mask = best[1]
    result_nodes = []
    for i in range(n):
        if best_mask & (1 << i):
            result_nodes.append(valid_nodes[i])
    
    if not result_nodes:
        result_nodes = [valid_nodes[0]]
        
    return result_nodes
