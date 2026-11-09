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
    
    # Build conflict masks as Python ints (needed for both MILP path and B&B)
    conflict_masks = [0] * n
    for i in range(n):
        for j in conflicts[i]:
            conflict_masks[i] |= (1 << j)
    
    # Try MILP for Minimum Vertex Cover with tighter time limit
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
        
        # Reduced time limit to leave room for B&B fallback
        remaining = deadline - time.time()
        milp_time = min(40, max(10, remaining - 30))  # Leave at least 30s for B&B
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": milp_time, "presolve": True, "mip_rel_gap": 1e-6}
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
        # Efficient local search using bit masks
        result_set = set(milp_result)
        result_mask = 0
        for v in milp_result:
            idx = node_index[v]
            result_mask |= (1 << idx)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding each excluded node
            for idx in range(n):
                if not (result_mask & (1 << idx)):
                    if not (conflict_masks[idx] & result_mask):
                        result_mask |= (1 << idx)
                        result_set.add(valid_nodes[idx])
                        improved = True
            # Try 1-for-2 swaps
            if not improved:
                for idx_remove in range(n):
                    if not (result_mask & (1 << idx_remove)):
                        continue
                    v_remove = valid_nodes[idx_remove]
                    # Nodes that conflict with v_remove are candidates to add
                    candidates = [idx for idx in range(n) 
                                  if not (result_mask & (1 << idx)) 
                                  and not (conflict_masks[idx_remove] & (1 << idx))]
                    # Filter: candidate must not conflict with v_remove
                    candidates = [idx for idx in candidates 
                                  if not (conflict_masks[idx] & (1 << idx_remove))]
                    
                    for i1 in range(len(candidates)):
                        if time.time() > deadline - 5:
                            break
                        c1 = candidates[i1]
                        for i2 in range(i1 + 1, len(candidates)):
                            c2 = candidates[i2]
                            # Check c1 and c2 don't conflict with each other
                            if conflict_masks[c1] & (1 << c2):
                                continue
                            # Check c1 and c2 don't conflict with rest of set (excluding v_remove)
                            rest_mask = result_mask & ~(1 << idx_remove)
                            if (conflict_masks[c1] & rest_mask) or (conflict_masks[c2] & rest_mask):
                                continue
                            # Valid swap
                            result_mask = (result_mask & ~(1 << idx_remove)) | (1 << c1) | (1 << c2)
                            result_set.discard(v_remove)
                            result_set.add(valid_nodes[c1])
                            result_set.add(valid_nodes[c2])
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
        
        result = list(result_set)
        if result:
            # Also run B&B to see if we can do better
            best_from_milp = len(result)
            bnb_result = _run_bnb(conflict_masks, n, best_from_milp, deadline, valid_nodes, node_index)
            if bnb_result is not None and len(bnb_result) > best_from_milp:
                return bnb_result
            return result
    
    # Fallback: B&B with fast bit operations and improved bounds
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
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for v in range(n):
                if not (current & (1 << v)):
                    if not (conflict_masks[v] & current):
                        current |= (1 << v)
                        improved = True
        return current, current.bit_count()
    
    best_mask, best_size = local_search(best_mask)
    
    # Run B&B
    bnb_result = _run_bnb(conflict_masks, n, best_size, deadline, valid_nodes, node_index)
    if bnb_result is not None:
        return bnb_result
    
    best_mask = best_mask  # fallback to local search result
    result_nodes = []
    for i in range(n):
        if best_mask & (1 << i):
            result_nodes.append(valid_nodes[i])
    
    if not result_nodes:
        result_nodes = [valid_nodes[0]]
        
    return result_nodes


def _run_bnb(conflict_masks, n, initial_best_size, deadline, valid_nodes, node_index):
    """Run B&B with greedy coloring upper bound."""
    
    def greedy_color_bound(available_mask):
        """Compute upper bound on independent set size using greedy coloring of complement."""
        # For MIS, we can use a greedy coloring bound on the conflict graph
        # The number of colors needed >= size of max independent set
        # Actually, we use: alpha(G) <= sum of 1/(d(v)+1) type bounds, or
        # simpler: greedy coloring gives an upper bound on clique number,
        # but for independent set we use: alpha <= n - min_vertex_cover
        # Better: use the fact that if we greedily color, each color class is
        # an independent set, so the max color class size is a lower bound on alpha.
        # For upper bound: alpha(G) <= chi(G) * alpha(color_class) ... not helpful.
        
        # Simple effective bound: process vertices, each vertex can contribute at most 1,
        # but if two vertices are adjacent, they can't both be in the independent set.
        # Use: repeatedly pick a vertex, add 1 to bound, remove it and its neighbors.
        # This gives an upper bound on the independent set.
        
        bound = 0
        remaining = available_mask
        while remaining:
            bound += 1
            # Pick a vertex with maximum degree in remaining (to remove more)
            best_v = -1
            best_deg = -1
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (conflict_masks[v] & remaining).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            remaining &= ~(1 << best_v | conflict_masks[best_v])
        return bound
    
    def bnb(available_mask, current_size, chosen_mask, best):
        if time.time() > deadline - 2:
            return
        
        # Upper bound using greedy coloring
        ub = current_size + greedy_color_bound(available_mask)
        if ub <= best[0]:
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
    
    best = [initial_best_size, 0]
    all_mask = (1 << n) - 1
    bnb(all_mask, 0, 0, best)
    
    if best[0] > initial_best_size:
        result_nodes = []
        for i in range(n):
            if best[1] & (1 << i):
                result_nodes.append(valid_nodes[i])
        if result_nodes:
            return result_nodes
    
    return None
