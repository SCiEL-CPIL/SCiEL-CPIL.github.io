import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    D = 11
    
    # 1. Generate Candidate Pool: Weight-4 vectors
    # Norm^2 = 4. Max norm in set is 2.
    # Constraint: ||v_i - v_j||^2 >= 4 for all i != j.
    # For weight-4 vectors, this means dot(v_i, v_j) <= 2.
    
    candidates_list = []
    supports_list = []
    
    # Map from support (tuple of positions) to list of vector indices
    support_to_indices = {}
    
    idx = 0
    for positions in itertools.combinations(range(D), 4):
        sup_tuple = tuple(positions)
        support_to_indices[sup_tuple] = []
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
            support_to_indices[sup_tuple].append(idx)
            idx += 1
            
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)
    
    # 2. Precompute Conflict Matrix
    c_int = candidates.astype(np.int32)
    dot = c_int @ c_int.T
    
    # Conflict if dot > 2
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Precompute conflict neighbor lists
    conflict_neighbors = [np.where(conflict[i] == 1)[0] for i in range(n)]
    
    # 3. Support-Level Graph Construction
    # Nodes: 330 supports
    # Edge if intersection size >= 3
    supports = list(support_to_indices.keys())
    num_supports = len(supports)
    
    # Map support tuple to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Build support conflict matrix
    sup_conflict = np.zeros((num_supports, num_supports), dtype=np.uint8)
    for i in range(num_supports):
        for j in range(i + 1, num_supports):
            inter = len(set(supports[i]) & set(supports[j]))
            if inter >= 3:
                sup_conflict[i, j] = 1
                sup_conflict[j, i] = 1
                
    # 4. Find Independent Set in Support Graph
    # Use a greedy heuristic with local search on the support graph
    # This is a much smaller graph (330 nodes)
    
    def solve_support_mis():
        """Find a large independent set in the support conflict graph."""
        best_set = set()
        best_size = 0
        
        rng = np.random.default_rng(42)
        
        # Multiple restarts
        for restart in range(20):
            if time.time() > deadline - 30.0:
                break
                
            selected = np.zeros(num_supports, dtype=np.uint8)
            
            # Greedy initialization
            remaining = np.ones(num_supports, dtype=np.uint8)
            while True:
                rem_idx = np.where(remaining > 0)[0]
                if len(rem_idx) == 0:
                    break
                # Pick node with min degree in remaining subgraph
                deg = (sup_conflict * remaining[np.newaxis, :]).sum(axis=1)
                deg[remaining == 0] = num_supports + 1
                min_deg = deg.min()
                candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
                if len(candidates_to_pick) == 0:
                    break
                pick_idx = rng.choice(candidates_to_pick)
                selected[pick_idx] = 1
                remaining[pick_idx] = 0
                remaining[sup_conflict[pick_idx] == 1] = 0
                
            # Local Search on Support Graph
            # Move: Remove 1 support, Add 2 supports
            improved = True
            max_moves = 100
            move_count = 0
            while improved and move_count < max_moves and time.time() < deadline - 30.0:
                improved = False
                move_count += 1
                sel_indices = np.where(selected == 1)[0]
                if len(sel_indices) == 0:
                    break
                
                order = rng.permutation(len(sel_indices))
                for idx in order:
                    v = sel_indices[idx]
                    # Candidates u compatible with S \ {v}
                    # Incompat count for u w.r.t S
                    # Since graph is small, we can just check directly
                    mask_in_S = selected == 1
                    # u is compatible with S\{v} if:
                    # 1. sup_conflict[u, S\{v}] == 0
                    # Let's compute conflicts with S
                    conflicts_with_S = sup_conflict[:, sel_indices].sum(axis=1)
                    # u is compatible with S\{v} if:
                    # conflicts_with_S[u] == 0 (compatible with all)
                    # OR conflicts_with_S[u] == 1 AND sup_conflict[u, v] == 1 (conflict only with v)
                    
                    cond_0 = (conflicts_with_S == 0)
                    cond_1 = (conflicts_with_S == 1) & (sup_conflict[:, v] == 1)
                    Cand_v = (cond_0 | cond_1) & (selected == 0)
                    
                    cand_indices = np.where(Cand_v > 0)[0]
                    if len(cand_indices) < 2:
                        continue
                    
                    # Find pair u, w in cand_indices that are compatible
                    found_pair = False
                    u_val = -1
                    w_val = -1
                    # Sample pairs
                    num_samples = min(100, len(cand_indices) * 2)
                    for _ in range(num_samples):
                        i_u = rng.integers(len(cand_indices))
                        i_w = rng.integers(len(cand_indices))
                        if i_u == i_w:
                            continue
                        u = cand_indices[i_u]
                        w = cand_indices[i_w]
                        if sup_conflict[u, w] == 0:
                            u_val = u
                            w_val = w
                            found_pair = True
                            break
                            
                    if found_pair:
                        u = u_val
                        w = w_val
                        selected[v] = 0
                        selected[u] = 1
                        selected[w] = 1
                        improved = True
                        break
                        
            cur_size = selected.sum()
            if cur_size > best_size:
                best_size = cur_size
                best_set = set(np.where(selected == 1)[0])
                
        return best_set

    best_supports = solve_support_mis()
    
    # 5. Initialize Vector Selection from Best Supports
    # Include ALL 16 vectors for each selected support
    selected = np.zeros(n, dtype=np.uint8)
    for sup_idx in best_supports:
        vec_indices = support_to_indices[supports[sup_idx]]
        for v_idx in vec_indices:
            selected[v_idx] = 1
            
    # 6. Local Search on Full Vector Graph
    # Use 1-for-2 moves
    # This refines the solution, potentially adding vectors from non-selected supports
    # or swapping sign patterns
    
    def local_search(selected):
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected
            
        # Initialize incompat
        incompat = np.zeros(n, dtype=np.int32)
        for v in sel_indices:
            incompat[conflict_neighbors[v]] += 1
            
        improved = True
        max_moves = 2000
        move_count = 0
        rng = np.random.default_rng(123)
        
        while improved and move_count < max_moves and time.time() < deadline - 10.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            if k_sel == 0:
                break
                
            # Try 1-for-2 moves
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 10.0:
                    break
                    
                v = sel_indices[idx]
                
                mask_in_S = selected == 1
                
                cond_0 = (incompat == 0)
                cond_1 = (incompat == 1) & (conflict[:, v] == 1)
                Cand_v = (cond_0 | cond_1) & (~mask_in_S)
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                # Find compatible pair in cand_indices
                found_pair = False
                u_val = -1
                w_val = -1
                
                # Optimization: Check first few candidates thoroughly, then sample
                check_limit = min(50, len(cand_indices))
                for i_u in range(check_limit):
                    u = cand_indices[i_u]
                    conflicts_with_cands = conflict[u, cand_indices]
                    compat_positions = np.where(conflicts_with_cands == 0)[0]
                    compat_positions = compat_positions[compat_positions != i_u]
                    if len(compat_positions) > 0:
                        w = cand_indices[compat_positions[0]]
                        u_val = u
                        w_val = w
                        found_pair = True
                        break
                        
                if not found_pair and len(cand_indices) > check_limit:
                    # Sample
                    num_samples = 100
                    for _ in range(num_samples):
                        i_u = rng.integers(len(cand_indices))
                        i_w = rng.integers(len(cand_indices))
                        if i_u == i_w:
                            continue
                        u = cand_indices[i_u]
                        w = cand_indices[i_w]
                        if conflict[u, w] == 0:
                            u_val = u
                            w_val = w
                            found_pair = True
                            break
                            
                if found_pair:
                    u = u_val
                    w = w_val
                    
                    # Update incompat
                    incompat[conflict_neighbors[v]] -= 1
                    incompat[conflict_neighbors[u]] += 1
                    incompat[conflict_neighbors[w]] += 1
                    
                    selected[v] = 0
                    selected[u] = 1
                    selected[w] = 1
                    improved = True
                    break
            
        return selected

    # Run local search
    final_selected = local_search(selected)
    
    # Extract Result
    sel_indices = np.where(final_selected == 1)[0]
    if len(sel_indices) == 0:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
        
    result = candidates[sel_indices.astype(np.int64)]
    return result.astype(np.int64)

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} vectors")
