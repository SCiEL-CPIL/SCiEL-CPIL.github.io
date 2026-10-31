import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute dot products
    dots = vectors_v4 @ vectors_v4.T
    
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Greedy MIS with multiple strategies, then local search
    rng = np.random.default_rng(42)
    best_selected = None
    best_size = 0
    
    # Strategy 1: Greedy with min-degree
    alive = np.ones(N4, dtype=bool)
    selected = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    while np.any(alive):
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        selected.append(chosen)
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        alive[to_remove] = False
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_selected = selected.copy()
    
    # Strategy 2: Greedy with random ordering
    alive = np.ones(N4, dtype=bool)
    selected = []
    order = rng.permutation(N4)
    for idx in order:
        if alive[idx]:
            selected.append(idx)
            neighbors = np.where(conflicts[idx])[0]
            to_remove = np.unique(np.concatenate(([idx], neighbors)))
            alive[to_remove] = False
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_selected = selected.copy()
    
    # Strategy 3: Greedy with max-neighborhood (pick node with fewest remaining neighbors after selection)
    alive = np.ones(N4, dtype=bool)
    selected = []
    remaining = set(range(N4))
    
    while remaining:
        # For each candidate, compute how many of its alive neighbors would be removed
        best_cand = None
        best_removed = -1
        best_tiebreak = -1
        for cand in list(remaining):
            neighbors = np.where(conflicts[cand])[0]
            alive_neighbors = [n for n in neighbors if n in remaining]
            removed_count = len(alive_neighbors)
            # Prefer removing more neighbors (higher degree in remaining graph)
            # But also consider tie-break: prefer lower index for determinism
            if removed_count > best_removed or (removed_count == best_removed and cand < best_tiebreak):
                best_removed = removed_count
                best_cand = cand
                best_tiebreak = cand
        
        if best_cand is None:
            break
        
        selected.append(best_cand)
        remaining.discard(best_cand)
        neighbors = np.where(conflicts[best_cand])[0]
        for n in neighbors:
            remaining.discard(n)
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_selected = selected.copy()
    
    # Local search: try to swap out one vector for two, or two for three
    if best_selected is not None:
        selected_arr = np.array(best_selected)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n_selected = len(selected_arr)
            
            # Try removing one selected vector and adding others
            for i in range(n_selected):
                if time.time() > deadline - 5:
                    break
                remove_idx = selected_arr[i]
                remaining = np.setdiff1d(selected_arr, [remove_idx])
                
                # Build conflict mask for remaining selected
                conflict_with_remaining = np.zeros(N4, dtype=bool)
                for idx in remaining:
                    conflict_with_remaining |= conflicts[idx]
                
                # Candidates: not in selected, not conflicting with remaining
                in_selected = np.zeros(N4, dtype=bool)
                in_selected[selected_arr] = True
                candidates = np.where(~conflict_with_remaining & ~in_selected)[0]
                
                if len(candidates) < 2:
                    continue
                
                # From candidates, find maximum independent set greedily
                # Use a simple greedy: repeatedly pick node with min degree in candidate subgraph
                cand_set = set(candidates.tolist())
                temp_selected = []
                remaining_cands = set(candidates.tolist())
                
                while remaining_cands:
                    # Find node with min degree in remaining_cands subgraph
                    min_deg = float('inf')
                    best_node = None
                    for node in remaining_cands:
                        deg = sum(1 for other in remaining_cands if conflicts[node, other])
                        if deg < min_deg:
                            min_deg = deg
                            best_node = node
                    
                    if best_node is None:
                        break
                    
                    temp_selected.append(best_node)
                    remaining_cands.discard(best_node)
                    neighbors = np.where(conflicts[best_node])[0]
                    for n in neighbors:
                        remaining_cands.discard(n)
                
                if len(temp_selected) >= 2:
                    # Perform the swap
                    new_arr = np.concatenate([np.delete(selected_arr, i), np.array(temp_selected)])
                    # Verify no conflicts in new set
                    valid = True
                    for a in range(len(new_arr)):
                        for b in range(a+1, len(new_arr)):
                            if conflicts[new_arr[a], new_arr[b]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        selected_arr = new_arr
                        improved = True
                        break
            
            if time.time() > deadline - 5:
                break
        
        best_selected = selected_arr.tolist()
    
    result_v4 = vectors_v4[best_selected]
    
    # Combine V4 and V1
    # V1 dot V4: V1 has one ±2, V4 has four ±1.
    # Dot = ±2 * (V4 component at that position) which is -2, 0, or 2
    # Conflict if dot > 2: only if dot = 3,4,... impossible since max is 2
    # So all V1 are compatible with all V4.
    # V1 among themselves: same position same sign: dot=4 (conflict)
    # Same position opposite sign: dot=-4 (ok)
    # Different positions: dot=0 (ok)
    # So we can take both +2 and -2 for each position: 22 vectors total.
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
