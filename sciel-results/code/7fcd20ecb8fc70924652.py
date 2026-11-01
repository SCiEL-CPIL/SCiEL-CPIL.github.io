import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    D = 11
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = 4
    
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    best_size = 0
    best_selected = []
    
    np.random.seed(42)
    
    def run_greedy_and_local_search():
        """Greedy min-degree initialization followed by efficient local search."""
        # Greedy min-degree
        remaining = np.ones(n, dtype=bool)
        deg = conflict.sum(axis=1).copy()
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            rem_degrees = deg[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            chosen_local = np.random.choice(min_indices_local)
            chosen = rem_indices[chosen_local]
            selected.append(chosen)
            
            to_remove = conflict[chosen].copy()
            to_remove[chosen] = True
            actual_remove = to_remove & remaining
            remaining[actual_remove] = False
            if np.any(actual_remove):
                deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                deg[remaining] -= deg_decrease
        
        S = set(selected)
        
        # Local search with efficient neighbor counting
        in_S = np.zeros(n, dtype=bool)
        for v in S:
            in_S[v] = True
        
        # conflict_count[v] = number of neighbors of v that are in S
        conflict_count = (conflict @ in_S.astype(np.int32)).astype(np.int32)
        
        # addable[v] = (not in_S[v]) and (conflict_count[v] == 0)
        addable = (~in_S) & (conflict_count == 0)
        
        improved = True
        iters = 0
        max_iters = 10000
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try to add a vertex
            addable_indices = np.where(addable)[0]
            if len(addable_indices) > 0:
                v = np.random.choice(addable_indices)
                in_S[v] = True
                S.add(v)
                addable[v] = False
                # Update conflict_count for neighbors of v not in S
                neighbors = np.where(conflict[v])[0]
                neighbors_not_in_S = neighbors[~in_S[neighbors]]
                if len(neighbors_not_in_S) > 0:
                    conflict_count[neighbors_not_in_S] += 1
                    newly_blocked = neighbors_not_in_S[conflict_count[neighbors_not_in_S] == 1]
                    if len(newly_blocked) > 0:
                        addable[newly_blocked] = False
                improved = True
                continue
            
            # Try 1-swap: remove u from S, add v not in S
            S_list = list(S)
            if len(S_list) == 0:
                break
            # Try up to 30 random u's
            np.random.shuffle(S_list)
            try_count = min(30, len(S_list))
            for u in S_list[:try_count]:
                # v becomes addable if conflict_count[v] == 1 and u is its only neighbor in S
                candidate_v = np.where((~in_S) & (conflict_count == 1) & conflict[:, u])[0]
                if len(candidate_v) > 0:
                    v = np.random.choice(candidate_v)
                    # Do the swap: remove u, add v
                    in_S[u] = False
                    S.discard(u)
                    in_S[v] = True
                    S.add(v)
                    
                    # Update conflict_count for removing u
                    neighbors_u = np.where(conflict[u])[0]
                    neighbors_u_not_in_S = neighbors_u[~in_S[neighbors_u]]
                    if len(neighbors_u_not_in_S) > 0:
                        conflict_count[neighbors_u_not_in_S] -= 1
                        newly_addable = neighbors_u_not_in_S[conflict_count[neighbors_u_not_in_S] == 0]
                        if len(newly_addable) > 0:
                            addable[newly_addable] = True
                    
                    # Update conflict_count for adding v
                    neighbors_v = np.where(conflict[v])[0]
                    neighbors_v_not_in_S = neighbors_v[~in_S[neighbors_v]]
                    if len(neighbors_v_not_in_S) > 0:
                        conflict_count[neighbors_v_not_in_S] += 1
                        newly_blocked = neighbors_v_not_in_S[conflict_count[neighbors_v_not_in_S] == 1]
                        if len(newly_blocked) > 0:
                            addable[newly_blocked] = False
                    addable[v] = False
                    
                    improved = True
                    break
            
            if time.time() > deadline - 5:
                break
        
        return list(S)
    
    # Run multiple restarts with time budgeting
    while time.time() < deadline:
        selected = run_greedy_and_local_search()
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
        
        if time.time() > deadline - 10:
            break
    
    result = candidates[best_selected]
    return result
