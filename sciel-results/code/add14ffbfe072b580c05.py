import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    D = 11
    
    # Generate candidates with norm squared 4
    # Type 1: (±2, 0, ..., 0)
    # Type 2: (±1, ±1, ±1, ±1, 0, ...)
    
    candidates = []
    
    # Type 1
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
            
    # Type 2
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    N = len(candidates)
    all_candidates = np.array(candidates, dtype=np.int64)
    
    # Build conflict graph
    # Conflict if dot > 2
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Heuristic search for Max Independent Set
    # Greedy + Local Search
    
    best_set = np.zeros(N, dtype=bool)
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    def greedy_init(seed):
        rng_local = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = np.sum(conflicts, axis=1).astype(np.int32)
        
        while np.any(alive) and time.time() < deadline - 10:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[rng_local.integers(len(cands))]
            in_set[chosen] = True
            nbrs = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search(in_set, max_time=10.0, seed=42):
        rng_local = np.random.default_rng(seed)
        in_set = in_set.copy()
        current_size = int(in_set.sum())
        best_size = current_size
        best_in_set = in_set.copy()
        
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[conflicts[i]] += 1
            
        ls_deadline = time.time() + max_time
        stuck = 0
        
        while time.time() < ls_deadline:
            improved = False
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0: break
            
            counts = conflict_count[not_in_set]
            
            # Add zero-conflict
            zero_mask = counts == 0
            if np.any(zero_mask):
                zero_idx = np.where(zero_mask)[0]
                n_add = min(10, len(zero_idx))
                chosen = rng_local.choice(zero_idx, size=n_add, replace=False)
                for v_idx in chosen:
                    v = not_in_set[v_idx]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[conflicts[v]] += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
                stuck = 0
                continue
                
            # 1-2 Swap
            one_mask = counts == 1
            if np.any(one_mask):
                one_idx = np.where(one_mask)[0]
                max_process = min(len(one_idx), 300)
                process_idx = one_idx[:max_process]
                
                groups = defaultdict(list)
                for idx in process_idx:
                    v = not_in_set[idx]
                    nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                    if len(nbrs_in_set) == 1:
                        groups[int(nbrs_in_set[0])].append(v)
                        
                swap_found = False
                for u, vs in groups.items():
                    if len(vs) < 2: continue
                    # Check pairs in vs
                    for i in range(len(vs)):
                        for j in range(i+1, len(vs)):
                            v1, v2 = vs[i], vs[j]
                            if not conflicts[v1, v2]:
                                in_set[u] = False
                                in_set[v1] = True
                                in_set[v2] = True
                                conflict_count[conflicts[u]] -= 1
                                conflict_count[conflicts[v1]] += 1
                                conflict_count[conflicts[v2]] += 1
                                current_size += 1
                                improved = True
                                swap_found = True
                                if current_size > best_size:
                                    best_size = current_size
                                    best_in_set = in_set.copy()
                                break
                        if swap_found: break
                    if swap_found: break
                if swap_found:
                    stuck = 0
                    continue
                    
            if not improved:
                stuck += 1
                if stuck > 10:
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        n_remove = rng_local.integers(1, 6)
                        to_remove = rng_local.choice(in_set_indices, size=min(n_remove, len(in_set_indices)), replace=False)
                        for u in to_remove:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count[conflicts[u]] -= 1
                        stuck = 0
                        
        return best_in_set, best_size

    # Run multiple restarts
    for seed in range(50):
        if time.time() > deadline - 15: break
        in_set = greedy_init(seed)
        size = int(in_set.sum())
        if size > best_size:
            best_size = size
            best_set = in_set.copy()
            
        remaining = deadline - 10 - time.time()
        if remaining > 2:
            ls_time = min(remaining * 0.2, 5.0)
            ls_res, ls_size = local_search(in_set, max_time=ls_time, seed=seed)
            if ls_size > best_size:
                best_size = ls_size
                best_set = ls_res.copy()
                
    final_indices = np.where(best_set)[0]
    return all_candidates[final_indices]
