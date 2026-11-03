import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Generate candidate integer vectors
    candidates = []
    
    # Type 1: One 2, rest 0 (squared norm = 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Two 1s, rest 0 (squared norm = 2)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type 3: One 1, rest 0 (squared norm = 1)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    # Type 4: Three 1s, rest 0 (squared norm = 3)
    for idxs in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
    
    # Type 5: One 2 and one 1 (squared norm = 5)
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = 2 * s1
                    v[j] = s2
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute squared norms
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix
    max_norm_sq = np.maximum(norms_sq[:, None], norms_sq[None, :])
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * M
    Conflict = (dist_sq < max_norm_sq).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Precompute neighbor arrays and sets
    neighbors = [np.where(Conflict[i])[0] for i in range(N)]
    neighbor_sets = [set(neighbors[i]) for i in range(N)]
    
    best_set = None
    best_size = 0
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
        
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
            active_deg = Conflict[active_indices, :].sum(axis=1)
            min_deg = np.min(active_deg)
            min_candidates = active_indices[active_deg == min_deg]
            idx = int(rng.choice(min_candidates))
            
            selected.append(idx)
            
            remove_mask = np.zeros(N, dtype=bool)
            remove_mask[idx] = True
            remove_mask[neighbors[idx]] = True
            active[remove_mask] = False
        
        return selected
    
    def local_search(selected_indices):
        if time.time() > deadline:
            return set(selected_indices)
        
        selected_set = set(selected_indices)
        improved = True
        max_rounds = 15
        
        while improved and time.time() < deadline:
            improved = False
            
            rng_ls = np.random.default_rng()
            order = list(selected_set)
            rng_ls.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    break
                if i not in selected_set:
                    continue
                
                rest = selected_set - {i}
                
                cand_js = [j for j in neighbors[i] if j not in selected_set]
                
                if len(cand_js) < 2:
                    continue
                
                compat = [j for j in cand_js if not neighbor_sets[j].intersection(rest)]
                
                if len(compat) < 2:
                    continue
                
                for p in range(len(compat)):
                    j1 = compat[p]
                    for q in range(p + 1, len(compat)):
                        j2 = compat[q]
                        if j2 not in neighbor_sets[j1]:
                            selected_set.remove(i)
                            selected_set.add(j1)
                            selected_set.add(j2)
                            improved = True
                            break
                    if improved:
                        break
                
                if improved:
                    break
            
            if time.time() > deadline:
                break
            for v in range(N):
                if time.time() > deadline:
                    break
                if v in selected_set:
                    continue
                if not neighbor_sets[v].intersection(selected_set):
                    selected_set.add(v)
                    improved = True
        
        return selected_set
    
    num_restarts = 200
    for s in range(num_restarts):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
        
        res_set = local_search(res)
        
        if len(res_set) > best_size:
            best_size = len(res_set)
            best_set = res_set
    
    if best_set is None:
        best_set = set()
        for i in range(N):
            if not neighbor_sets[i].intersection(best_set):
                best_set.add(i)
    
    return candidates[list(best_set)].astype(np.int64)
