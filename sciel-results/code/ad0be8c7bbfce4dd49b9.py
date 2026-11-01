"""D11 kissing number: constructive + local search refinement."""
import numpy as np
import itertools
import time
import heapq

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    vectors = []
    
    # Type W1: One entry ±2, norm² = 4, sum even
    for i in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[i] = 2 * sign
            vectors.append(v)
    
    # Type W4: Four entries ±1, norm² = 4, sum even
    for positions in itertools.combinations(range(D), 4):
        for sign_mask in range(16):
            n_plus = bin(sign_mask).count('1')
            if n_plus % 2 == 0:
                v = [0] * D
                for j, pos in enumerate(positions):
                    v[pos] = 1 if (sign_mask >> j) & 1 else -1
                vectors.append(v)
    
    arr = np.array(vectors, dtype=np.int64)
    N = len(arr)
    
    # Build conflict graph: conflict if dot > 2
    dots = arr @ arr.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    def greedy_min_degree():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            alive[chosen] = False
            nbrs = neighbors[chosen]
            alive[nbrs] = False
            
            # Update degrees
            rem = np.where(alive)[0]
            if len(rem) > 0:
                removed = np.zeros(N, dtype=bool)
                removed[chosen] = True
                removed[nbrs] = True
                for i in rem:
                    degrees[i] -= removed[neighbors[i]].sum()
        
        return in_set
    
    def local_search(in_set_init, max_time=2.0):
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best = in_set.copy()
        local_best_size = current_size
        ls_deadline = time.time() + max_time
        
        # Conflict counts
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        no_improve = 0
        while time.time() < ls_deadline:
            improved = False
            
            # Add zero-conflict vertices
            not_in = np.where(~in_set)[0]
            if len(not_in) > 0:
                zero_conf = not_in[conflict_count[not_in] == 0]
                while len(zero_conf) > 0 and time.time() < ls_deadline:
                    v = zero_conf[rng.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best = in_set.copy()
                    not_in = np.where(~in_set)[0]
                    if len(not_in) == 0: break
                    zero_conf = not_in[conflict_count[not_in] == 0]
            
            # 1-for-1 swaps
            if time.time() < ls_deadline:
                in_idx = np.where(in_set)[0]
                not_in = np.where(~in_set)[0]
                if len(in_idx) > 0 and len(not_in) > 0:
                    for _ in range(200):
                        if time.time() >= ls_deadline: break
                        u = in_idx[rng.integers(len(in_idx))]
                        v_cands = np.where(conflicts[u] & ~in_set)[0]
                        if len(v_cands) == 0: continue
                        v = v_cands[rng.integers(len(v_cands))]
                        v_conf = np.where(conflicts[v] & in_set)[0]
                        if len(v_conf) == 1 and v_conf[0] == u:
                            in_set[u] = False
                            in_set[v] = True
                            conflict_count[neighbors[u]] -= 1
                            conflict_count[neighbors[v]] += 1
                            current_size = in_set.sum()
                            improved = True
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best = in_set.copy()
                            in_idx = np.where(in_set)[0]
                            not_in = np.where(~in_set)[0]
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 20:
                    in_idx = np.where(in_set)[0]
                    if len(in_idx) >= 2:
                        n_rem = min(3, len(in_idx))
                        to_rem = in_idx[rng.choice(len(in_idx), size=n_rem, replace=False)]
                        for u in to_rem:
                            in_set[u] = False
                            conflict_count[neighbors[u]] -= 1
                        no_improve = 0
                    else:
                        break
        
        return local_best, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    while time.time() < deadline - 10 and n_restarts < 200:
        n_restarts += 1
        in_set = greedy_min_degree()
        cur_size = in_set.sum()
        if cur_size > best_size:
            best_size = cur_size
            best_in_set = in_set.copy()
        
        rem_time = deadline - 10 - time.time()
        if rem_time < 1.0: break
        ls_time = min(rem_time * 0.5, 2.5)
        ls_res, ls_sz = local_search(in_set, max_time=ls_time)
        if ls_sz > best_size:
            best_size = ls_sz
            best_in_set = ls_res.copy()
    
    return arr[best_in_set]
