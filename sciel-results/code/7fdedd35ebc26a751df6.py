"""Kissing number in dimension 11 - Improved independent set search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate candidates: norm squared 4
    # Type A: weight 4, entries in {-1, 1}
    # Type B: weight 1, entries in {-2, 2}
    
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute conflict matrix
    # Conflict if dot > 2
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to list of lists for faster iteration
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def greedy_init():
        """Greedy min-degree initialization."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search(in_set, max_time=5.0):
        """Local search with 1-add, 1-1 swap, and 1-2 swap."""
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # conflict_count[i]: number of neighbors of i that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        ls_deadline = time.time() + max_time
        stuck = 0
        
        while time.time() < ls_deadline:
            improved = False
            
            # 1. Try to add a vertex with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = conflict_count[not_in_set]
                zero_conf = not_in_set[counts == 0]
                if len(zero_conf) > 0:
                    v = zero_conf[rng.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                        stuck = 0
                    continue
            
            # 2. Try 1-1 Swap
            if not improved and stuck > 10:
                cands_v = np.where((~in_set) & (conflict_count == 1))[0]
                if len(cands_v) > 0:
                    v = cands_v[rng.integers(len(cands_v))]
                    nbrs_v_in_set = np.where(conflicts[v] & in_set)[0]
                    if len(nbrs_v_in_set) == 1:
                        u = nbrs_v_in_set[0]
                        in_set[u] = False
                        in_set[v] = True
                        conflict_count[neighbors[u]] -= 1
                        conflict_count[neighbors[v]] += 1
                        stuck = 0
                        improved = True
                        continue

            # 3. Try 1-2 Swap (Remove 1, Add 2)
            if not improved and stuck > 50:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 0:
                    u = in_set_indices[rng.integers(len(in_set_indices))]
                    cands_v = np.where((~in_set) & (conflict_count == 1))[0]
                    if len(cands_v) >= 2:
                        idx = rng.choice(len(cands_v), size=2, replace=False)
                        v1 = cands_v[idx[0]]
                        v2 = cands_v[idx[1]]
                        
                        nbrs_v1 = np.where(conflicts[v1] & in_set)[0]
                        nbrs_v2 = np.where(conflicts[v2] & in_set)[0]
                        
                        if len(nbrs_v1) == 1 and nbrs_v1[0] == u and \
                           len(nbrs_v2) == 1 and nbrs_v2[0] == u and \
                           not conflicts[v1, v2]:
                           
                           in_set[u] = False
                           in_set[v1] = True
                           in_set[v2] = True
                           
                           conflict_count[neighbors[u]] -= 1
                           conflict_count[neighbors[v1]] += 1
                           conflict_count[neighbors[v2]] += 1
                           
                           current_size += 1
                           improved = True
                           stuck = 0
                           if current_size > best_size:
                               best_size = current_size
                               best_in_set = in_set.copy()
                           continue

            if not improved:
                stuck += 1
                if stuck > 100:
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        u = in_set_indices[rng.integers(len(in_set_indices))]
                        in_set[u] = False
                        current_size -= 1
                        conflict_count[neighbors[u]] -= 1
                        stuck = 0

        return best_in_set, best_size

    def polish_pass(in_set, max_time=2.0):
        """Final polish: add all zero-conflict vertices, then try swaps."""
        in_set = in_set.copy()
        current_size = in_set.sum()
        
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        polish_deadline = time.time() + max_time
        
        # Add all zero-conflict vertices
        not_in_set = np.where(~in_set)[0]
        if len(not_in_set) > 0:
            zero_conf = not_in_set[conflict_count[not_in_set] == 0]
            for v in zero_conf:
                in_set[v] = True
                current_size += 1
                conflict_count[neighbors[v]] += 1
        
        # Try some swaps
        stuck = 0
        while time.time() < polish_deadline and stuck < 200:
            improved = False
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                zero_conf = not_in_set[conflict_count[not_in_set] == 0]
                if len(zero_conf) > 0:
                    v = zero_conf[rng.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    stuck = 0
                    continue
            
            cands_v = np.where((~in_set) & (conflict_count == 1))[0]
            if len(cands_v) > 0:
                v = cands_v[rng.integers(len(cands_v))]
                nbrs_v_in_set = np.where(conflicts[v] & in_set)[0]
                if len(nbrs_v_in_set) == 1:
                    u = nbrs_v_in_set[0]
                    in_set[u] = False
                    in_set[v] = True
                    conflict_count[neighbors[u]] -= 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    stuck = 0
                    continue
            
            if not improved:
                stuck += 1
                if stuck > 100:
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        u = in_set_indices[rng.integers(len(in_set_indices))]
                        in_set[u] = False
                        current_size -= 1
                        conflict_count[neighbors[u]] -= 1
                        stuck = 0

        return in_set, current_size

    # Run multiple restarts
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    # Initial Greedy
    in_set = greedy_init()
    size = in_set.sum()
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
        
    # Local Search on initial
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 2:
        ls_time = min(remaining_time * 0.5, 10.0)
        ls_result, ls_size = local_search(best_in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
        
        # Polish pass
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 1:
            polish_result, polish_size = polish_pass(best_in_set, max_time=min(remaining_time * 0.5, 2.0))
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
            
    # Restarts
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 30:
        n_restarts += 1
        in_set = greedy_init()
        size = in_set.sum()
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
            
        ls_time = min(remaining_time * 0.3, 4.0)
        ls_result, ls_size = local_search(in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
        
        # Polish pass
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 1:
            polish_result, polish_size = polish_pass(best_in_set, max_time=min(remaining_time * 0.3, 1.5))
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()

    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors

# EVOLVE-BLOCK-END
