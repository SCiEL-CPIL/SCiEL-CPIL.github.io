import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Type A: 22 vectors (±2e_i)
    type_A = []
    for i in range(D):
        v = [0]*D
        v[i] = 2
        type_A.append(tuple(v))
        v2 = [0]*D
        v2[i] = -2
        type_A.append(tuple(v2))
    
    # Type M: 220 vectors (two ±2 entries, norm²=8)
    type_M = []
    for i in range(D):
        for j in range(i+1, D):
            for si in [2, -2]:
                for sj in [2, -2]:
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    type_M.append(tuple(v))
    
    # Type D: Use support-based approach, limit to manageable size
    # Use only supports with 4 coordinates, but limit total count
    supports = list(itertools.combinations(range(D), 4))
    
    # For each support, include all 16 sign combinations
    # But this gives 330*16=5280 which is too many
    # Instead, use the conflict-based selection from the original
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    N_S = len(supports)
    
    # Build conflict between supports (same as original: conflict if |intersection| >= 3)
    support_conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i+1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                support_conflict[i, j] = True
                support_conflict[j, i] = True
    
    # Select a good set of supports using greedy
    selected_supports = set()
    blocked = set()
    remaining = set(range(N_S))
    while remaining:
        candidates = [v for v in remaining if v not in blocked]
        if not candidates:
            break
        # Pick support with fewest conflicts
        best_v = min(candidates, key=lambda v: sum(1 for u in candidates if support_conflict[v, u]))
        selected_supports.add(best_v)
        for u in range(N_S):
            if support_conflict[best_v, u]:
                blocked.add(u)
        remaining -= (blocked | selected_supports)
    
    # Build Type D vectors from selected supports
    type_D = []
    for si in selected_supports:
        support = supports[si]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0]*D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            type_D.append(tuple(v))
    
    # All vectors
    all_vectors = type_A + type_M + type_D
    n_vec = len(all_vectors)
    
    vectors_arr = np.array(all_vectors, dtype=np.int32)
    norms_sq = np.array([sum(x*x for x in v) for v in all_vectors], dtype=np.int32)
    max_norm_sq = float(norms_sq.max())
    
    # Build conflict matrix
    dots = vectors_arr @ vectors_arr.T
    dist_sq = norms_sq[:, None].astype(np.float64) + norms_sq[None, :].astype(np.float64) - 2.0 * dots
    np.fill_diagonal(dist_sq, np.inf)
    conflict = (dist_sq < max_norm_sq).astype(np.int8)
    
    adj = [set(np.where(conflict[i])[0]) for i in range(n_vec)]
    
    # Greedy MIS
    def greedy_mis(order=None):
        selected = set()
        remaining = set(range(n_vec))
        if order is None:
            order = list(range(n_vec))
        for v in order:
            if v not in remaining:
                continue
            if not (adj[v] & remaining):
                selected.add(v)
                remaining -= (adj[v] | {v})
        return selected
    
    # Initial greedy
    best_set = greedy_mis()
    best_size = len(best_set)
    
    # Local search
    def local_search(current_set, max_iters=3000):
        current = set(current_set)
        curr_size = len(current)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            # Try adding
            for v in range(n_vec):
                if v in current:
                    continue
                if not (adj[v] & current):
                    current.add(v)
                    curr_size += 1
                    improved = True
                    break
            if improved:
                continue
            # Try 1-for-2 swap
            for u in list(current):
                temp = current - {u}
                added = 0
                for v in range(n_vec):
                    if v in temp:
                        continue
                    if not (adj[v] & temp):
                        temp.add(v)
                        added += 1
                        if added >= 2:
                            break
                if added > 1:
                    current = temp
                    curr_size = len(current)
                    improved = True
                    break
        return current
    
    best_set = local_search(best_set)
    best_size = len(best_set)
    
    # Multi-restart with different orderings
    rng = np.random.RandomState(42)
    for restart in range(30):
        if time.time() > deadline - 15.0:
            break
        order = rng.permutation(n_vec).tolist()
        greedy_sol = greedy_mis(order)
        improved = local_search(greedy_sol, max_iters=1000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Final polish
    changed = True
    while changed and time.time() < deadline - 3.0:
        changed = False
        for v in range(n_vec):
            if v in best_set:
                continue
            if not (adj[v] & best_set):
                best_set.add(v)
                changed = True
                break
    
    selected_indices = sorted(best_set)
    result = np.array([all_vectors[i] for i in selected_indices], dtype=np.int8)
    
    return result
