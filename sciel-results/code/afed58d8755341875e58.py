import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build conflict graph: conflict if intersection size >= 3
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute popcount for 11-bit integers (0 to 2047)
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix computation
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3

    # Degree of each node in the conflict graph
    degrees = np.sum(conflict, axis=1)

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Greedy initialization with different orderings
    def greedy_init(order):
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x, np.sum(x)

    # MRV-based greedy: at each step, pick the unblocked node with minimum "remaining values"
    # i.e., minimum number of unblocked neighbors it would block (or maximum compatible unselected)
    def mrv_greedy_init(seed_order):
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        unblocked = np.arange(N_S)
        size = 0
        # We'll maintain a list of currently unblocked nodes
        # At each step, among unblocked nodes, pick one that minimizes the number
        # of unblocked nodes it would block (i.e., minimizes future conflicts)
        # This is the "minimum remaining values" heuristic
        
        unblocked_list = list(unblocked)
        while unblocked_list:
            # For each candidate in unblocked_list, compute how many unblocked nodes it conflicts with
            best_idx_in_list = 0
            best_score = float('inf')
            for i, v in enumerate(unblocked_list):
                # Number of unblocked nodes that v conflicts with
                # conflict[v] is a boolean array; we want count of True in conflict[v] restricted to unblocked_list
                # This is expensive to compute fully, so approximate: use precomputed degree within unblocked set
                # For efficiency, we'll use a sampled approach or just use degree as proxy
                # Actually, let's compute it properly but efficiently using the conflict matrix
                # Count unblocked neighbors of v
                # We can precompute this but it changes as we remove nodes
                # For now, use the static degree as a proxy for MRV
                score = degrees[v]  # proxy: lower degree = fewer conflicts = MRV
                if score < best_score:
                    best_score = score
                    best_idx_in_list = i
            
            v = unblocked_list.pop(best_idx_in_list)
            x[v] = True
            size += 1
            # Remove all nodes that conflict with v from unblocked_list
            conflict_with_v = np.where(conflict[v])[0]
            # Remove from unblocked_list
            new_unblocked = []
            conflict_set = set(conflict_with_v.tolist())
            for u in unblocked_list:
                if u not in conflict_set:
                    new_unblocked.append(u)
            unblocked_list = new_unblocked
        
        return x, size

    # Fast local search: only do swaps and adds, vectorized blocked computation
    def local_search_fast(x, current_size, num_iterations):
        best_x = x.copy()
        best_size_local = current_size
        x = x.copy()
        
        for it in range(num_iterations):
            if time.time() > deadline - 15:
                break
            
            # Compute blocked nodes
            if current_size > 0:
                blocked = conflict[x].any(axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            add_candidates = np.where(~x & ~blocked)[0]
            
            if len(add_candidates) > 0:
                # Try to add a random non-conflicting node
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
            else:
                # No add possible, try a swap: remove one, then try to add
                selected = np.where(x)[0]
                if len(selected) == 0:
                    break
                
                v_rem = py_rng.choice(selected)
                x[v_rem] = False
                current_size -= 1
                
                # Recompute blocked after removal
                if current_size > 0:
                    blocked = conflict[x].any(axis=0)
                else:
                    blocked = np.zeros(N_S, dtype=bool)
                
                add_candidates = np.where(~x & ~blocked)[0]
                if len(add_candidates) > 0:
                    v_add = py_rng.choice(add_candidates)
                    x[v_add] = True
                    current_size += 1
                    if current_size > best_size_local:
                        best_size_local = current_size
                        best_x = x.copy()
                else:
                    # Couldn't add, revert removal
                    x[v_rem] = True
                    current_size += 1
        
        return best_x, best_size_local

    # Multiple strategies for initialization
    strategies = []
    
    # Strategy 1: degree-ordered greedy (ascending)
    order_deg = np.argsort(degrees)
    strategies.append(order_deg)
    
    # Strategy 2: reverse degree-ordered
    strategies.append(order_deg[::-1])
    
    # Strategy 3-12: random orderings
    for _ in range(10):
        order = np.arange(N_S)
        rng.shuffle(order)
        strategies.append(order)

    # Run MRV-based greedy first
    x_mrv, size_mrv = mrv_greedy_init(None)
    best_x_mrv, best_size_mrv = local_search_fast(x_mrv, size_mrv, 5000)
    if best_size_mrv > best_size:
        best_size = best_size_mrv
        best_support_set = set(np.where(best_x_mrv)[0].tolist())

    num_restarts = 150
    iterations_per_restart = 5000

    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        # Use different initialization strategies
        if restart < len(strategies):
            order = strategies[restart]
        else:
            order = np.arange(N_S)
            rng.shuffle(order)
        
        x, size = greedy_init(order)
        best_x, best_size_local = local_search_fast(x, size, iterations_per_restart)
        
        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    # Greedy fill: try to add more Type D vectors from non-selected supports
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))

    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 10:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue
        can_add = True
        for u in current_set_vectors:
            dot = np.dot(v, u)
            if dot >= 3:
                can_add = False
                break
        if can_add:
            current_set_vectors.append(v)
            current_set_vectors_set.add(tuple(v))

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
