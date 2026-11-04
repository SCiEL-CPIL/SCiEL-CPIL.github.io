import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    # Norm is 2.
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
    # Norm is sqrt(4) = 2.
    # Condition for two Type D vectors v_i, v_j:
    # ||v_i - v_j||^2 >= max(||v_i||, ||v_j||)^2 = 4
    # ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j >= 4
    # 4 + 4 - 2 v_i.v_j >= 4  =>  4 >= 2 v_i.v_j  =>  v_i.v_j <= 2
    # Since v_i, v_j have entries in {-1, 0, 1}, the dot product is an integer.
    # If they share k indices, the dot product is sum of products of signs at those indices.
    # Max possible dot product if they share k indices is k (if all signs match).
    # So we need k <= 2. If they share 3 or 4 indices, it's possible for the dot product to be 3 or 4, which is > 2.
    # Specifically, if they share 3 indices, can we always make dot product <= 2?
    # If we choose signs, we can control the dot product.
    # However, the "support" is the set of indices. The conflict depends on the signs too.
    # The current program groups by support first, then assigns signs.
    # Conflict between supports: If two supports intersect in k >= 3 indices, is it ALWAYS a conflict?
    # No, if we choose signs such that the dot product is small.
    # Example: Support {1,2,3,4} and {1,2,3,5}. Intersection {1,2,3}.
    # v1 = (1, 1, 1, 1, 0, ...)
    # v2 = (1, 1, -1, 0, 1, ...)
    # Dot = 1*1 + 1*1 + 1*(-1) = 1. This is <= 2. So they are compatible!
    # The original program's conflict matrix `conflict[i, j] = (intersection >= 3)` is a SUFFICIENT condition for conflict 
    # ONLY IF we assume worst-case signs or if we are building a code where signs are fixed by support?
    # No, the original program generates ALL 16 sign combinations for each selected support.
    # If two supports intersect in 3 indices, there EXIST sign combinations that are compatible (dot <= 2) and some that are not (dot = 3 or 4).
    # The original program's strategy:
    # 1. Find a set of supports such that NO two supports intersect in >= 3 indices.
    #    This guarantees that for ANY sign combination of these supports, the dot product is <= 2 (since max intersection is 2, max dot is 2).
    #    This is a safe subset of valid vectors.
    # 2. Then it tries to add more vectors from OTHER supports (or same supports with different signs?) 
    #    The "Greedy fill" loop iterates over ALL Type D vectors (all supports, all signs).
    #    It checks compatibility with the current set.
    #    This allows adding vectors from supports that might intersect in >= 3 indices, PROVIDED the specific sign combination is compatible.
    
    # So the "Support Packing" step finds a large set of mutually non-conflicting supports (intersection <= 2).
    # This ensures a large block of vectors is definitely valid.
    # Then the greedy fill tries to squeeze in more.

    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build support masks for fast intersection
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

    # Vectorized conflict matrix for supports
    # Conflict if intersection size >= 3
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict_supports = popcounts >= 3

    # Degrees for greedy heuristic
    degrees = conflict_supports.sum(axis=1)

    py_rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    best_support_set = set()
    best_size = 0

    def greedy_init_heuristic():
        x = np.zeros(N_S, dtype=bool)
        available = np.ones(N_S, dtype=bool)
        current_size = 0
        # We want to pick supports that don't conflict.
        # Greedy: pick the one with lowest degree among available, or random tie-break.
        # To speed up, we can just iterate.
        
        # Simple greedy with randomization
        order = np.arange(N_S)
        np_rng.shuffle(order)
        
        # Actually, a better heuristic is to pick the node with minimum degree in the remaining graph.
        # But updating the graph is expensive. 
        # Let's stick to a randomized greedy: shuffle, pick if not blocked.
        # Or: repeatedly pick the available node with min degree.
        
        # Let's use the min-degree heuristic for better initial packing
        avail_deg = degrees.copy()
        available_nodes = set(range(N_S))
        
        while available_nodes:
            # Find node with min degree
            min_deg = min(avail_deg[n] for n in available_nodes)
            candidates = [n for n in available_nodes if avail_deg[n] == min_deg]
            v = py_rng.choice(candidates)
            
            x[v] = True
            current_size += 1
            available_nodes.remove(v)
            
            # Remove neighbors from available and update degrees
            neighbors = np.where(conflict_supports[v])[0]
            for n in neighbors:
                if n in available_nodes:
                    available_nodes.remove(n)
            
            # Note: We don't strictly need to update avail_deg for remaining nodes for the *correctness* of the independent set,
            # but it helps for the heuristic. However, removing neighbors from the set of candidates is sufficient for validity.
            # The degree of remaining nodes decreases, but we just need to pick from the remaining valid nodes.
            
        return x, current_size

    def local_search(x, current_size):
        # 1-Swap local search on the support independent set
        # State: x (bool array of length N_S)
        # Move: Remove i, Add j.
        # Valid if j is not in x, and j does not conflict with any node in x (except i which is being removed).
        
        # Precompute conflict counts for current set
        # conflict_count[k] = number of neighbors of k that are in x
        conflict_count = np.zeros(N_S, dtype=np.int32)
        if current_size > 0:
            conflict_count = conflict_supports[x].sum(axis=0).astype(np.int32)
        
        best_x = x.copy()
        best_size_local = current_size
        no_improve = 0
        max_no_improve = 1000
        
        while no_improve < max_no_improve:
            if time.time() > deadline - 20:
                break
            
            improved = False
            
            # Try to add a node that has 0 conflicts with current set
            add_candidates = np.where((~x) & (conflict_count == 0))[0]
            if len(add_candidates) > 0:
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                conflict_count += conflict_supports[v]
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                improved = True
                continue
            
            # Try to swap: Remove i, Add j
            # j must have conflict_count[j] == 1, and the single conflict must be i.
            # And i must be in x.
            
            # This is harder to vectorize simply.
            # Let's pick a random i in x, and see if we can replace it.
            in_x = np.where(x)[0]
            if len(in_x) == 0:
                break
            
            i = py_rng.choice(in_x)
            
            # Find j not in x such that j conflicts ONLY with i in the current set.
            # conflict_count[j] == 1 and conflict_supports[j, i] is True.
            # Also j must not be i.
            
            potential_j = np.where((~x) & (conflict_count == 1))[0]
            if len(potential_j) == 0:
                no_improve += 1
                continue
                
            # Filter potential_j to those that conflict with i
            valid_j = [j for j in potential_j if conflict_supports[j, i]]
            
            if valid_j:
                j = py_rng.choice(valid_j)
                # Perform swap
                x[i] = False
                x[j] = True
                # Update conflict counts
                # Remove i's contribution
                conflict_count -= conflict_supports[i]
                # Add j's contribution
                conflict_count += conflict_supports[j]
                # Size remains same, but structure changes.
                # We don't update best_size_local because size is same.
                # But we reset no_improve? No, size didn't increase.
                # However, we moved to a new state.
                # Let's just continue.
                improved = True
            
            if not improved:
                no_improve += 1
                # Random removal to escape local optimum?
                # Remove a random node to allow re-adding different ones
                if len(in_x) > 0:
                    i = py_rng.choice(in_x)
                    x[i] = False
                    current_size -= 1
                    conflict_count -= conflict_supports[i]
                    # This decreases size, so we definitely didn't improve best.
                    # But it might allow future additions.
        
        return best_x, best_size_local

    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break

        x, size = greedy_init_heuristic()
        best_x, best_size_local = local_search(x, size)

        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())
            
            # Early stop if we hit a theoretical upper bound?
            # Upper bound for 4-subsets of 11 with intersection <= 2 is the packing number.
            # Total pairs in 11 is 55. Each 4-set has 6 pairs. 55/6 = 9.16. So max 9 supports?
            # Wait, intersection <= 2 means they can share 0, 1, or 2 elements.
            # This is a 2-(11,4,1) packing? No, it's a packing where blocks intersect in at most 2 points.
            # This is related to constant weight codes with min distance 4 (since distance = 2*(4-k), if k=2, dist=4).
            # A(11, 4, 4) = 55? No.
            # If we just count: 11 choose 4 = 330 supports.
            # The limit is much higher than 9.
            # Let's not early stop on a small bound.

    # Generate selected Type D vectors
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64) if selected_vectors_D else np.zeros((0, D), dtype=np.int64)

    # Greedy fill with ALL Type D vectors
    # This allows adding vectors from supports NOT in best_support_set,
    # or even vectors from supports IN best_support_set but with different signs?
    # No, we already added ALL 16 signs for the selected supports.
    # So we are looking for vectors from OTHER supports.
    
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    
    if len(current_set_vectors) > 0:
        current_matrix = np.array(current_set_vectors, dtype=np.int64)
    else:
        current_matrix = np.zeros((0, D), dtype=np.int64)

    indices_all = np.arange(N_all_D)
    np_rng.shuffle(indices_all)

    # To speed up, we can pre-compute the dot products of all_vectors_D with current_matrix?
    # No, current_matrix grows.
    # We can just iterate.
    
    for i in indices_all:
        if time.time() > deadline - 5:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue

        if len(current_matrix) > 0:
            dots = current_matrix @ v
            # Check if any dot product > 2
            # We also need to check against Type A vectors?
            # Type A has norm 2. Type D has norm 2.
            # Same condition: dot <= 2.
            # But Type A vectors are fixed.
            # We should check against the FULL set of vectors (Type A + Type D).
            
            # Let's combine Type A and current Type D into one matrix for checking.
            # This is expensive if we do it every iteration.
            # Instead, we can check against Type A once per v, and against Type D incrementally.
            
            # Check against Type A
            dots_A = vectors_A @ v
            if np.any(dots_A > 2):
                continue
            
            # Check against current Type D
            if np.any(dots > 2):
                continue

        current_set_vectors.append(v)
        current_set_vectors_set.add(tuple(v))
        current_matrix = np.vstack([current_matrix, v.reshape(1, -1)])

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    
    # Final validation check (optional, but good for debugging)
    # For production, skip to save time.
    
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
