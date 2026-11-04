import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    # These have norm^2 = 4.
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
    # These have norm^2 = 4.
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build conflict graph: conflict if intersection size >= 3
    # If two supports intersect in 3 or more indices, any two vectors with those supports
    # will have dot product at least 1 (if signs align) or -1 (if signs oppose) or similar.
    # Wait, let's check the distance condition.
    # v1, v2 in R^11.
    # dist^2 = ||v1 - v2||^2 = ||v1||^2 + ||v2||^2 - 2 v1.v2
    # We want dist^2 >= max_norm^2.
    # All vectors have norm^2 = 4 (Type A: 4, Type D: 1+1+1+1=4).
    # So max_norm^2 = 4.
    # Condition: 4 + 4 - 2 v1.v2 >= 4  =>  4 >= 2 v1.v2  =>  v1.v2 <= 2.
    
    # Let's verify compatibility between Type A and Type D.
    # Type A: 2 e_i. Type D: signs on 4 indices.
    # If i is in support of D: v1.v2 = 2 * (+/- 1) = +/- 2. Max is 2. OK.
    # If i not in support of D: v1.v2 = 0. OK.
    # So Type A and Type D are always compatible.
    
    # Now Type D vs Type D.
    # v1 has support S1, v2 has support S2.
    # v1.v2 = sum_{i in S1 \cap S2} v1_i v2_i.
    # Each term is +/- 1.
    # Let k = |S1 \cap S2|.
    # Max possible dot product is k (if all signs align).
    # We need v1.v2 <= 2.
    # So if k <= 2, max dot is 2. Always OK.
    # If k = 3, max dot is 3. Bad if signs align.
    # If k = 4, max dot is 4. Bad if signs align.
    
    # So, for supports with intersection size 3 or 4, we must choose signs carefully.
    # Actually, the current program's conflict logic says:
    # "conflict if intersection size >= 3"
    # And it removes all pairs of supports that intersect in >= 3.
    # This is a safe but conservative approach: it assumes we can't control signs to avoid conflict?
    # No, the current program generates ALL sign combinations for the selected supports.
    # Wait, if it selects a set of supports that are pairwise disjoint (intersection < 3),
    # then ANY sign assignment is valid.
    # But if it allows supports with intersection 3, it must ensure that for that pair,
    # the specific sign assignments chosen have dot product <= 2.
    
    # The current program finds an independent set of supports where intersection < 3.
    # This guarantees that for any pair of selected supports, intersection is 0, 1, or 2.
    # Thus, max dot product is 2.
    # So ANY sign assignment for these supports is valid.
    # This allows 16 vectors per support.
    
    # Is it better to allow supports with intersection 3?
    # If we allow intersection 3, we can only choose signs such that dot <= 2.
    # For intersection 3, there are 8 sign patterns for the 3 shared indices.
    # Dot product is sum of 3 terms.
    # Values: 3, 1, -1, -3.
    # We need <= 2. So we must avoid the pattern where all 3 signs are the same (dot=3).
    # So for each pair of supports with intersection 3, we have a constraint on signs.
    # This becomes a constraint satisfaction problem or a weighted independent set on a larger graph.
    
    # The current approach (disjoint-ish supports) is simpler and likely near-optimal.
    # Let's stick to the MIS on supports with intersection < 3.
    
    # The key is to maximize the number of supports selected.
    # Max number of 4-subsets of 11 with pairwise intersection < 3.
    # This is equivalent to a packing design or constant weight code.
    # A(11, 4, 6)? No, intersection < 3 means distance > 2*4 - 2*2 = 4?
    # Hamming distance between supports: d = |S1 \ S2| + |S2 \ S1| = 4 + 4 - 2|S1 \cap S2|.
    # If |S1 \cap S2| <= 2, then d >= 4 + 4 - 4 = 4.
    # So we want a binary code of length 11, weight 4, min distance 4.
    # This is A(11, 4, 4).
    # Let's check known values.
    # A(11, 4, 4) is related to 4-uniform hypergraphs with no 3-uniform sub-hypergraph?
    # Actually, just max number of edges in a 4-uniform hypergraph on 11 vertices with no two edges sharing 3 vertices.
    
    # Let's run the SA to find the max size.
    # The current program does this.
    
    # Improvements:
    # 1. Better initialization for SA.
    # 2. More restarts or steps if time permits.
    # 3. Vectorized operations where possible.
    
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute popcount for 11-bit integers
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3
    
    # Remove self-conflicts
    np.fill_diagonal(conflict, False)

    # Degree of each node
    degrees = np.sum(conflict, axis=1)

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_size = 0
    best_support_set = set()

    # Simulated Annealing for Maximum Independent Set
    def simulate_annealing(num_restarts=200, steps_per_restart=10000):
        best_local_size = 0
        best_local_set = set()
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Greedy init with random noise
            in_set = np.zeros(N_S, dtype=bool)
            blocked = np.zeros(N_S, dtype=bool)
            
            noise = rng.uniform(0, 5, size=N_S)
            scores = degrees + noise
            init_order = np.argsort(scores)
            
            for v in init_order:
                if not blocked[v]:
                    in_set[v] = True
                    blocked |= conflict[v]
            
            current_size = np.sum(in_set)
            current_set = set(np.where(in_set)[0].tolist())
            
            if current_size > best_local_size:
                best_local_size = current_size
                best_local_set = current_set.copy()
            
            # SA Phase
            T_start = 1.0
            T_end = 0.001
            alpha = (T_end / T_start) ** (1.0 / steps_per_restart)
            T = T_start
            
            for step in range(steps_per_restart):
                if time.time() > deadline - 5:
                    break
                
                T *= alpha
                
                move_type = py_rng.random()
                
                if move_type < 0.5:
                    # Try Add
                    not_in_set = np.where(~in_set)[0]
                    if len(not_in_set) == 0:
                        continue
                    v = not_in_set[py_rng.randint(0, len(not_in_set)-1)]
                    
                    if np.any(conflict[v] & in_set):
                        continue
                    
                    in_set[v] = True
                    current_size += 1
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local_set = set(np.where(in_set)[0].tolist())
                        
                elif move_type < 0.8:
                    # Try Remove
                    if current_size == 0:
                        continue
                    in_set_idx = np.where(in_set)[0]
                    v = in_set_idx[py_rng.randint(0, len(in_set_idx)-1)]
                    
                    if py_rng.random() < np.exp(-1.0 / T):
                        in_set[v] = False
                        current_size -= 1
                        
                else:
                    # Try Swap
                    if current_size == 0:
                        continue
                    in_set_idx = np.where(in_set)[0]
                    v_rem = in_set_idx[py_rng.randint(0, len(in_set_idx)-1)]
                    
                    not_in_set = np.where(~in_set)[0]
                    if len(not_in_set) == 0:
                        continue
                    v_add = not_in_set[py_rng.randint(0, len(not_in_set)-1)]
                    
                    temp_in_set = in_set.copy()
                    temp_in_set[v_rem] = False
                    
                    if np.any(conflict[v_add] & temp_in_set):
                        continue
                    
                    if py_rng.random() < 0.9:
                        in_set[v_rem] = False
                        in_set[v_add] = True

        return best_local_size, best_local_set

    # Run SA
    best_size, best_support_set = simulate_annealing(100, 20000)
    
    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    
    if selected_vectors_D:
        selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
        final_vectors = np.vstack([vectors_A, selected_vectors_D])
    else:
        final_vectors = vectors_A

    return final_vectors
