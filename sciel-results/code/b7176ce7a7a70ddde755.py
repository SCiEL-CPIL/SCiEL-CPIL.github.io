import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    # Norm = 2. Distance to any other Type A is at least 2 (if same index, diff 4 or 0; if diff index, diff 2 or 4).
    # Wait, if v1 = e_i * 2 and v2 = -e_i * 2, dist = 4.
    # If v1 = e_i * 2 and v2 = e_j * 2 (i!=j), dist^2 = 4+4=8, dist=sqrt(8) > 2.
    # So Type A vectors are always compatible with each other (min dist >= 2).
    # Compatibility with Type D (norm 2):
    # Type D has 4 non-zeros of +/- 1.
    # If Type D has 0 at index i, dist^2 = 4 + 4 = 8 (if Type A is e_i*2). dist > 2.
    # If Type D has +/- 1 at index i, dist^2 = (2 - 1)^2 + 3*1^2 = 1 + 3 = 4 (if signs match) or (2 - (-1))^2 + 3 = 9+3=12.
    # So if Type D has non-zero at i, min dist is 2. This is valid (>= 2).
    # So ALL Type A vectors are compatible with ALL Type D vectors?
    # Let's check: min pairwise distance >= max norm.
    # Max norm is 2.
    # Dist between two Type A: min 2 (e.g. e1*2 and e2*2 -> sqrt(8)? No. e1*2 and e1*2 is 0. But they are distinct vectors?
    # The problem says "nonzero integer vectors". Usually distinct vectors.
    # If we include e1*2 and -e1*2, dist is 4.
    # If we include e1*2 and e2*2, dist is sqrt(8) ~ 2.82.
    # So Type A set is internally compatible.
    
    # Type D: Vectors with four non-zero entries +/- 1.
    # Norm = sqrt(4) = 2.
    # Two Type D vectors v, w.
    # dist^2 = ||v-w||^2 = ||v||^2 + ||w||^2 - 2 v.w = 4 + 4 - 2 v.w = 8 - 2 v.w.
    # We need dist >= 2 => dist^2 >= 4 => 8 - 2 v.w >= 4 => 2 v.w <= 4 => v.w <= 2.
    # v.w is the dot product.
    # v and w have supports of size 4.
    # Let I = support(v) \cap support(w).
    # v.w = sum_{i in I} v_i w_i.
    # v_i, w_i are +/- 1. So v.w is an integer between -|I| and |I|.
    # Condition: v.w <= 2.
    # This fails if v.w >= 3.
    # v.w >= 3 implies |I| >= 3.
    # If |I| = 4 (same support), v.w is in {-4, -2, 0, 2, 4}.
    #   If v=w, v.w=4 (dist 0). But we pick distinct sign configs?
    #   If v != w but same support, v.w is -4, -2, 0, 2. Max is 2. So dist^2 >= 4.
    #   So all 16 sign configs on the same support are mutually compatible?
    #   Wait. If v.w = 2, dist^2 = 4. Dist = 2. OK.
    #   If v.w = 4, dist = 0. Not OK. But v=w is not allowed (distinct vectors).
    #   So for a fixed support, we can include ALL 16 sign vectors?
    #   Let's check: v1 = (1,1,1,1), v2 = (1,1,1,-1). dot = 3. dist^2 = 8 - 6 = 2 < 4. FAIL.
    #   Ah, v.w = 3 is possible if |I|=4 and 3 signs match.
    #   So on the same support, we cannot include all 16.
    #   We need v.w <= 2.
    #   This is a constraint on the set of sign vectors for a given support.
    
    # However, the current program and inspirations assume a simpler model:
    # "Conflict if intersection size is 3" for supports.
    # If supports intersect in 3 indices, can we have compatible vectors?
    # Let supp(v) = A, supp(w) = B. |A|=|B|=4. |A \cap B| = 3.
    # v.w = sum_{i in A \cap B} v_i w_i.
    # |A \cap B| = 3.
    # Max v.w = 3 (if all signs match).
    # If v.w = 3, dist^2 = 8 - 6 = 2 < 4. FAIL.
    # If v.w = 2, dist^2 = 8 - 4 = 4. OK.
    # So if |A \cap B| = 3, we CAN have compatible vectors, PROVIDED their signs on the intersection are not all matching (i.e., dot product <= 2).
    # The inspiration 1/2 approach simplifies this by saying: if |A \cap B| >= 3, then supports conflict, so we cannot select both supports.
    # This is a conservative approximation. It finds a set of supports such that any pair intersects in <= 2.
    # If supports intersect in <= 2, then |A \cap B| <= 2.
    # Then max v.w = 2.
    # Then min dist^2 = 8 - 4 = 4. Dist = 2.
    # So ANY sign vectors on non-conflicting supports are mutually compatible.
    # And for a single support, we need to select a subset of the 16 sign vectors such that pairwise dot product <= 2.
    
    # Let's analyze the "same support" constraint.
    # We need a set S of sign vectors in {+1, -1}^4 such that for any u, v in S, u.v <= 2.
    # This is equivalent to finding a maximum subset of the 16 vertices of the 4-cube with Hamming distance >= 2?
    # u.v = 4 - 2 * Hamming(u, v).
    # u.v <= 2 => 4 - 2 d_H <= 2 => 2 d_H >= 2 => d_H >= 1.
    # Wait.
    # If u = v, d_H = 0, u.v = 4. Dist = 0.
    # If u != v, d_H >= 1.
    # If d_H = 1, u.v = 2. Dist^2 = 4. Dist = 2. OK.
    # If d_H = 2, u.v = 0. Dist^2 = 8. OK.
    # If d_H = 3, u.v = -2. OK.
    # If d_H = 4, u.v = -4. OK.
    # So the condition u.v <= 2 is satisfied for ANY distinct u, v in {+1, -1}^4.
    # Therefore, for a single support, we can include ALL 16 sign vectors!
    
    # So the problem reduces to:
    # 1. Select a set of supports S such that any two supports in S intersect in at most 2 indices.
    #    (This ensures cross-support compatibility for any sign choices).
    # 2. For each selected support, include all 16 sign vectors.
    # 3. Include all Type A vectors (22 of them).
    # 4. Check compatibility between Type A and Type D.
    #    We established earlier that Type A and Type D are always compatible (min dist 2).
    
    # So the total count is 22 + 16 * |S|.
    # The current program finds |S| using a greedy/local search on the conflict graph (intersection >= 3).
    # The score 2.18 corresponds to some |S|.
    # Score = value / seed. Seed is likely 100 (since 22 + 16*5 = 102 ~ 1.0? No).
    # Let's assume seed is 100.
    # If score is 2.18, value is 218.
    # 22 + 16 * |S| = 218 => 16 * |S| = 196 => |S| = 12.25.
    # Maybe seed is different.
    # Inspiration 1 score 2.54. Value ~ 254.
    # 22 + 16 * |S| = 254 => 16 * |S| = 232 => |S| = 14.5.
    
    # The task asks to combine strategies.
    # Strategy:
    # 1. Use the "Conflict Graph" of supports (intersection >= 3).
    # 2. Find Maximum Independent Set (MIS) in this graph.
    # 3. Use Tomita's branch-and-bound or a very strong local search / simulated annealing to find a large MIS.
    # 4. Add Type A vectors.
    # 5. Return all vectors.

    # The inspirations use local search / simulated annealing.
    # I will implement a robust Simulated Annealing for the MIS problem on the conflict graph.
    # The graph has 330 nodes.
    # Conflict if intersection >= 3.
    
    # Precomputation
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # Conflict matrix
    # conflict[i, j] = True if intersection >= 3
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    # We want to maximize the number of selected supports.
    # State: x (boolean vector of length N_S).
    # Valid state: x is an independent set. i.e., for all i, j with x[i]=1 and x[j]=1, conflict[i,j] is False.
    
    # Simulated Annealing
    rng = np.random.RandomState(42)
    
    def random_init():
        # Random greedy
        x = np.zeros(N_S, dtype=bool)
        order = rng.permutation(N_S)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not x[v] and not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x

    def local_search(x, max_iter=1000):
        # 1-move: add a node, remove a node, or swap.
        # To ensure validity, we only consider moves that maintain independence.
        # Adding a node v: v must not be in x, and v must not conflict with any node in x.
        # Removing a node v: v must be in x.
        # Swapping: remove u, add v. v must not conflict with x \ {u}.
        
        current_x = x.copy()
        current_score = np.sum(current_x)
        best_x = current_x.copy()
        best_score = current_score
        
        # Precompute neighbors for fast access?
        # adj_list is useful.
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Generate a random move
            r = rng.random()
            if r < 0.3:
                # Try to add a node
                # Find candidates: not in x, not blocked by x
                in_set = np.where(current_x)[0]
                if len(in_set) > 0:
                    blocked = np.any(conflict[in_set], axis=0)
                else:
                    blocked = np.zeros(N_S, dtype=bool)
                
                candidates = np.where(~(current_x | blocked))[0]
                if len(candidates) > 0:
                    v = int(rng.choice(candidates))
                    new_x = current_x.copy()
                    new_x[v] = True
                    new_score = current_score + 1
                    # Acceptance
                    if new_score > current_score:
                        current_x = new_x
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_x = new_x.copy()
                    # Always accept improvement.
                    
            elif r < 0.6:
                # Try to remove a node (to escape local optima)
                in_set = np.where(current_x)[0]
                if len(in_set) > 0:
                    v = int(rng.choice(in_set))
                    new_x = current_x.copy()
                    new_x[v] = False
                    new_score = current_score - 1
                    # Metropolis acceptance for worse moves
                    # T starts high, cools down
                    T = 1.0 / (1 + _ / 100.0) # Simple cooling
                    if new_score > current_score or rng.random() < np.exp((new_score - current_score) / T):
                        current_x = new_x
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_x = new_x.copy()
                            
            else:
                # Swap: remove u, add v
                in_set = np.where(current_x)[0]
                if len(in_set) == 0:
                    continue
                u = int(rng.choice(in_set))
                
                # Candidates for v: not in x, not blocked by x \ {u}
                # x \ {u}
                temp_x = current_x.copy()
                temp_x[u] = False
                in_set_no_u = np.where(temp_x)[0]
                
                if len(in_set_no_u) > 0:
                    blocked_no_u = np.any(conflict[in_set_no_u], axis=0)
                else:
                    blocked_no_u = np.zeros(N_S, dtype=bool)
                
                # v must not be u (already removed) and not in x (and u is in x, so v != u is covered if we check ~x)
                # Also v must not be blocked by x \ {u}
                candidates = np.where(~(current_x | blocked_no_u))[0]
                # Note: current_x has u=True. So ~current_x has u=False.
                # So u is not a candidate.
                if len(candidates) > 0:
                    v = int(rng.choice(candidates))
                    new_x = current_x.copy()
                    new_x[u] = False
                    new_x[v] = True
                    new_score = current_score # Size unchanged
                    # Acceptance for neutral moves
                    T = 1.0 / (1 + _ / 100.0)
                    if rng.random() < np.exp((new_score - current_score) / T):
                        current_x = new_x
                        # current_score unchanged
                        if new_score > best_score:
                            best_score = new_score
                            best_x = new_x.copy()
                            
        return best_x

    best_support_set = set()
    best_size = 0
    
    # Run multiple restarts
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        x0 = random_init()
        x_best = local_search(x0, max_iter=500)
        size = np.sum(x_best)
        
        if size > best_size:
            best_size = int(size)
            best_support_set = set(np.where(x_best)[0].tolist())
            
    # Construct vectors
    vectors = []
    
    # Type A
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)
        
    # Type D
    for s_idx in best_support_set:
        support = supports[s_idx]
        # All 16 sign vectors
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
