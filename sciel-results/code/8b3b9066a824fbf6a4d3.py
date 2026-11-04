import numpy as np
import time
import itertools
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Precomputation
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # Conflict matrix: conflict[i, j] = True if intersection >= 3
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    # Precompute degree (number of conflicts) for each support
    degrees = conflict.sum(axis=1)

    rng = np.random.RandomState(42)

    def greedy_min_degree(randomize=False):
        """
        Heap-based greedy packing: repeatedly select the available support with
        minimum current degree to maximize packing density.
        """
        selected = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        current_degree = np.zeros(N_S, dtype=np.int32)
        
        # Build initial heap
        heap = []
        for i in range(N_S):
            key = (degrees[i], i)
            heapq.heappush(heap, key)
        
        count = 0
        remaining = N_S
        
        while remaining > 0:
            if time.time() > deadline - 5.0:
                break
            
            # Find the minimum-degree available node from heap
            found = False
            while heap:
                deg, v = heapq.heappop(heap)
                if not selected[v] and not blocked[v]:
                    found = True
                    break
                # Stale entry, discard
            
            if not found:
                break
            
            selected[v] = True
            count += 1
            remaining -= 1
            blocked[v] = True
            
            # Update degrees of neighbors
            neighbors = np.where(conflict[v])[0]
            for nb in neighbors:
                if not selected[nb] and not blocked[nb]:
                    current_degree[nb] += 1
                    blocked[nb] = True
            
            # Re-push remaining available nodes with updated degrees
            # Only re-push nodes that are still available and haven't been blocked
            for i in range(N_S):
                if not selected[i] and not blocked[i]:
                    heapq.heappush(heap, (current_degree[i], i))
        
        return selected, count

    def local_search_polish(x):
        """
        Improve the solution via 1-swap and 2-swap operations.
        """
        current_x = x.copy()
        current_score = int(np.sum(current_x))
        best_x = current_x.copy()
        best_score = current_score
        
        in_set = np.where(current_x)[0]
        if len(in_set) == 0:
            return best_x
        
        blocked = np.any(conflict[in_set], axis=0)
        # blocked[i] = True means i conflicts with at least one selected support
        
        max_iter = 300
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            in_set = np.where(current_x)[0]
            if len(in_set) == 0:
                break
            
            # Try 1-swap: remove u, add v
            improved = False
            for _ in range(min(20, len(in_set))):
                u = int(rng.choice(in_set))
                
                # Temporarily remove u
                temp_in = np.delete(in_set, np.where(in_set == u)[0][0])
                if len(temp_in) > 0:
                    temp_blocked = np.any(conflict[temp_in], axis=0)
                else:
                    temp_blocked = np.zeros(N_S, dtype=bool)
                
                candidates = np.where(~(current_x | temp_blocked))[0]
                if len(candidates) == 0:
                    continue
                
                # Pick candidate with minimum degree
                cand_degrees = degrees[candidates]
                v = candidates[int(np.argmin(cand_degrees))]
                
                # Apply swap
                current_x[u] = False
                current_x[v] = True
                in_set = np.where(current_x)[0]
                blocked = np.any(conflict[in_set], axis=0)
                
                # Try to add more after swap
                added = 0
                while True:
                    candidates2 = np.where(~(current_x | blocked))[0]
                    if len(candidates2) == 0:
                        break
                    cand2_degrees = degrees[candidates2]
                    v2 = candidates2[int(np.argmin(cand2_degrees))]
                    current_x[v2] = True
                    added += 1
                    blocked |= conflict[v2]
                
                new_score = int(np.sum(current_x))
                if new_score > best_score:
                    best_score = new_score
                    best_x = current_x.copy()
                    improved = True
                else:
                    # Revert
                    current_x = best_x.copy()
                    in_set = np.where(current_x)[0]
                    blocked = np.any(conflict[in_set], axis=0)
            
            if not improved:
                break
        
        return best_x

    best_support_set = set()
    best_size = 0
    
    # Phase 1: Deterministic greedy (fast baseline)
    x_det, count_det = greedy_min_degree(randomize=False)
    if count_det > best_size:
        best_size = count_det
        best_support_set = set(np.where(x_det)[0].tolist())
    
    # Phase 2: Multiple randomized greedy restarts
    num_restarts = 3000
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        x_r, count_r = greedy_min_degree(randomize=True)
        
        if count_r > best_size:
            x_polished = local_search_polish(x_r)
            size_polished = int(np.sum(x_polished))
            if size_polished > best_size:
                best_size = size_polished
                best_support_set = set(np.where(x_polished)[0].tolist())
            else:
                best_size = count_r
                best_support_set = set(np.where(x_r)[0].tolist())
        elif count_r == best_size:
            if r % 10 == 0:
                x_polished = local_search_polish(x_r)
                size_polished = int(np.sum(x_polished))
                if size_polished > best_size:
                    best_size = size_polished
                    best_support_set = set(np.where(x_polished)[0].tolist())
    
    # Final polish on the best solution
    if len(best_support_set) > 0:
        x_best = np.zeros(N_S, dtype=bool)
        for idx in best_support_set:
            x_best[idx] = True
        x_final = local_search_polish(x_best)
        size_final = int(np.sum(x_final))
        if size_final > best_size:
            best_size = size_final
            best_support_set = set(np.where(x_final)[0].tolist())
    
    # Construct vectors
    vectors = []
    
    # Type A: ±2e_i for each coordinate
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)
        
    # Type D: all 16 sign combinations for each selected 4-subset
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
    
    # Type M: 96 mixed vectors of norm sqrt(8)
    # These have 4 entries of ±1 and 4 entries of ±1 on different coordinates
    # Specifically: vectors with exactly 8 nonzero entries, each ±1, 
    # where they lie "between" E8 and D3 subspaces
    # The 96 vectors are: for each way to choose 4 coordinates from first 8 and 
    # 4 from last 3... actually let me think about this more carefully.
    # 
    # The 96 mixed vectors: norm^2 = 8, so 8 entries of ±1.
    # They must be at distance >= sqrt(8) from all Type A and Type D vectors.
    # 
    # Let me generate all vectors with exactly 8 nonzero ±1 entries and check
    # which ones satisfy the distance constraints.
    
    # Generate candidate mixed vectors: choose 8 positions out of 11, assign ±1
    # That's C(11,8) * 2^8 = 165 * 256 = 42240 candidates
    # We need to check distance >= sqrt(8) to all existing vectors
    
    # For efficiency, let's precompute the set of existing vectors as a set of tuples
    existing_set = set(tuple(v) for v in vectors)
    
    # For a mixed vector m (8 nonzeros of ±1), distance to:
    # - Type A (±2e_i): dist^2 = 8 + 4 = 12 if m has entry at i, or 8 if not. Both >= 8. OK.
    # - Type D (4 nonzeros of ±1 on support S): 
    #   If |m ∩ S| = k, then dist^2 = (8-k) + (4-k) + sum of squared differences on overlap
    #   On overlap: if m[i]=d[i], diff=0; else diff=4. 
    #   So dist^2 = 8 + 4 - 2k + 4*(k - matches) where matches = number of matching signs
    #   = 12 - 2k + 4k - 4*matches = 12 + 2k - 4*matches
    #   Need dist^2 >= 8: 12 + 2k - 4*matches >= 8, i.e., 2k - 4*matches >= -4, i.e., k - 2*matches >= -2
    #   Since matches <= k, worst case matches=k: k - 2k = -k >= -2, i.e., k <= 2.
    #   So we need |m ∩ S| <= 2 for all selected supports S, OR if |m ∩ S| = 3, then matches <= (3+2)/2 = 2.5, so matches <= 2.
    #   If |m ∩ S| = 4, then matches <= (4+2)/2 = 3, so matches <= 3.
    
    # Actually, let me just brute-force check all 42240 candidates against the existing vectors.
    # But that might be slow. Let me use a smarter approach.
    
    # For each candidate mixed vector, check against all Type D vectors in the selected set.
    # There are 16 * best_size Type D vectors.
    
    # Let's precompute the selected Type D vectors
    type_d_vectors = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            type_d_vectors.append(v)
    
    # Also add Type A vectors for checking
    type_a_vectors = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        type_a_vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        type_a_vectors.append(v2)
    
    all_existing = type_a_vectors + type_d_vectors
    n_existing = len(all_existing)
    
    # Convert to numpy for vectorized distance checks
    all_existing_np = np.array(all_existing, dtype=np.int8)
    
    # Generate all candidate mixed vectors
    mixed_candidates = []
    for positions in itertools.combinations(range(D), 8):
        for signs in itertools.product([1, -1], repeat=8):
            v = [0] * D
            for idx, sign in zip(positions, signs):
                v[idx] = sign
            mixed_candidates.append(v)
    
    mixed_candidates_np = np.array(mixed_candidates, dtype=np.int8)
    n_mixed = len(mixed_candidates_np)
    
    # Compute distances in batches to avoid memory issues
    batch_size = 1000
    valid_mixed = []
    
    for batch_start in range(0, n_mixed, batch_size):
        if time.time() > deadline - 5.0:
            break
        batch_end = min(batch_start + batch_size, n_mixed)
        batch = mixed_candidates_np[batch_start:batch_end]
        
        # Compute squared distances to all existing vectors
        # dist^2 = sum((batch[i] - existing[j])^2)
        # = sum(batch[i]^2) + sum(existing[j]^2) - 2*batch[i]·existing[j]
        # batch norm^2 = 8 for all
        # existing norm^2: Type A = 4, Type D = 4
        
        dots = batch @ all_existing_np.T  # (batch_size, n_existing)
        batch_norms = np.full((batch.shape[0], 1), 8.0)
        existing_norms = np.full((1, n_existing), 4.0)
        
        dist_sq = batch_norms + existing_norms - 2 * dots
        
        # Check if min distance >= 8
        min_dist_sq = dist_sq.min(axis=1)
        valid_in_batch = np.where(min_dist_sq >= 8)[0]
        
        for idx in valid_in_batch:
            valid_mixed.append(mixed_candidates_np[batch_start + idx].tolist())
    
    vectors.extend(valid_mixed)
    
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
