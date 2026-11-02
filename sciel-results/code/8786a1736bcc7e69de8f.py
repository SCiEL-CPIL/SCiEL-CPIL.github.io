import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute subsets_of_block: for each block j, list of t-subset indices
    subsets_of_block = [[] for _ in range(n_k)]
    # Precompute blocks_of_subset: for each t-subset i, list of block indices
    blocks_of_subset = [[] for _ in range(n_t)]
    
    # Build sparse matrix for LP
    rows = []
    cols = []
    data = []
    
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            
    best_solution = None
    best_size = float('inf')
    
    # 2. Get LP Duals for weighting
    duals = np.ones(n_t)
    try:
        if time.time() < deadline - 10:
            M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
            c = np.ones(n_k)
            A_ub = -M.T.tocsr()
            b_ub = -np.ones(n_t)
            bounds = [(0, 1)] * n_k
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success and res.ineqlin is not None and res.ineqlin.marginals is not None:
                duals = np.abs(res.ineqlin.marginals)
                if np.sum(duals) == 0:
                    duals = np.ones(n_t)
    except Exception:
        duals = np.ones(n_t)

    # 3. Greedy Construction with Restarts
    # We will try a few different weightings.
    # Base weight for a block is sum of duals of subsets it covers.
    # We add noise to diversify.
    
    num_restarts = 8 # Adjust based on time
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Create weights
        block_weights = np.zeros(n_k)
        for j in range(n_k):
            if subsets_of_block[j]:
                block_weights[j] = np.sum(duals[subsets_of_block[j]])
        
        # Add noise for restarts > 0
        if restart > 0:
            # Small noise to change ordering slightly
            noise = np.random.uniform(0, 0.05 * np.max(block_weights), n_k)
            block_weights += noise
        
        # Heap of (-weight, index)
        heap = [(-block_weights[j], j) for j in range(n_k)]
        heapq.heapify(heap)
        
        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        solution_blocks = []
        
        while heap:
            if time.time() > deadline - 10:
                break
                
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
                
            # Lazy deletion check: if weight in heap is higher than current, skip
            # Note: weights only decrease, so if w > current, it's stale.
            # Actually, we push new weights when they decrease.
            # The heap contains old high weights. We need to check if this entry is still valid.
            # A simpler check: if the block is already picked, skip.
            # We don't strictly need to check weight validity if we just skip picked blocks,
            # but it helps to skip blocks that have become useless (weight 0).
            
            if w <= 1e-9:
                break
                
            # Select block j
            picked[j] = True
            solution_blocks.append(j)
            
            # Update weights of other blocks
            for s_idx in subsets_of_block[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    y_s = duals[s_idx]
                    if y_s > 0:
                        for b_prime in blocks_of_subset[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                heapq.heappush(heap, (-block_weights[b_prime], b_prime))
                                
            if np.all(covered):
                break
                
        if len(solution_blocks) < best_size and np.all(covered):
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()
            
            # If we found a very good solution (e.g., close to known best 151), 
            # we might want to spend more time polishing it.
            # But let's continue with a few more restarts to see if we can beat it.
            
    # 4. Local Search: Deletion and Swap
    if best_solution is not None:
        # Run local search on the best solution found
        local_best = best_solution.copy()
        local_best_size = best_size
        
        # Deletion Pass
        def perform_deletion(sol):
            if not sol:
                return sol, 0
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in sol:
                for s_idx in subsets_of_block[b_idx]:
                    cover_count[s_idx] += 1
                    
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                new_sol = []
                for b_idx in sol:
                    subs_b = subsets_of_block[b_idx]
                    can_remove = True
                    for s_idx in subs_b:
                        if cover_count[s_idx] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        cover_count[subs_b] -= 1
                        changed = True
                    else:
                        new_sol.append(b_idx)
                sol = new_sol
            return sol, len(sol)

        local_best, local_best_size = perform_deletion(local_best)

        # Swap Pass (1-exchange)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Recalculate cover counts
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in local_best:
                for s_idx in subsets_of_block[b_idx]:
                    cover_count[s_idx] += 1
            
            for i, b_idx in enumerate(local_best):
                if time.time() > deadline - 5:
                    break
                    
                # Find unique subsets covered by b_idx
                unique_subs = []
                for s_idx in subsets_of_block[b_idx]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                    
                if not unique_subs:
                    # Redundant block, remove it
                    for s_idx in subsets_of_block[b_idx]:
                        cover_count[s_idx] -= 1
                    local_best.pop(i)
                    local_best_size -= 1
                    improved = True
                    break # Restart loop
                    
                # Check if we can swap
                # Union of points in unique subsets
                union_points = set()
                for s_idx in unique_subs:
                    union_points.update(t_subsets[s_idx])
                    
                if len(union_points) > k:
                    continue # No single block can cover these points
                    
                # Find candidate blocks that cover all unique subsets
                # This is equivalent to covering the union of points? 
                # No, it must cover the specific 4-subsets.
                # If B' covers all points in union_points, it covers all subsets of union_points.
                # Since unique_subs are subsets of b_idx, and we want B' to cover them,
                # it suffices that B' contains the union of points in unique_subs.
                
                # Find blocks containing union_points
                # We can iterate through blocks_of_subset of one of the unique subsets
                # and check if it contains all points in union_points.
                
                candidate_blocks = set(blocks_of_subset[unique_subs[0]])
                candidate_blocks.discard(b_idx)
                
                for s_idx in unique_subs[1:]:
                    candidate_blocks.intersection_update(blocks_of_subset[s_idx])
                    if not candidate_blocks:
                        break
                        
                if not candidate_blocks:
                    continue
                    
                # Among candidates, pick one that maximizes "value" 
                # (e.g., covers many other currently uncovered or weakly covered subsets)
                # For now, just pick the first one or the one with highest weight?
                # Let's pick the one that minimizes the number of new "unique" subsets it creates?
                # Or simply pick the first valid one.
                
                # To be more robust, we can try a few random candidates if there are many.
                cand_list = list(candidate_blocks)
                if len(cand_list) > 1:
                    random.shuffle(cand_list)
                    # Try up to 5 candidates
                    for cand_idx in cand_list[:5]:
                        # Check if swapping improves or maintains size
                        # Since we replace 1 with 1, size is same.
                        # We just need to ensure validity.
                        # The swap is valid if cand_idx covers all unique_subs.
                        # We already checked this via intersection.
                        
                        # Perform swap
                        local_best[i] = cand_idx
                        improved = True
                        break # Restart loop to recalculate cover counts
                        
                if improved:
                    break

        if local_best_size < best_size:
            best_size = local_best_size
            best_solution = local_best

    # Final Deletion Pass
    if best_solution is not None and time.time() < deadline - 2:
        # Quick deletion check
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                can_remove = True
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)
                    
            best_solution = new_solution
            best_size = len(best_solution)

    return [list(k_subsets[idx]) for idx in best_solution]
