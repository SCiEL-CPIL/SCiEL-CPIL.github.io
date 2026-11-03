import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import heapq
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = [None] * n_blocks
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Build sparse matrix for LP
    rows = []
    cols = []
    data = []
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
            rows.append(ts_idx)
            cols.append(bi)
            data.append(1.0)
            
    M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
    
    # Precompute set versions of tsub_blocks for fast intersection in local search
    # Note: This uses memory but saves time in swap checks.
    tsub_block_sets = [set(lst) for lst in tsub_blocks]
    
    # 2. LP Relaxation to get dual weights
    c = np.ones(n_blocks)
    A_ub = -M.T.tocsr()
    b_ub = -np.ones(n_tsubsets)
    bounds = [(0, 1)] * n_blocks
    
    dual_weights = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # HiGHS marginals for A_ub @ x <= b_ub
            # The dual of min c^T x s.t. M^T x >= 1 is max 1^T y s.t. M y <= 1, y >= 0.
            # In linprog with <= constraints, marginals are typically negative of the duals for >= constraints?
            # Let's just take absolute value and ensure non-negative.
            dual_weights = np.abs(res.ineqlin.marginals)
            dual_weights[dual_weights < 0] = 0
            if np.sum(dual_weights) == 0:
                dual_weights = np.ones(n_tsubsets)
    except Exception:
        dual_weights = np.ones(n_tsubsets)
        
    # 3. LP-Guided Greedy with Heap
    def lp_greedy_heap(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        
        # Initialize weights: sum of dual weights of t-subsets in block
        # We use a max-heap via negative values
        # heap entries: (-weight, block_index, weight_version)
        # We need to track if the weight in the heap is still valid.
        # Simpler: Just push new entries. When popping, check if the weight matches the current calculated weight?
        # Or maintain a 'current_weight' array and a 'version' array?
        # Let's use a simple approach: 
        # current_weight[bi] is the true weight.
        # heap contains (-weight, bi).
        # When we pop (-w, bi), if -w == current_weight[bi] and current_weight[bi] > 0, it's valid.
        # Otherwise, it's stale.
        
        current_weight = np.zeros(n_blocks, dtype=np.float64)
        for bi in range(n_blocks):
            current_weight[bi] = np.sum(dual_weights[block_tsubs[bi]])
            
        heap = []
        for bi in range(n_blocks):
            heapq.heappush(heap, (-current_weight[bi], bi))
            
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            valid_bi = -1
            valid_w = 0.0
            
            while heap:
                neg_w, bi = heapq.heappop(heap)
                w = -neg_w
                # Check if this entry is stale
                # The weight could have changed. 
                # If w is close to current_weight[bi] and current_weight[bi] > 0, it's valid.
                # Due to floating point, we might need tolerance, but weights are sums of doubles.
                # Actually, if we subtracted weights, current_weight[bi] might be slightly different from w if we didn't push the exact same float?
                # We push the exact current_weight value. So they should match.
                
                if abs(w - current_weight[bi]) < 1e-9 and w > 1e-9:
                    valid_bi = bi
                    valid_w = w
                    break
                # If w is 0 or negative, it's definitely not a candidate for max positive weight
                # If w != current_weight[bi], it's stale.
            
            if valid_bi == -1:
                # No valid block with positive weight?
                # This might happen if all weights became 0 but uncovered is not empty?
                # This implies a bug or LP weights were zero for some uncovered sets.
                # Fallback: pick any block covering an uncovered t-subset.
                # Find an uncovered t-subset
                unc_idx = np.where(uncovered)[0][0]
                # Pick a random block covering it
                candidates = tsub_blocks[unc_idx]
                if not candidates:
                    break
                valid_bi = candidates[rng.randint(0, len(candidates)-1)]
                valid_w = 1.0 # Just to proceed
            
            # Select block
            selected_blocks.append(valid_bi)
            
            # Update uncovered and weights
            ts_list = block_tsubs[valid_bi]
            # Get indices of t-subsets in this block that are currently uncovered
            active_mask = uncovered[ts_list]
            if np.any(active_mask):
                active_ts_indices = ts_list[active_mask]
                
                # Mark them as covered
                uncovered[active_ts_indices] = False
                
                # Update weights for all blocks covering these t-subsets
                for ts_idx in active_ts_indices:
                    w_sub = dual_weights[ts_idx]
                    if w_sub == 0:
                        continue
                    # Get blocks covering this t-subset
                    block_indices = tsub_blocks[ts_idx]
                    for bi in block_indices:
                        current_weight[bi] -= w_sub
                        # Clamp to 0
                        if current_weight[bi] < 0:
                            current_weight[bi] = 0
                        # Push updated weight to heap
                        heapq.heappush(heap, (-current_weight[bi], bi))
                        
            # Invalidate the selected block's weight to prevent re-selection
            current_weight[valid_bi] = 0
            heapq.heappush(heap, (-current_weight[valid_bi], valid_bi))
            
        return selected_blocks

    def local_search_1_for_1(block_indices, seed=0):
        """
        Local search using 1-for-1 swaps based on unique coverage.
        """
        rng = random.Random(seed)
        selected = set(block_indices)
        
        # Compute cover counts
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in selected:
            cover_count[block_tsubs[bi]] += 1
            
        best_selected = set(selected)
        best_count = len(selected)
        
        improved = True
        max_passes = 50
        passes = 0
        
        while improved and passes < max_passes and time.time() < deadline - 2:
            improved = False
            passes += 1
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in list(selected):
                # Check if any t-subset in bi has cover_count == 1
                # If not, it's redundant.
                ts_indices = block_tsubs[bi]
                if np.min(cover_count[ts_indices]) > 1:
                    to_remove.append(bi)
                    
            if to_remove:
                for bi in to_remove:
                    selected.discard(bi)
                    cover_count[block_tsubs[bi]] -= 1
                improved = True
                if len(selected) < best_count:
                    best_selected = set(selected)
                    best_count = len(selected)
                continue # Restart loop
            
            # Phase 2: Try 1-for-1 swaps
            # Iterate through a shuffled list of blocks
            current_blocks = list(selected)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in selected:
                    continue
                
                ts_indices = block_tsubs[bi]
                # Find unique t-subsets for bi
                unique_mask = cover_count[ts_indices] == 1
                unique_tsubs = ts_indices[unique_mask]
                
                if len(unique_tsubs) == 0:
                    # Redundant, should have been caught in Phase 1, but safe to remove
                    selected.discard(bi)
                    cover_count[ts_indices] -= 1
                    improved = True
                    break # Restart loop
                
                # Find candidate blocks that cover all unique_tsubs
                # Intersect tsub_block_sets for each unique t-subset
                # Start with the one with fewest candidates to speed up
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_block_sets[x]))
                candidates = set(tsub_block_sets[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]
                    
                # Remove current block from candidates
                candidates.discard(bi)
                
                if not candidates:
                    continue
                
                # Try candidates
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                # Limit candidates to try to avoid long stalls
                for bj in cand_list[:20]:
                    if bj in selected:
                        continue
                    
                    # Perform swap
                    # Remove bi
                    selected.discard(bi)
                    cover_count[ts_indices] -= 1
                    
                    # Add bj
                    selected.add(bj)
                    cover_count[block_tsubs[bj]] += 1
                    
                    improved = True
                    swapped = True
                    break # Restart loop after successful swap
                
                if swapped:
                    if len(selected) < best_count:
                        best_selected = set(selected)
                        best_count = len(selected)
                    break # Restart while loop
            
        return list(best_selected)

    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts
    num_restarts = 15
    for seed in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Get initial solution from LP-Greedy
        init_blocks = lp_greedy_heap(seed)
        if not init_blocks:
            continue
        
        # Refine with Local Search
        refined = local_search_1_for_1(init_blocks, seed)
        
        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = refined
            
        # Early termination if we hit a very good score (optional)
        if best_len <= 140:
            # We can stop early or continue to see if we can go lower
            # Given time constraints, stopping early might be safer to ensure valid return
            pass

    if best_blocks is None:
        # Fallback: simple greedy without LP (should not happen if LP works)
        uncovered = set(range(n_tsubsets))
        blocks_used = []
        gain = [len(block_tsubs[i]) for i in range(n_blocks)]
        while uncovered:
             max_gain = max(gain)
             if max_gain == 0: break
             candidates = [i for i in range(n_blocks) if gain[i] == max_gain]
             chosen = candidates[0]
             blocks_used.append(chosen)
             for ts_idx in block_tsubs[chosen]:
                 if ts_idx in uncovered:
                     uncovered.discard(ts_idx)
                     for bi in tsub_blocks[ts_idx]:
                         if gain[bi] > 0:
                             gain[bi] -= 1
             gain[chosen] = 0
        best_blocks = [all_blocks[bi] for bi in blocks_used]

    return best_blocks
