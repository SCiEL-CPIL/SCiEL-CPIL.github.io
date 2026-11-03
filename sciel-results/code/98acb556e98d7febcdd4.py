import itertools
import time
import random
import heapq
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    # Also build sparse matrix for LP
    block_tsubs = [None] * n_blocks
    rows = []
    cols = []
    data = []
    
    # Precompute combinations of t from k for vectorized generation
    t_in_k = list(itertools.combinations(range(k), t))
    
    for bi, block in enumerate(all_blocks):
        # Vectorized: for each t-subset of positions, map to actual point indices
        # block is a tuple of k points
        # We need tsub_idx for each combination
        # This is still a loop in Python but we can try to optimize
        tsubs_list = []
        for pos_combo in t_in_k:
            ts = tuple(block[i] for i in pos_combo)
            tsubs_list.append(tsub_idx[ts])
        
        tsubs_arr = np.array(tsubs_list, dtype=np.int32)
        block_tsubs[bi] = tsubs_arr
        
        for ts_idx in tsubs_list:
            rows.append(ts_idx)
            cols.append(bi)
            data.append(1.0)
            
    # Sparse matrix M where M[ts_idx, bi] = 1 if block bi covers ts_idx
    M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
    
    # Precompute t-subset sets for all blocks (for use in local search)
    # Use frozenset or just keep the arrays and use numpy set operations?
    # For the swap check, we need to check if unique_tsubs (array) is subset of block_tsubs[ci] (array)
    # Using sets is O(1) lookup per element, but creating sets for 50k blocks is expensive.
    # Alternative: Use bitsets? n_tsubsets = C(19,4) = 3876. Fits in 60 64-bit ints.
    # Let's use a list of sets for now, but maybe optimize later if slow.
    block_tsub_sets = [set(block_tsubs[i]) for i in range(n_blocks)]
    
    # Precompute inverse mapping: for each t-subset index, which blocks cover it
    # M is already CSR, so M[ts_idx, :].indices gives the blocks.
    # But accessing M[ts_idx, :] in a loop is slow.
    # Let's precompute a list of arrays for faster access in local search.
    blocks_covering_ts = [M[i, :].indices.copy() for i in range(n_tsubsets)]
    
    # 2. LP Relaxation to get dual weights
    # Minimize sum(x_j)
    # Subject to M.T @ x >= 1
    # Which is -M.T @ x <= -1
    c = np.ones(n_blocks)
    A_ub = -M.T.tocsr()
    b_ub = -np.ones(n_tsubsets)
    bounds = [(0, 1)] * n_blocks
    
    dual_weights = None
    try:
        # Use a tighter time limit for LP to ensure we have time for search
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'time_limit': 10})
        if res.success:
            dual_weights = np.abs(res.ineqlin.marginals)
            dual_weights[dual_weights < 0] = 0
            if np.sum(dual_weights) == 0:
                dual_weights = np.ones(n_tsubsets)
        else:
            dual_weights = np.ones(n_tsubsets)
    except Exception:
        dual_weights = np.ones(n_tsubsets)
        
    # 3. LP-Guided Greedy Initialization (Vectorized update for speed)
    def lp_greedy(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        
        # Vectorized weight calculation
        # M.T @ dual_weights gives the initial weight for each block
        # But we need to update as uncovered changes.
        # Let's maintain a vector `active_duals` which is dual_weights * uncovered
        active_duals = dual_weights.copy()
        block_weights = M.T @ active_duals
        
        # Build max-heap
        heap = [(-block_weights[i], i) for i in range(n_blocks)]
        heapq.heapify(heap)
        
        selected_blocks = []
        selected_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            # Pop from heap until we find a valid (unselected, up-to-date) block
            valid_block = None
            while heap:
                neg_w, bi = heapq.heappop(heap)
                if bi in selected_set:
                    continue
                # Check if weight is up-to-date
                # We need to compare -neg_w with block_weights[bi]
                # But block_weights is updated via vector ops, so it should be current
                # However, the heap entry might be stale if we pushed duplicates.
                # In this implementation, we push new entries every time a weight changes.
                # So we just need to check if the heap entry matches the current block_weights[bi]
                if abs(-neg_w - block_weights[bi]) < 1e-6:
                    valid_block = (bi, block_weights[bi])
                    break
                else:
                    # Stale entry, discard
                    pass
            
            if valid_block is None:
                break
            
            bi, w = valid_block
            if w <= 1e-9:
                break
            
            selected_blocks.append(bi)
            selected_set.add(bi)
            
            # Update uncovered and weights
            ts_list = block_tsubs[bi]
            active_mask = uncovered[ts_list]
            if np.any(active_mask):
                active_ts_indices = ts_list[active_mask]
                uncovered[active_ts_indices] = False
                
                # Update active_duals
                active_duals[active_ts_indices] = 0.0
                
                # Update block_weights vectorized
                # block_weights = M.T @ active_duals
                # But we can do it incrementally:
                # The change is -dual_weights[active_ts_indices] for blocks covering those t-subsets
                # Let's just recompute the full vector? M is 3876 x 50388.
                # M.T @ active_duals is 50388 x 3876 dot 3876 x 1.
                # This is a sparse-dense matvec.
                # scipy.sparse matvec is fast.
                block_weights = M.T @ active_duals
                # Ensure non-negative (numerical errors)
                block_weights[block_weights < 0] = 0
                
                # Push updated weights for all blocks? No, that's too many.
                # We only need to push blocks whose weight changed.
                # Which blocks changed? Those covering any active_ts_indices.
                # We can find them by union of blocks_covering_ts[ts] for ts in active_ts_indices.
                # Or just push all blocks? 50k pushes is okay if infrequent.
                # But we do this ~150 times. 150 * 50k = 7.5M pushes. Might be slow.
                # Let's be smarter.
                
                # Identify affected blocks
                affected_blocks_set = set()
                for ts_idx in active_ts_indices:
                    affected_blocks_set.update(blocks_covering_ts[ts_idx])
                
                for b_idx in affected_blocks_set:
                    if b_idx not in selected_set:
                        heapq.heappush(heap, (-block_weights[b_idx], b_idx))
                    
        return selected_blocks

    def walksat_refine(block_indices, seed=0):
        """WalkSAT-style stochastic local search to minimize block count."""
        rng = random.Random(seed)
        
        # Current state: set of selected block indices
        selected = set(block_indices)
        
        # Compute coverage: for each t-subset, count how many selected blocks cover it
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in selected:
            cover_count[block_tsubs[bi]] += 1
        
        # Track best solution found
        best_selected = set(selected)
        best_count = len(selected)
        
        # Parameters
        p_remove = 0.5
        max_steps = 30000  # Reduced to allow more restarts
        no_improve_limit = 3000
        
        step = 0
        no_improve = 0
        
        while step < max_steps and time.time() < deadline - 5:
            step += 1
            
            if len(selected) < best_count:
                best_selected = set(selected)
                best_count = len(selected)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > no_improve_limit:
                    break
            
            if not selected:
                break
            
            # Convert to list for random access
            sel_list = list(selected)
            
            if rng.random() < p_remove:
                # Try to remove a random block
                bi = sel_list[rng.randint(0, len(sel_list)-1)]
                
                ts_indices = block_tsubs[bi]
                
                min_count = np.min(cover_count[ts_indices])
                if min_count > 1:
                    selected.discard(bi)
                    cover_count[ts_indices] -= 1
            else:
                # Try to swap
                bi_to_remove = sel_list[rng.randint(0, len(sel_list)-1)]
                
                ts_indices = block_tsubs[bi_to_remove]
                unique_mask = cover_count[ts_indices] == 1
                unique_tsubs = ts_indices[unique_mask]
                
                if len(unique_tsubs) == 0:
                    selected.discard(bi_to_remove)
                    cover_count[ts_indices] -= 1
                    continue
                
                # Try to find a replacement
                n_samples = 20
                found_swap = False
                
                for _ in range(n_samples):
                    ci = rng.randint(0, n_blocks - 1)
                    if ci in selected:
                        continue
                    
                    cand_set = block_tsub_sets[ci]
                    if all(ts in cand_set for ts in unique_tsubs):
                        selected.discard(bi_to_remove)
                        cover_count[ts_indices] -= 1
                        
                        ci_ts_indices = block_tsubs[ci]
                        selected.add(ci)
                        cover_count[ci_ts_indices] += 1
                        
                        found_swap = True
                        break
                
                if not found_swap:
                    if rng.random() < 0.05:
                        ci = rng.randint(0, n_blocks - 1)
                        if ci not in selected:
                            selected.add(ci)
                            cover_count[block_tsubs[ci]] += 1
        
        # Final polish: exhaustive 1-for-1 swap
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for bi in list(selected):
                ts_indices = block_tsubs[bi]
                unique_mask = cover_count[ts_indices] == 1
                unique_tsubs = ts_indices[unique_mask]
                
                if len(unique_tsubs) == 0:
                    selected.discard(bi)
                    cover_count[ts_indices] -= 1
                    improved = True
                    if len(selected) < best_count:
                        best_selected = set(selected)
                        best_count = len(selected)
                    continue
                
                # Try to find a replacement
                for ci in range(n_blocks):
                    if ci in selected:
                        continue
                    cand_set = block_tsub_sets[ci]
                    if all(ts in cand_set for ts in unique_tsubs):
                        selected.discard(bi)
                        cover_count[ts_indices] -= 1
                        ci_ts_indices = block_tsubs[ci]
                        selected.add(ci)
                        cover_count[ci_ts_indices] += 1
                        improved = True
                        if len(selected) < best_count:
                            best_selected = set(selected)
                            best_count = len(selected)
                        break
        
        return [all_blocks[bi] for bi in best_selected]

    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts
    # We have ~100s left after LP.
    # lp_greedy takes ~1-2s. walksat takes ~5-10s.
    # Let's try 10 restarts.
    num_restarts = 10
    for seed in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        init_blocks = lp_greedy(seed)
        if not init_blocks:
            continue
        
        refined = walksat_refine(init_blocks, seed)
        
        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = refined
            # print(f"Restart {seed}: {best_len} blocks")
            
    if best_blocks is None:
        # Fallback: simple greedy with heap
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        gain = np.array([len(block_tsubs[i]) for i in range(n_blocks)], dtype=np.int32)
        heap = [(-gain[i], i) for i in range(n_blocks)]
        heapq.heapify(heap)
        selected_set = set()
        
        while np.any(uncovered):
            valid_block = None
            while heap:
                neg_g, bi = heapq.heappop(heap)
                if bi in selected_set:
                    continue
                ts_list = block_tsubs[bi]
                active = ts_list[uncovered[ts_list]]
                current_g = len(active)
                if abs(-neg_g - current_g) < 1:
                    valid_block = (bi, current_g)
                    break
                else:
                    heapq.heappush(heap, (-current_g, bi))
            
            if valid_block is None or valid_block[1] == 0:
                break
            
            bi, g = valid_block
            blocks_used.append(bi)
            selected_set.add(bi)
            
            ts_list = block_tsubs[bi]
            active_mask = uncovered[ts_list]
            if np.any(active_mask):
                active_ts = ts_list[active_mask]
                uncovered[active_ts] = False
                
                for ts_idx in active_ts:
                    block_indices = M[ts_idx, :].indices
                    for b_idx in block_indices:
                        if b_idx not in selected_set:
                            gain[b_idx] -= 1
                            gain[b_idx] = max(gain[b_idx], 0)
                            heapq.heappush(heap, (-gain[b_idx], b_idx))
        
        best_blocks = [all_blocks[bi] for bi in blocks_used]

    return best_blocks
