"""
Combines LP-relaxation guided greedy initialization with intensive local search.
Uses scipy.optimize.linprog for the LP relaxation to obtain dual weights.
"""
import itertools
import time
import random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    t0 = time.time()
    
    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_tsubs = [None] * n_blocks
    rows = []
    cols = []
    data = []
    
    # C(k, t) is the number of t-subsets in each block
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)
        
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = np.array(tsubs, dtype=np.int32)
        for ts_idx in tsubs:
            rows.append(ts_idx)
            cols.append(bi)
            data.append(1.0)
            
    # Build sparse matrix M for LP: M[ts_idx, bi] = 1 if block bi covers ts_idx
    M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
    
    # Build inverse index: for each t-subset, list of blocks covering it
    # This is crucial for fast updates in the greedy step
    tsub_blocks = [None] * n_tsubsets
    for i in range(n_tsubsets):
        # M is CSR, so M[i, :] gives row i
        tsub_blocks[i] = M[i, :].indices.astype(np.int32)
        
    # 2. LP Relaxation to get dual weights
    dual_weights = None
    try:
        # Minimize sum(x_j)
        # Subject to M.T @ x >= 1
        # Convert to standard form for linprog: -M.T @ x <= -1
        c = np.ones(n_blocks)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_tsubsets)
        bounds = [(0, 1)] * n_blocks
        
        # Use highs solver for speed
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # Dual variables are marginals of the inequality constraints
            # In linprog, marginals for A_ub @ x <= b_ub are typically negative or zero
            # for a minimization problem where relaxing b_ub (making it larger) 
            # doesn't change the objective if the constraint is not tight.
            # However, the dual of min c^T x s.t. A x >= b is max b^T y s.t. A^T y <= c, y >= 0.
            # Here we have -M.T x <= -1.
            # Let y be the dual of the original >= constraints.
            # The marginal returned by HiGHS for A_ub @ x <= b_ub corresponds to 
            # the change in objective if b_ub is increased.
            # If we increase b_ub (make it less negative, e.g. -0.9), the constraint 
            # -M.T x <= -0.9 is looser than -M.T x <= -1.
            # So the marginal should be >= 0 if the constraint is tight?
            # Actually, standard LP duality: 
            # Primal: min c^T x s.t. A x >= b, x >= 0.
            # Dual: max b^T y s.t. A^T y <= c, y >= 0.
            # linprog formulation: min c^T x s.t. A_ub x <= b_ub, bounds.
            # A_ub = -A, b_ub = -b.
            # The dual variable y_i in linprog corresponds to the constraint A_ub_i x <= b_ub_i.
            # The relationship is y_linprog = -y_dual (if we define y_dual >= 0 for >= constraints).
            # Let's just take the absolute value and ensure non-negativity.
            dual_weights = np.abs(res.ineqlin.marginals)
            if np.sum(dual_weights) == 0:
                dual_weights = np.ones(n_tsubsets)
            else:
                # Normalize to sum to 1? Not strictly necessary, but helps with scaling.
                # Keep raw values as they represent relative difficulty.
                pass
    except Exception:
        dual_weights = np.ones(n_tsubsets)
        
    # 3. LP-Guided Greedy
    def lp_greedy(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        
        # Initial weights for each block: sum of dual_weights of t-subsets it covers
        # We can compute this efficiently using the sparse matrix structure
        # block_weights[j] = sum(dual_weights[block_tsubs[j]])
        block_weights = np.zeros(n_blocks, dtype=np.float64)
        for bi in range(n_blocks):
            block_weights[bi] = np.sum(dual_weights[block_tsubs[bi]])
            
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            # Find block with max weight
            # To add randomness for tie-breaking, we can look at top few
            # But for speed, just argmax is often good enough with different seeds for local search
            # However, pure greedy with LP weights is deterministic. 
            # To get diversity, we can add small noise or pick randomly among top N.
            
            # Let's pick randomly among the top 10% or top 10 blocks if close in weight
            # For simplicity and speed, let's just pick argmax. 
            # The diversity comes from running local search on the result with different seeds.
            # Actually, pure LP greedy is very stable. 
            # Let's introduce a small random perturbation to weights initially? 
            # No, that changes the LP solution.
            # Let's pick randomly among blocks whose weight is within 1% of the max.
            
            max_w = block_weights.max()
            if max_w <= 1e-9:
                break
                
            # Find candidates
            # threshold = max_w * 0.99
            # candidates = np.where(block_weights >= threshold)[0]
            # chosen = candidates[rng.randint(0, len(candidates)-1)]
            
            # For speed, just argmax
            chosen = np.argmax(block_weights)
            
            selected_blocks.append(chosen)
            
            # Update weights
            # For each t-subset in the chosen block that is currently uncovered:
            #   mark as covered
            #   subtract dual_weights[ts] from block_weights of all blocks containing ts
            
            ts_list = block_tsubs[chosen]
            active_mask = uncovered[ts_list]
            
            if np.any(active_mask):
                active_ts_indices = ts_list[active_mask]
                
                # Mark covered
                uncovered[active_ts_indices] = False
                
                # Update block weights
                # We need to iterate over active_ts_indices and update their blocks
                for ts_idx in active_ts_indices:
                    w_sub = dual_weights[ts_idx]
                    if w_sub == 0:
                        continue
                    # Get blocks containing this t-subset
                    blocks_to_update = tsub_blocks[ts_idx]
                    block_weights[blocks_to_update] -= w_sub
                    
                    # Ensure non-negative (floating point errors)
                    # block_weights[blocks_to_update] = np.maximum(block_weights[blocks_to_update], 0)
                    
            # Set weight of chosen block to -inf to prevent re-selection
            block_weights[chosen] = -1e10
            
        return selected_blocks

    def standard_greedy(seed=0):
        """Standard greedy for comparison/fallback."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        
        # gain[i] = number of uncovered t-subsets in block i
        gain = np.zeros(n_blocks, dtype=np.int32)
        # Initialize gain
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi]) # All are uncovered initially
            
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
                
            max_gain = gain.max()
            if max_gain == 0:
                break
                
            max_blocks = np.where(gain == max_gain)[0]
            chosen = max_blocks[rng.randint(0, len(max_blocks) - 1)]
            
            selected_blocks.append(chosen)
            
            # Update gain
            ts_list = block_tsubs[chosen]
            active_mask = uncovered[ts_list]
            
            if np.any(active_mask):
                active_ts_indices = ts_list[active_mask]
                uncovered[active_ts_indices] = False
                
                # Decrement gain for blocks covering these t-subsets
                for ts_idx in active_ts_indices:
                    blocks_to_update = tsub_blocks[ts_idx]
                    gain[blocks_to_update] -= 1
                    
            gain[chosen] = -1
            
        return selected_blocks

    def remove_redundant(blocks):
        """Remove all redundant blocks."""
        if not blocks:
            return blocks
            
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsub_lists = []
        for b in blocks:
            tsubs = block_tsubs[b].tolist()
            block_tsub_lists.append(tsubs)
            for idx in tsubs:
                cover_count[idx] += 1
                
        kept = [True] * len(blocks)
        for i in range(len(blocks)):
            redundant = all(cover_count[idx] > 1 for idx in block_tsub_lists[i])
            if redundant:
                kept[i] = False
                for idx in block_tsub_lists[i]:
                    cover_count[idx] -= 1
                    
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not kept[i]:
                    continue
                if all(cover_count[idx] > 1 for idx in block_tsub_lists[i]):
                    kept[i] = False
                    for idx in block_tsub_lists[i]:
                        cover_count[idx] -= 1
                    changed = True
                    
        return [blocks[i] for i in range(len(blocks)) if kept[i]]

    def local_search(blocks, seed=0, max_iterations=200):
        """Intensive local search: iterative removal + targeted addition."""
        if not blocks:
            return blocks
            
        rng = random.Random(seed)
        reduced = list(blocks)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2:
                break
                
            # Phase 1: Remove redundant blocks
            reduced = remove_redundant(reduced)
            
            # Phase 2: Try to swap blocks to reduce count
            if len(reduced) <= 1:
                break
                
            # Build cover counts
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            block_tsub_lists = []
            for b in reduced:
                tsubs = block_tsubs[b].tolist()
                block_tsub_lists.append(tsubs)
                for idx in tsubs:
                    cover_count[idx] += 1
                    
            improved = False
            n = len(reduced)
            order = list(range(n))
            rng.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 2:
                    break
                if len(reduced) <= 1:
                    break
                    
                # Find t-subsets uniquely covered by block i
                unique_tsubs = [idx for idx in block_tsub_lists[i] if cover_count[idx] == 1]
                if not unique_tsubs:
                    # Block is redundant, remove it
                    for idx in block_tsub_lists[i]:
                        cover_count[idx] -= 1
                    block_tsub_lists.pop(i)
                    reduced.pop(i)
                    improved = True
                    continue
                    
                # Try to find a replacement block that covers all unique_tsubs
                candidate_set = set()
                for ts_idx in unique_tsubs:
                    for bi in tsub_blocks[ts_idx]:
                        # Check if bi is not already in reduced
                        # This check is O(n) if done naively. 
                        # For speed, we can skip this check initially and verify later,
                        # or use a set of current block indices.
                        if bi not in reduced:
                            candidate_set.add(bi)
                            
                if not candidate_set:
                    continue
                    
                candidates = list(candidate_set)
                rng.shuffle(candidates)
                
                found_swap = False
                for ci in candidates[:50]:  # Limit candidates tried
                    if time.time() > deadline - 2:
                        break
                        
                    cand_tsubs_set = set(block_tsubs[ci].tolist())
                    
                    # Check if candidate covers all unique t-subsets of block i
                    if not all(ts in cand_tsubs_set for ts in unique_tsubs):
                        continue
                        
                    # Check: after removing block i and adding ci, no t-subset is uncovered
                    ok = True
                    for idx in block_tsub_lists[i]:
                        if cover_count[idx] == 1 and idx not in cand_tsubs_set:
                            ok = False
                            break
                            
                    if ok:
                        # Perform swap
                        for idx in block_tsub_lists[i]:
                            cover_count[idx] -= 1
                        for ts_idx in block_tsubs[ci]:
                            cover_count[ts_idx] += 1
                        reduced[i] = ci
                        block_tsub_lists[i] = block_tsubs[ci].tolist()
                        found_swap = True
                        improved = True
                        break
                        
                if found_swap:
                    continue
                    
            if not improved:
                # Try adding a block that covers many currently singly-covered t-subsets
                if len(reduced) > 2 and time.time() < deadline - 5:
                    single_covered = np.where(cover_count == 1)[0]
                    if len(single_covered) > 0:
                        # Find block that covers the most single-covered t-subsets
                        best_gain = -1
                        best_block = -1
                        for bi in range(n_blocks):
                            if bi in reduced:
                                continue
                            cnt = 0
                            for ts_idx in block_tsubs[bi]:
                                if cover_count[ts_idx] == 1:
                                    cnt += 1
                            if cnt > best_gain:
                                best_gain = cnt
                                best_block = bi
                                
                        if best_gain >= 2 and best_block >= 0:
                            # Add this block, then try to remove a redundant one
                            new_reduced = reduced + [best_block]
                            new_reduced = remove_redundant(new_reduced)
                            if len(new_reduced) < len(reduced):
                                reduced = new_reduced
                                improved = True
                                
            if not improved:
                break
                
        return reduced

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple trials
    # First, try LP-guided greedy
    num_lp_trials = 30
    num_std_trials = 30
    
    for seed in range(num_lp_trials):
        if time.time() > deadline - 10:
            break
            
        blocks_idx = lp_greedy(seed)
        if not blocks_idx:
            continue
            
        # Local search
        blocks_idx = local_search(blocks_idx, seed, max_iterations=100)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks = blocks_idx
            
    for seed in range(num_std_trials):
        if time.time() > deadline - 10:
            break
            
        blocks_idx = standard_greedy(seed)
        if not blocks_idx:
            continue
            
        # Local search
        blocks_idx = local_search(blocks_idx, seed + 100, max_iterations=100)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks = blocks_idx

    # Final intensive local search on best result
    if best_blocks is not None and time.time() < deadline - 15:
        final = local_search(best_blocks, 9999, max_iterations=500)
        if len(final) < best_len:
            best_blocks = final
            best_len = len(final)
            
    if best_blocks is None:
        # Fallback
        best_blocks = standard_greedy(0)
        best_blocks = local_search(best_blocks, 0, max_iterations=100)
        
    # Convert indices to actual blocks
    return [all_blocks[i] for i in best_blocks]
