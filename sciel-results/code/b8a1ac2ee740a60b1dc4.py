import numpy as np
import itertools
import time
import random
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s/150s kill
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute block to triple indices
    # We use a list of arrays for fast access, or a sparse matrix
    # Sparse matrix is better for global dot products
    rows = []
    cols = []
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for sub in subs:
            rows.append(i)
            cols.append(idx_map[sub])
            
    # Create sparse matrix M (n_ks x n_ts)
    data = np.ones(len(rows), dtype=np.int32)
    M = sparse.csr_matrix((data, (rows, cols)), shape=(n_ks, n_ts))
    
    # Convert to COO for easy row access if needed, or keep CSR
    # CSR is good for row slicing
    
    best_solution = None
    best_score = float('inf')
    
    def run_grasp(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        # --- Phase 1: Greedy Construction ---
        uncovered = np.ones(n_ts, dtype=np.int32)
        selected_blocks = []
        
        # Pre-compute row indices for M to speed up updates
        # M is CSR, so M.getrow(i) is fast, but we can just use M[i]
        
        while np.sum(uncovered) > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            # Calculate gain for each block: number of currently uncovered triples it covers
            # gains = M.dot(uncovered)
            # Since M is sparse and uncovered is dense, this is efficient
            gains = M.dot(uncovered)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find indices with max gain
            # To diversify, pick randomly from top K
            # Threshold: max_gain
            # Get all indices with gain == max_gain
            # This might be many. Let's take top 20 random ones.
            
            # Efficient way to get top K:
            # argsort is O(N log N). N=346k. 346k log 346k ~ 6M ops.
            # Doing this every step (up to ~80 steps) is ~500M ops. Might be slow.
            
            # Alternative: Random sampling?
            # Or: Use argpartition.
            
            # Let's try argpartition for top 50
            if n_ks > 50:
                top_indices = np.argpartition(gains, -50)[-50:]
            else:
                top_indices = np.arange(n_ks)
                
            # Filter top_indices to only those with max_gain? 
            # Actually, picking from top 50 is a good heuristic (GRASP style)
            
            best_idx = top_indices[random.randint(0, len(top_indices) - 1)]
            
            selected_blocks.append(best_idx)
            
            # Update uncovered
            # uncovered -= M[best_idx].toarray()
            # M[best_idx] is a sparse row. 
            # Converting to array is fast for 1 row of 2024 elements.
            row = M[best_idx].toarray().flatten()
            uncovered -= row
            
        current_blocks = list(selected_blocks)
        
        # --- Phase 2: Local Search (Deletion + Exchange) ---
        
        # Helper: Update coverage and try to delete
        def update_and_delete(blocks):
            if not blocks:
                return blocks, np.zeros(n_ts, dtype=np.int32)
            
            # Calculate coverage
            coverage = np.zeros(n_ts, dtype=np.int32)
            for b in blocks:
                coverage += M[b].toarray().flatten()
            
            # Iterative Deletion
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for b in blocks:
                    row = M[b].toarray().flatten()
                    # Check if any triple in this block is covered ONLY by this block
                    # i.e., coverage[triples] == 1
                    # If min(coverage[row > 0]) > 1, it is redundant
                    if np.all(coverage[row > 0] > 1):
                        # Redundant, do not add to new_blocks
                        coverage -= row
                        changed = True
                    else:
                        new_blocks.append(b)
                blocks = new_blocks
                
            return blocks, coverage
            
        current_blocks, coverage = update_and_delete(current_blocks)
        
        # Exchange Phase (Hill Climbing)
        # Try to swap a block with another to allow deletion of a 3rd block
        # Simplified: Try to remove a block. If fails, find a replacement that covers the lost triples.
        
        improved = True
        while improved and (time.time() - start_time < time_limit - 2.0):
            improved = False
            # Shuffle blocks to vary search order
            random.shuffle(current_blocks)
            
            for i, b_old in enumerate(current_blocks):
                if time.time() - start_time > time_limit - 2.0:
                    break
                    
                # Try to remove b_old
                row_old = M[b_old].toarray().flatten()
                # Check if b_old is essential
                # If any triple in b_old has coverage == 1, it's essential
                lost_triples_mask = (coverage == 1) & (row_old == 1)
                
                if not np.any(lost_triples_mask):
                    # Can remove directly
                    coverage -= row_old
                    current_blocks.pop(i)
                    improved = True
                    # After removal, other blocks might become redundant
                    current_blocks, coverage = update_and_delete(current_blocks)
                    # Restart loop as indices changed
                    break 
                
                # If essential, try to find a replacement b_new that covers lost_triples
                # We want b_new to cover ALL lost_triples? Or just enough to allow deletion of b_old?
                # If we add b_new, coverage increases. Then we try to remove b_old.
                # If b_old becomes redundant after adding b_new, we can remove it.
                # Net change: +1 (add b_new), -1 (remove b_old) = 0.
                # But then we run update_and_delete, which might remove MORE blocks.
                
                # Find best b_new
                # Score for b_new: number of lost_triples it covers
                # Also, prefer b_new that is NOT already in current_blocks?
                # If b_new is already in current_blocks, we just increase its coverage count.
                
                lost_indices = np.where(lost_triples_mask)[0]
                
                # Calculate coverage of lost triples for all blocks
                # gains_new = M[:, lost_indices].sum(axis=1)
                # This is a sub-matrix dot product.
                # M is CSR. M[:, lost_indices] is inefficient.
                # Better: 
                # For each block, count how many lost triples it covers.
                
                # Optimization: Only check blocks that cover at least one lost triple.
                # We can precompute for each triple, which blocks cover it.
                # But that's a large structure.
                
                # Let's just calculate gains for all blocks, but only for lost triples.
                # M_lost = M[:, lost_indices]
                # This creates a copy. 
                
                # Alternative:
                # gains = np.zeros(n_ks)
                # for idx in lost_indices:
                #    gains += M[:, idx].toarray().flatten()
                # This is O(len(lost) * nnz_in_col).
                # len(lost) is small (<= 35).
                # nnz_in_col is ~ n_ks * 35 / n_ts ~ 346k * 35 / 2024 ~ 6000.
                # 35 * 6000 = 210k ops. Very fast.
                
                gains_new = np.zeros(n_ks, dtype=np.int32)
                for idx in lost_indices:
                    # M[:, idx] is a sparse column
                    # Convert to array
                    gains_new += M[:, idx].toarray().flatten()
                    
                # Find best candidate
                # We want to maximize gains_new.
                # Also, we should avoid picking a block that is already in current_blocks if possible?
                # If we pick a block already in current_blocks, we just increment its count.
                # Then we try to remove b_old.
                # If b_old becomes redundant, we remove it.
                # Size stays same. Then update_and_delete might remove others.
                
                # Let's pick the block with max gains_new
                # Tie-break: random
                max_gain_new = np.max(gains_new)
                if max_gain_new == 0:
                    continue # No block covers the lost triples? Impossible, they are covered by b_old.
                
                # Get candidates with max gain
                candidates = np.where(gains_new == max_gain_new)[0]
                b_new = candidates[random.randint(0, len(candidates) - 1)]
                
                # Apply swap
                # 1. Add b_new
                row_new = M[b_new].toarray().flatten()
                coverage += row_new
                if b_new not in current_blocks:
                    current_blocks.append(b_new)
                else:
                    # If it's already there, we don't add it to the list, 
                    # but coverage is updated.
                    # Wait, my logic for "coverage" assumes 1 block = 1 count.
                    # If I add a block that is already in the list, I need to track multiplicity?
                    # No, the problem requires DISTINCT blocks.
                    # So b_new MUST NOT be in current_blocks.
                    # If b_new is in current_blocks, we can't "add" it again.
                    # So we must filter candidates to exclude current_blocks.
                    
                    # Let's refine:
                    # candidates = [c for c in candidates if c not in current_blocks]
                    # This is slow if done every time.
                    
                    # Let's assume for now we pick a new block.
                    # If the best block is already in the set, we should pick the next best.
                    
                    # Re-evaluate candidates excluding current_blocks
                    # Create a mask
                    in_set = np.zeros(n_ks, dtype=bool)
                    for b in current_blocks:
                        in_set[b] = True
                        
                    valid_candidates = candidates[~in_set[candidates]]
                    
                    if len(valid_candidates) == 0:
                        # All best candidates are already in the set.
                        # Pick a random new block that covers at least 1 lost triple?
                        # Or skip.
                        continue
                        
                    b_new = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                    
                # 2. Try to remove b_old
                # Check if b_old is now redundant
                # coverage has been updated with b_new.
                # Check b_old's rows
                if np.all(coverage[row_old > 0] > 1):
                    # b_old is redundant
                    coverage -= row_old
                    current_blocks.remove(b_old)
                    improved = True
                    # Run deletion pass
                    current_blocks, coverage = update_and_delete(current_blocks)
                    break # Restart loop
                else:
                    # b_old is still essential.
                    # The swap didn't help remove b_old.
                    # Revert the addition of b_new?
                    # If we keep b_new, size increases by 1. Bad.
                    # So we must revert.
                    coverage -= row_new
                    if b_new not in current_blocks: # It shouldn't be, we checked
                        pass
                    # We didn't add it to the list if it was already there, 
                    # but we did add to coverage.
                    # If it was NOT in list, we appended it. So we must pop it.
                    # Wait, my logic above: "if b_new not in current_blocks: append"
                    # So if it was not in list, we appended.
                    # We need to remove it.
                    if b_new in current_blocks:
                        # Remove the last occurrence? Or just pop if it's the last one?
                        # We appended it. So it's the last one.
                        current_blocks.pop()
                    
        return current_blocks, len(current_blocks)

    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_grasp(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
        
    if best_solution is None:
        blocks, score = run_grasp(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
