import itertools
import time
import random
import sys
import numpy as np
from scipy.optimize import linprog

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Precomputation
    # Map each t-subset to an index
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Map each k-subset (block) to an index
    # There are C(19, 8) = 75582 blocks. This is manageable.
    k_subs = list(itertools.combinations(range(v), k))
    k_sub_to_idx = {sub: i for i, sub in enumerate(k_subs)}
    n_k = len(k_subs)
    
    # Build incidence matrix A (n_t x n_k)
    # A[i, j] = 1 if t_sub i is contained in k_sub j
    # Using NumPy for speed.
    # A[i, j] = 1 if set(t_subs[i]) is subset of set(k_subs[j])
    
    # Optimization: 
    # For each k-subset, generate all t-subsets it covers.
    # This is C(8, 5) = 56 per block. Total 75582 * 56 ~ 4.2M operations.
    # This is much faster than checking all pairs.
    
    # We want A as a sparse matrix or dense float matrix for linprog.
    # linprog accepts sparse matrices. Let's build a sparse matrix structure or just dense if small enough.
    # 116280 * 75582 ~ 8.8 billion entries. Too big for dense.
    # But we only need non-zeros.
    # Let's use scipy.sparse if available, or just build the constraint matrix row by row?
    # Actually, linprog can take sparse arrays.
    
    # Let's construct the sparse matrix A.
    # Rows: t-subsets (0 to n_t-1)
    # Cols: k-subsets (0 to n_k-1)
    
    # We will accumulate row indices, column indices, and data (all 1s).
    row_indices = []
    col_indices = []
    
    # For each k-subset (column j), find all t-subsets (row i) it covers.
    for j, k_sub in enumerate(k_subs):
        # Generate all t-subsets of this k-subset
        for t_sub in itertools.combinations(k_sub, t):
            i = t_sub_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            
    # Create sparse matrix
    try:
        from scipy.sparse import coo_matrix
        A = coo_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(n_t, n_k))
        A = A.tocsr() # Compressed Sparse Row is good for linprog? Actually linprog handles sparse input well.
        # Ensure it's a format linprog likes. It accepts sparse.
    except ImportError:
        # Fallback to dense if scipy.sparse fails (unlikely)
        # This would be 116280 * 75582 floats = ~35GB. Impossible.
        # So scipy.sparse is required.
        A = np.zeros((n_t, n_k))
        for r, c in zip(row_indices, col_indices):
            A[r, c] = 1.0

    # 2. Solve LP Relaxation
    # Minimize sum(x_j)
    # Subject to A @ x >= 1
    # x >= 0
    
    c = np.ones(n_k)
    
    # linprog standard form: A_ub @ x <= b_ub
    # We have A @ x >= 1  =>  -A @ x <= -1
    A_ub = -A
    b_ub = -np.ones(n_t)
    
    # Bounds: 0 <= x_j <= 1 (Upper bound 1 is valid because if x_j > 1, we can set it to 1 and still satisfy constraints, reducing cost)
    bounds = [(0, 1)] * n_k
    
    # Solve
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            # res.x gives the optimal fractional solution
            # The dual variables (shadow prices) are in res.ineqlin.marginals
            # These represent the "value" of each constraint (t-subset).
            # High marginal => this t-subset is "expensive" to cover / hard to cover.
            dual_vars = res.ineqlin.marginals
        else:
            dual_vars = np.ones(n_t) # Fallback
    except Exception:
        dual_vars = np.ones(n_t)
        
    # 3. Weighted Greedy Construction
    # We want to pick blocks.
    # Score of a block = sum of dual_vars of the t-subsets it covers that are currently uncovered.
    # We maintain a bitset of uncovered t-subsets.
    
    # Precompute for each block, the list of t-subset indices it covers.
    # We already have row_indices and col_indices.
    # Let's invert this: for each block j, list of t-subset indices.
    block_to_tsubs = [[] for _ in range(n_k)]
    for r, c in zip(row_indices, col_indices):
        block_to_tsubs[c].append(r)
        
    # Convert to numpy arrays for faster summing? Or just lists.
    # List of lists is fine.
    
    # State:
    uncovered = np.ones(n_t, dtype=bool)
    blocks_selected = []
    
    # Optimization: 
    # Instead of recalculating scores for all 75k blocks from scratch each step (75k * 56 ops * 300 steps ~ 1.2B ops, might be slow in Python),
    # we can use a heap or just iterate.
    # Let's try iterating first. 75k blocks * 56 subsets = 4.2M checks per step.
    # If we need 300 blocks, that's 1.26 billion operations. Python might take > 100s.
    # We need a faster update or a smarter selection.
    
    # Alternative: 
    # Use the LP solution x directly? 
    # Round the LP solution?
    # The LP solution x_j is the probability/fraction.
    # We can sort blocks by x_j descending and pick the top ones?
    # Or use x_j to weight the greedy?
    
    # Let's try a hybrid:
    # 1. Sort blocks by LP value x_j descending.
    # 2. Iterate through them. If adding this block covers any uncovered t-subset, add it.
    # This is "Greedy by LP ordering".
    
    # This is much faster: O(n_k * (cost to check coverage)).
    # Cost to check coverage: iterate through t-subsets of the block, check if uncovered.
    # 75k blocks * 56 checks = 4.2M checks total. Very fast.
    
    # Let's try this strategy first.
    # Get LP solution x
    if 'x' in res and res.x is not None:
        lp_solution = res.x
    else:
        lp_solution = np.zeros(n_k)
        
    # Sort block indices by lp_solution descending
    sorted_blocks = np.argsort(lp_solution)[::-1]
    
    # Iterate and select
    for idx in sorted_blocks:
        if not np.any(uncovered):
            break
        if time.time() > deadline - 10:
            break
            
        block = k_subs[idx]
        # Check if it covers anything new
        # We need to check if any t-subset in this block is still uncovered.
        # And then mark them as covered.
        
        # Get t-subset indices for this block
        t_indices = block_to_tsubs[idx]
        
        # Check if any are uncovered
        # uncovered is a numpy bool array
        # mask = uncovered[t_indices]
        # if np.any(mask):
        #     uncovered[t_indices] = False
        #     blocks_selected.append(block)
        
        # To be efficient, we can just check the first one? No, need to mark all.
        # Let's just do the numpy slice.
        if np.any(uncovered[t_indices]):
            uncovered[t_indices] = False
            blocks_selected.append(list(block))
            
    # 4. Local Search / Refinement
    # The "Greedy by LP ordering" might leave some holes if the LP solution is degenerate or if the ordering is bad.
    # Also, it might include redundant blocks.
    
    # Check validity
    if np.any(uncovered):
        # We missed some. We need to cover the remaining uncovered t-subsets.
        # Find an uncovered t-subset
        remaining_uncovered_indices = np.where(uncovered)[0]
        for t_idx in remaining_uncovered_indices:
            if time.time() > deadline - 5:
                break
            t_sub = t_subs[t_idx]
            # Find a block containing this t_sub
            # We can just pick the first block from k_subs that contains t_sub?
            # Or search?
            # Since we have block_to_tsubs, we can invert? No, that's expensive.
            # Let's just iterate k_subs and find one.
            # This is O(n_k * k) which is fine for a few missing subsets.
            for b_idx, b in enumerate(k_subs):
                if set(t_sub).issubset(set(b)):
                    # Add this block
                    blocks_selected.append(list(b))
                    # Mark all t-subsets in this block as covered
                    # We need to find the t_indices for this block
                    # We have block_to_tsubs[b_idx]
                    uncovered[block_to_tsubs[b_idx]] = False
                    break
                    
    # Now we have a valid cover (hopefully).
    # Remove redundant blocks.
    # This is the critical step for score.
    
    # Strategy for redundancy removal:
    # Iterate through blocks. For each block, check if it's the *only* block covering any t-subset.
    # If not, remove it.
    
    # To do this efficiently, we need to know the "count" of how many blocks cover each t-subset.
    # Let's build a count array.
    cover_count = np.zeros(n_t, dtype=int)
    
    # First pass: count
    for block in blocks_selected:
        # Find block index
        # This lookup is slow if we do it via dict every time.
        # Let's store the block indices we selected.
        pass
        
    # Let's re-do the selection keeping indices.
    # Actually, let's just rebuild the cover counts from the final list of blocks.
    # We need to map block tuple to index.
    # k_sub_to_idx exists.
    
    for block in blocks_selected:
        b_idx = k_sub_to_idx[tuple(block)]
        for t_idx in block_to_tsubs[b_idx]:
            cover_count[t_idx] += 1
            
    # Now, try to remove blocks.
    # We iterate through the list.
    # If a block has no "exclusive" t-subset (i.e., all its t-subsets have cover_count > 1), it can be removed.
    # When we remove a block, we decrement cover_count for its t-subsets.
    
    # To make this efficient, we should process blocks in an order that might allow more removals?
    # Or just iterate once? Multiple passes are better.
    
    # Let's do a few passes.
    # Note: Removing a block changes cover_count, which might make other blocks removable.
    
    # Optimization: 
    # If we remove a block, we don't need to re-check blocks already processed in this pass?
    # Actually, if we remove block A, it might make block B (processed earlier) removable? 
    # No, if B was not removable before, it had an exclusive t-subset. Removing A (which doesn't cover that exclusive t-subset, because if it did, B wouldn't have been exclusive... wait.
    # If B has exclusive t-subset S, then only B covers S. A does not cover S. So removing A doesn't affect S's count. 
    # So B remains non-removable.
    # So one pass is sufficient to find all blocks that are "currently" redundant given the others?
    # Wait. If we remove A, the count of t-subsets covered by A decreases.
    # This might make a t-subset covered by C (processed later) become exclusive to C?
    # No, removing A *decreases* counts. It makes it *harder* for a t-subset to be non-exclusive.
    # It makes it *easier* for a t-subset to be exclusive?
    # If count(S) was 2 (A and C). Remove A. Count(S) becomes 1 (C).
    # So C becomes "more" essential? No, C was covering S. If C was the *only* one left, then S is exclusive to C.
    # So removing blocks makes other blocks *less* likely to be removable (more likely to be essential).
    # Therefore, if a block is removable *before* removing others, it is removable.
    # If it's not removable, it might become removable?
    # No. If it's not removable, it has an exclusive t-subset S.
    # Removing other blocks does not remove S from that block's coverage.
    # And it does not add coverage to S from other blocks.
    # So S remains exclusive to that block.
    # So the set of removable blocks is monotonic?
    # If B is removable, and we remove A, is B still removable?
    # B is removable if all t-subsets in B have count > 1.
    # Removing A decreases counts.
    # So a t-subset in B might drop from 2 to 1.
    # Then B is *no longer* removable.
    # So we should remove blocks in an order?
    # Actually, if we remove blocks, we make the remaining blocks *more* critical.
    # So we should check for removability *before* removing?
    # Or, we can just iterate and remove if currently redundant.
    # But if we remove A, and B was redundant (count 2 for all its t-subs), and one of B's t-subs was covered by A, then B's count drops to 1, so B becomes essential.
    # So we should NOT remove B if we are going to remove A?
    # This implies we should find a set of blocks to remove simultaneously?
    # Or just do a single pass from end to start?
    # If we iterate from end to start, and remove B, then check A...
    # Let's just do a single pass from end to start.
    # If block i is redundant (all its t-subs have count > 1), remove it.
    # This is safe.
    
    new_blocks = []
    # We need to be careful with the order.
    # Let's just iterate through the list.
    # We'll keep a list of blocks to keep.
    
    # Actually, a simpler heuristic:
    # Just try to remove each block one by one in a loop until no more can be removed.
    # Since n_blocks is ~300-400, and n_t is 116k.
    # Checking redundancy for one block: 56 lookups.
    # Removing one block: 56 updates.
    # If we do 300 removals, that's 300 * 56 * 2 ~ 33k ops. Very fast.
    # The bottleneck is the "while improved" loop if it takes many iterations.
    # But usually, one or two passes clear most redundancy.
    
    # Let's implement the iterative removal.
    
    # We need to map block tuple to its index in block_to_tsubs for fast access.
    # We have k_sub_to_idx.
    
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        i = 0
        while i < len(blocks_selected):
            block = blocks_selected[i]
            b_idx = k_sub_to_idx[tuple(block)]
            t_indices = block_to_tsubs[b_idx]
            
            # Check if any t-subset in this block has count == 1
            # If yes, this block is essential. Keep it.
            # If no, this block is redundant. Remove it.
            
            # np.any(cover_count[t_indices] == 1)
            if np.any(cover_count[t_indices] == 1):
                i += 1
            else:
                # Remove block
                blocks_selected.pop(i)
                # Update counts
                cover_count[t_indices] -= 1
                changed = True
                # Do not increment i, check next block (which is now at i)
                
    # Final verification (just in case)
    # Rebuild cover count to ensure validity?
    # The logic above preserves validity if implemented correctly.
    # Removing a block only if all its t-subsets are covered by others ensures no hole is opened.
    
    # Return list of blocks
    return blocks_selected

