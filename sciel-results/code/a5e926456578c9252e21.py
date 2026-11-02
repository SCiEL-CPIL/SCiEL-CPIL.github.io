import itertools
import random
import time
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using LP-dual-guided greedy set cover
    followed by local search reduction.
    """
    start_time = time.time()
    timeout = 140  # Safety margin for 150s limit
    
    # 1. Precompute all t-subsets and k-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the indices of t-subsets it covers
    # This is needed for both LP setup and greedy construction
    block_to_tsubs = []
    for block in all_blocks:
        covered_indices = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_tsubs.append(covered_indices)
    
    # 2. Solve the Dual LP to get weights y_t
    # Dual: Max sum(y_t) subject to sum_{t in B} y_t <= 1 for all blocks B, y_t >= 0
    # This is equivalent to: Min -sum(y_t) subject to sum_{t in B} y_t <= 1, y_t >= 0
    
    # Build the constraint matrix for the dual problem
    # Variables: y (size n_tsubs)
    # Constraints: For each block b, sum_{t in b} y_t <= 1
    # A_ub is n_blocks x n_tsubs
    # We build it sparsely
    
    A_ub = lil_matrix((n_blocks, n_tsubs))
    for b_idx, t_indices in enumerate(block_to_tsubs):
        for t_idx in t_indices:
            A_ub[b_idx, t_idx] = 1.0
            
    A_ub = A_ub.tocsr()
    
    # Objective: Min -sum(y) => c = -1 for all y
    c = -np.ones(n_tsubs)
    
    # Bounds: y >= 0
    bounds = [(0, None)] * n_tsubs
    
    # Solve LP
    # Using 'highs' is generally robust
    try:
        res = linprog(c, A_ub=A_ub, b=np.ones(n_blocks), bounds=bounds, method='highs')
        if res.status == 0:
            y_dual = res.x
        else:
            # Fallback to uniform weights if LP fails
            y_dual = np.ones(n_tsubs)
    except Exception:
        y_dual = np.ones(n_tsubs)
    
    if time.time() - start_time > timeout - 20:
        # If LP took too long, just use uniform weights and proceed with greedy
        y_dual = np.ones(n_tsubs)

    # 3. Weighted Greedy Set Cover
    # We want to cover all t-subsets.
    # At each step, pick block B that maximizes sum of y_t for t in B that are currently uncovered.
    
    uncovered = set(range(n_tsubs))
    current_blocks = []
    
    # To speed up greedy, we can pre-sort blocks or use a priority queue, 
    # but the gains change as uncovered set changes.
    # Simple approach: Iterate through blocks, calculate gain, pick best.
    # This is O(N_blocks * N_tsubs_per_block) per step.
    # Number of steps is roughly N_tsubs / (k choose t) * something.
    # Actually, number of blocks in solution is ~150.
    # So 150 steps.
    # 150 * 50388 * 35 (approx tsubs per block) ~ 260M ops. This might be slow in Python.
    
    # Optimization: 
    # Instead of iterating all blocks every time, we can be smarter.
    # But let's try the direct approach first with NumPy vectorization if possible, 
    # or just optimize the inner loop.
    
    # Let's use a slightly different greedy:
    # Pick a random uncovered t-subset? No, weighted greedy is better.
    # We can approximate the best block by maintaining a list of candidate blocks?
    
    # Given the time limit, a pure Python loop over 50k blocks 150 times is 7.5M iterations.
    # Each iteration involves summing ~35 floats. ~260M float adds.
    # This is likely too slow in pure Python (10-20 seconds per greedy run is possible).
    # We need to be faster.
    
    # Alternative: Use the LP solution x* (primal) if available? 
    # We solved dual. We can get primal from dual solution? Not directly from linprog output if we only asked for dual.
    # But we can run a second LP for primal? No, time.
    
    # Let's optimize the greedy step using NumPy.
    # Represent uncovered as a boolean array or mask.
    # Gain for block b = sum(y_dual[t] for t in block_to_tsubs[b] if t in uncovered)
    
    # We can precompute block_to_tsubs as a list of arrays.
    block_tsubs_arrays = [np.array(t_indices, dtype=np.int32) for t_indices in block_to_tsubs]
    
    # We will perform the greedy selection.
    # To speed up, we can restrict candidates.
    # But initially, let's just try to make the inner loop fast.
    
    # Actually, a better heuristic for weighted greedy:
    # 1. Calculate initial gain for all blocks: gain[b] = sum(y[t] for t in block[b])
    # 2. Pick max gain block b*.
    # 3. Add b*. Update uncovered.
    # 4. For all blocks b, update gain[b] by subtracting y[t] for t in b* intersect b.
    #    This requires knowing intersections.
    
    # This is complex. Let's stick to the simpler "pick best from all" but optimize the gain calculation.
    # Since y_dual is fixed, and uncovered changes, we can't precompute everything.
    
    # However, we can use a "Lazy" approach:
    # Only re-evaluate gains for blocks that might have changed?
    # No, gains change for all blocks that intersect the newly covered t-subsets.
    
    # Let's try a different tactic: Randomized Weighted Greedy with a time check.
    # If it's too slow, we fall back to a simpler heuristic.
    
    # Let's try to run the greedy in a way that is vectorized where possible.
    # But the "uncovered" check is the bottleneck.
    
    # Let's try this:
    # Instead of checking all 50k blocks every time, we can sample?
    # No, we need good coverage.
    
    # Let's use the fact that we have y_dual.
    # We can sort blocks by their initial total weight (sum of y_dual in block).
    # Blocks with higher total weight are more likely to be useful.
    # We can iterate through blocks in this sorted order, and pick the first one that covers the most uncovered?
    # Or just pick the first one that covers ANY uncovered? No, that's weak.
    
    # Let's try a hybrid:
    # 1. Sort blocks by initial potential (sum of y_dual).
    # 2. In each step, look at the top M blocks (e.g., 1000) and pick the one with max gain on uncovered.
    # 3. If no block in top M covers any uncovered (unlikely), expand M.
    
    initial_potentials = np.array([np.sum(y_dual[ts]) for ts in block_to_tsubs])
    sorted_block_indices = np.argsort(initial_potentials)[::-1]
    
    # Greedy Loop
    uncovered_mask = np.ones(n_tsubs, dtype=bool)
    current_blocks = []
    
    # Precompute block_tsubs as lists for fast iteration
    # We already have block_to_tsubs as lists of ints.
    
    while np.any(uncovered_mask):
        if time.time() - start_time > timeout - 10:
            break
            
        best_gain = -1
        best_b_idx = -1
        
        # Check top candidates
        # Start with top 500, expand if necessary
        check_limit = 500
        while check_limit <= n_blocks:
            found = False
            for b_idx in sorted_block_indices[:check_limit]:
                t_indices = block_to_tsubs[b_idx]
                # Calculate gain
                gain = 0
                # Optimization: if gain + max_possible < best_gain, skip?
                # Max possible is sum of all y in block.
                # We have initial_potentials[b_idx].
                if initial_potentials[b_idx] <= best_gain:
                    continue
                
                for t_idx in t_indices:
                    if uncovered_mask[t_idx]:
                        gain += y_dual[t_idx]
                        # Early exit if we can't beat best?
                        # If gain + (initial_potentials[b_idx] - current_sum) ... no, just compute full.
                        # But we can break if gain + remaining_max < best_gain?
                        # remaining_max is hard to estimate quickly.
                        
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
                    if gain > 0:
                        found = True
                        # We can break early if we find a "perfect" cover? No.
                        # But we can break the inner loop over candidates? No, we want the BEST among candidates.
            
            if found and best_gain > 0:
                break
            check_limit *= 2
            
        if best_b_idx == -1 or best_gain == 0:
            # Fallback: pick any block that covers something
            # This shouldn't happen if uncovered is not empty and we checked all blocks eventually
            # Force check all
            for b_idx in range(n_blocks):
                t_indices = block_to_tsubs[b_idx]
                gain = 0
                for t_idx in t_indices:
                    if uncovered_mask[t_idx]:
                        gain += y_dual[t_idx]
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
            
            if best_b_idx == -1:
                break # Should not happen
        
        current_blocks.append(best_b_idx)
        
        # Update uncovered mask
        for t_idx in block_to_tsubs[best_b_idx]:
            uncovered_mask[t_idx] = False
        
    # 4. Local Search Reduction
    # Convert current_blocks to set for faster lookup? No, list is fine for counts.
    tsub_count = np.zeros(n_tsubs, dtype=np.int32)
    for b_idx in current_blocks:
        for t_idx in block_to_tsubs[b_idx]:
            tsub_count[t_idx] += 1
            
    improved = True
    while improved:
        improved = False
        for i in range(len(current_blocks)):
            if time.time() - start_time > timeout - 5:
                break
                
            b_idx = current_blocks[i]
            # Check if redundant
            is_redundant = True
            for t_idx in block_to_tsubs[b_idx]:
                if tsub_count[t_idx] == 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                current_blocks.pop(i)
                for t_idx in block_to_tsubs[b_idx]:
                    tsub_count[t_idx] -= 1
                improved = True
                break # Restart loop

    # 5. Finalize
    if not current_blocks:
        # Fallback to simple greedy if LP/greedy failed
        # This is a safety net
        uncovered = set(range(n_tsubs))
        simple_blocks = []
        while uncovered:
            # Pick random uncovered tsub
            target = random.choice(list(uncovered))
            # Pick random block covering it
            # We need inverse index. We didn't build it for speed.
            # Let's just return what we have or a trivial covering.
            # Actually, if current_blocks is empty, we have a problem.
            # Let's try a basic greedy again with random seed.
            pass
        
    final_blocks = [list(all_blocks[b_idx]) for b_idx in current_blocks]
    
    return final_blocks
