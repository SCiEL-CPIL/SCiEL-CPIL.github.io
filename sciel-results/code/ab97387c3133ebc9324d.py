import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precomputation
    # 1. Map all t-subsets to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Precompute for each k-block, the indices of the t-subsets it covers
    #    Also, for each t-subset, the list of k-blocks that cover it
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    block_t_indices = [] # list of arrays, each of length C(k,t)
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        # Get all t-subsets of this block
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[s] for s in t_subs_in_block], dtype=np.int32)
        block_t_indices.append(indices)
        
        # Update reverse mapping
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)
            
    # Convert lists to numpy arrays for faster access if needed, 
    # but list of lists is fine for the reverse map since we iterate once.
    
    best_blocks = None
    best_score = float('inf')
    
    # Multiple restarts
    # How many restarts can we do?
    # Each restart involves the greedy phase.
    # Greedy phase: ~200 steps. Each step: 364 candidates * numpy sum.
    # Numpy sum on array of 56 elements is very fast.
    # Let's try to run as many as possible within deadline.
    
    for restart in range(50): # Max 50, break early if deadline approaches
        if time.time() > deadline - 10.0:
            break
            
        random.seed(restart)
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        uncovered_count = num_t_subsets
        blocks = []
        
        # To pick a random uncovered subset, we can maintain a list of uncovered indices?
        # Or just pick a random index and check if coverage==0?
        # Since coverage is dense, picking random and checking is O(1) but might fail often?
        # No, initially all are uncovered. Later, most are covered.
        # Better: maintain a list of uncovered indices.
        # When we add a block, we remove the newly covered ones from the list.
        # This requires efficient removal.
        # Alternatively, just pick a random index from the full range.
        # If it's covered, pick another.
        # As coverage increases, this becomes inefficient.
        
        # Let's use a list of uncovered indices.
        uncovered_indices = list(range(num_t_subsets))
        
        while uncovered_indices:
            if time.time() > deadline - 5.0:
                # If we run out of time, we can't complete the covering.
                # We must return a valid covering. 
                # If we break here, 'blocks' might not cover everything.
                # So we should ensure we finish the current restart if possible, 
                # or fall back to a slower but guaranteed method?
                # Actually, if we don't finish, the result is INVALID.
                # So we must be careful.
                # If time is running out, we should probably stop starting new restarts,
                # and if we are mid-restart, we must finish it.
                # Given the speed, a single restart takes < 1s.
                # So 10s margin is plenty to finish a restart.
                pass
            
            # Pick a random uncovered subset
            # To avoid O(N) scan for an uncovered one, we rely on the list 'uncovered_indices'
            # Pick a random element from the list
            pos = random.randint(0, len(uncovered_indices) - 1)
            u_idx = uncovered_indices[pos]
            
            # Get candidates: blocks containing u_idx
            candidates = t_subset_to_blocks[u_idx]
            
            # Find best candidate
            best_gain = -1
            best_block_idx = -1
            
            # Optimization: We can vectorize the gain calculation for all candidates?
            # Candidates is a list of block indices.
            # We need to sum (coverage[block_t_indices[b]] == 0) for each b in candidates.
            # This is hard to vectorize directly because block_t_indices are different arrays.
            # But we can loop through candidates. 364 is small.
            
            for b_idx in candidates:
                # Count how many of this block's t-subsets are currently uncovered
                # coverage is int32 array
                # block_t_indices[b_idx] is int32 array
                # gain = np.sum(coverage[block_t_indices[b_idx]] == 0)
                # This is fast in numpy.
                gain = np.count_nonzero(coverage[block_t_indices[b_idx]] == 0)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            # Add the best block
            blocks.append(all_k_blocks[best_block_idx])
            
            # Update coverage
            idxs = block_t_indices[best_block_idx]
            # We need to know which of these were 0 (uncovered) to remove from uncovered_indices
            # And update coverage
            # old_cov = coverage[idxs]
            # new_cov = old_cov + 1
            # coverage[idxs] = new_cov
            
            # To update uncovered_indices:
            # We need to remove any idx in idxs that was 0.
            # Since we are using a list, removal is O(N).
            # N starts at 11628.
            # Total removals = 11628.
            # Total operations ~ 11628 * 200 steps? No.
            # Each step removes ~56 items.
            # List removal is O(N).
            # This might be slow if N is large.
            
            # Alternative: Don't maintain a list.
            # Just pick random indices from range(num_t_subsets) until we find one with coverage 0.
            # As coverage gets high, this might take many tries.
            # But we can track `uncovered_count`.
            # Probability of picking uncovered is `uncovered_count / num_t_subsets`.
            # Initially 1.0. Finally 0.
            # Average tries?
            
            # Let's stick to the list but optimize removal.
            # Instead of removing one by one, we can rebuild the list?
            # Or use a set?
            
            # Let's try the "random pick from range" approach first, it's simpler.
            # If it's too slow, we'll switch.
            
            # Actually, let's update coverage first.
            coverage[idxs] += 1
            
            # Now, remove from uncovered_indices any that are now covered.
            # We can filter the list.
            # uncovered_indices = [x for x in uncovered_indices if coverage[x] == 0]
            # This is O(N) per step.
            # 200 steps * 11628 ~ 2.3 million operations. This is fine.
            
            # But we picked `u_idx` from the OLD list.
            # We need to update the list AFTER adding the block.
            # The next iteration will pick from the new list.
            
            # Wait, the list `uncovered_indices` contains indices.
            # We can just filter it.
            uncovered_indices = [x for x in uncovered_indices if coverage[x] == 0]
            
            # Optimization: If the list is getting small, the filtering is fast.
            # If it's large, it's 11628 checks.
            
        # End while loop. We have a complete covering.
        
        # Reduction Phase
        # Try to remove blocks that are not essential.
        # A block is essential if it is the ONLY block covering some t-subset.
        # i.e., if there exists i in block_t_indices[b] such that coverage[i] == 1.
        
        # We can do this in a single pass or multiple passes.
        # Multiple passes might remove more.
        # Let's do a few passes.
        
        # To make removal efficient, we can mark blocks for removal.
        # But if we remove one, coverage drops, making others potentially removable.
        # So we must update coverage as we remove.
        
        # Let's iterate until no more removals.
        # Max blocks ~ 300.
        
        removed_any = True
        while removed_any:
            removed_any = False
            # We need to check blocks in reverse order? Or any order?
            # If we remove a block, it might affect the status of others.
            # Let's just iterate through a copy of the list of block indices.
            
            # Note: `blocks` contains the tuples. We need indices to access precomputed data.
            # We should store block indices in `blocks` during construction.
            
            # Let's refactor: `blocks` stores indices.
            
            # Re-doing the loop with indices:
            # ...
            
            # Since I already wrote the loop above with tuples, let's adapt.
            # It's better to store indices.
            
            # Let's restart the inner logic with indices to be clean.
            
            # Actually, let's just implement the whole thing cleanly in the final code.
            
        # End reduction
        
        if len(blocks) < best_score:
            best_score = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    # If no blocks found (should not happen), return empty?
    # But must be valid.
    # The greedy always finds a covering.
    
    return best_blocks
