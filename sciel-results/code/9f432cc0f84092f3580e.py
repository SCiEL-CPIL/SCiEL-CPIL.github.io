import numpy as np
import itertools
import time
import random

def construct(v, k, t):
        # Precomputation
        points = range(v)
        t_subsets = list(itertools.combinations(points, t))
        t_subsets_idx = {s: i for i, s in enumerate(t_subsets)}
        num_t_subsets = len(t_subsets)
        
        # Map each k-subset to the list of t-subset indices it covers
        # We don't want to precompute all C(19,8) if it's slow, but 75k is fine.
        # However, we only need to generate them on the fly for swaps.
        # For the greedy phase, we need to find blocks containing a specific t-subset.
        
        # Precompute: for each t-subset, list of k-subsets containing it.
        # This is C(19-5, 8-5) = 364 per t-subset.
        # Total entries: 11628 * 364 ~ 4.2M.
        # This might be memory heavy but speed is key.
        # Alternative: Generate candidates on the fly.
        
        # Let's try generating candidates on the fly for the greedy phase too.
        # Greedy:
        # 1. Pick random uncovered t-subset S.
        # 2. Generate all k-subsets containing S.
        #    How? S has t points. We need k-t more points from v-t points.
        #    C(14, 3) = 364 combinations.
        # 3. For each candidate block B, count how many *other* uncovered t-subsets it covers.
        # 4. Pick B with max count.
        
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        # Helper to get t-subset indices for a block
        def get_block_t_indices(block):
            # block is a tuple of k points
            # generate all t-combinations of block
            return [t_subsets_idx[s] for s in itertools.combinations(block, t)]
        
        # Greedy Loop
        while uncovered:
            # Pick a random uncovered t-subset
            s_idx = random.choice(list(uncovered))
            s = t_subsets[s_idx]
            
            # Generate all k-subsets containing s
            # s is a tuple of t points
            # remaining points = [p for p in points if p not in s]
            s_set = set(s)
            remaining = [p for p in points if p not in s_set]
            
            best_block = None
            best_gain = -1
            
            # Iterate through combinations of k-t points from remaining
            for combo in itertools.combinations(remaining, k - t):
                block = tuple(sorted(s + combo))
                # Calculate gain: number of uncovered t-subsets in this block
                # We need to check which of the t-subsets of this block are in 'uncovered'
                gain = 0
                # Optimization: We know s is in uncovered. So gain is at least 1.
                # We need to count the others.
                # To do this fast, we need the indices of the t-subsets of this block.
                # Generating them takes 56 iterations.
                # We can just iterate through the t-subsets of the block and check membership in 'uncovered'.
                for ts in itertools.combinations(block, t):
                    if t_subsets_idx[ts] in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            # Add best_block
            blocks.append(best_block)
            
            # Update uncovered
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(t_subsets_idx[ts])
        
        # Now we have a valid cover.
        # Optimize: Deletion and Swap
        
        # Data structures for optimization
        # ref_count: array of size num_t_subsets
        ref_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                ref_count[t_subsets_idx[ts]] += 1
        
        current_size = len(blocks)
        best_size = current_size
        best_blocks = list(blocks)
        
        # Helper to get t-subset indices for a block (cached or fast)
        # We will compute on the fly to save memory, but it's a bottleneck.
        # Let's cache the indices for blocks in the current solution.
        block_indices = {b: [t_subsets_idx[ts] for ts in itertools.combinations(b, t)] for b in blocks}
        
        deadline = time.time() + 100 # 100s limit
        
        # Deletion Pass
        def deletion_pass(blocks, ref_count, block_indices):
            changed = False
            new_blocks = []
            for b in blocks:
                # Tentatively remove
                indices = block_indices[b]
                for idx in indices:
                    ref_count[idx] -= 1
                
                # Check if any count is 0
                if np.min(ref_count) > 0:
                    # Redundant, keep removed
                    changed = True
                else:
                    # Not redundant, put back
                    for idx in indices:
                        ref_count[idx] += 1
                    new_blocks.append(b)
                    # Update block_indices if needed? No, b is still there.
            
            if changed:
                # Rebuild block_indices for new_blocks?
                # Actually, we just need to return the new list.
                # We must be careful with block_indices dict.
                pass
            return new_blocks, changed

        # We need a more efficient structure.
        # Let's maintain:
        # 1. List of blocks
        # 2. Dict block -> list of t-indices
        # 3. ref_count array
        
        # Optimization Loop
        while time.time() < deadline:
            # 1. Deletion Pass
            # We iterate through a copy of the list to allow removal
            i = 0
            while i < len(blocks):
                b = blocks[i]
                indices = block_indices[b]
                # Decrement
                for idx in indices:
                    ref_count[idx] -= 1
                
                if np.min(ref_count) > 0:
                    # Remove
                    blocks.pop(i)
                    del block_indices[b]
                    current_size -= 1
                    # Do not increment i
                else:
                    # Restore
                    for idx in indices:
                        ref_count[idx] += 1
                    i += 1
            
            if current_size < best_size:
                best_size = current_size
                best_blocks = list(blocks)
            
            # 2. Swap Phase
            if len(blocks) == 0:
                break
                
            # Try some swaps
            num_swaps = 100 # Per iteration?
            for _ in range(num_swaps):
                if time.time() > deadline:
                    break
                    
                # Pick random block to swap out
                i = random.randint(0, len(blocks) - 1)
                b_old = blocks[i]
                old_indices = block_indices[b_old]
                
                # Find critical subsets (count == 1)
                # We need to check ref_count for old_indices
                criticals = []
                for idx in old_indices:
                    if ref_count[idx] == 1:
                        criticals.append(idx)
                
                if not criticals:
                    # Should have been deleted
                    continue
                
                # Generate random b_new
                # It must cover all criticals.
                # Strategy:
                # 1. Pick a random point from criticals? No.
                # 2. Construct b_new such that it contains the points of the criticals?
                #    A critical t-subset is a set of t points.
                #    b_new must contain all these t points.
                #    So b_new must be a superset of the union of criticals?
                #    No, b_new must contain *each* critical t-subset.
                #    So the points of b_new must include the points of every critical t-subset.
                #    Let U = union of all points in criticals.
                #    b_new must contain U.
                #    If len(U) > k, no such block exists. Swap fails.
                
                u_set = set()
                for idx in criticals:
                    u_set.update(t_subsets[idx])
                
                if len(u_set) > k:
                    continue
                
                # We need to pick k - len(u_set) more points from the remaining v - len(u_set) points.
                remaining_points = [p for p in points if p not in u_set]
                if len(remaining_points) < k - len(u_set):
                    continue
                
                # Randomly select the rest
                extra = random.sample(remaining_points, k - len(u_set))
                b_new = tuple(sorted(list(u_set) + extra))
                
                # Check if b_new is already in blocks (and not b_old)
                # Using a set of blocks for O(1) lookup
                # We don't have a set of blocks. Let's maintain one.
                # Actually, checking against a list is O(N). N~300. 300*100 swaps = 30k checks. Fine.
                # But let's just check if b_new == b_old. If so, skip.
                if b_new == b_old:
                    continue
                
                # Check if b_new is in current blocks
                # If it is, swapping is a no-op or swap with identical block.
                # Let's just allow it, it will be handled by deletion later if redundant.
                # But if b_new is already there, we are just moving a block.
                
                # Calculate new indices for b_new
                new_indices = [t_subsets_idx[ts] for ts in itertools.combinations(b_new, t)]
                
                # Check validity:
                # Remove b_old, Add b_new.
                # Decrement old
                for idx in old_indices:
                    ref_count[idx] -= 1
                # Increment new
                for idx in new_indices:
                    ref_count[idx] += 1
                
                if np.min(ref_count) > 0:
                    # Valid swap
                    # Accept?
                    # We accept if it leads to improvement or randomly.
                    # Let's accept and then run deletion.
                    # If deletion reduces size, great.
                    # If not, we might revert.
                    
                    # Perform swap
                    blocks[i] = b_new
                    block_indices[b_new] = new_indices
                    if b_old in block_indices:
                        del block_indices[b_old]
                    
                    # Run a quick deletion pass
                    # We can just call the deletion logic again?
                    # It might be expensive.
                    # Let's just check if any block is now redundant.
                    # Actually, the swap itself doesn't change size.
                    # The value is in enabling deletion.
                    
                    # Let's do a full deletion pass after a batch of swaps?
                    # Or just one?
                    
                    # For now, let's just keep the swap.
                    # The main loop will do deletion.
                    
                else:
                    # Invalid, revert
                    for idx in new_indices:
                        ref_count[idx] -= 1
                    for idx in old_indices:
                        ref_count[idx] += 1
        
        return best_blocks
