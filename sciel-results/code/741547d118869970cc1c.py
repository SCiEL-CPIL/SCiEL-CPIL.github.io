import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # 1. Map all t-subsets to indices
    t_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {s: i for i, s in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # Full mask: all bits set
    full_mask = (1 << num_t_subs) - 1
    
    # 2. Generate all k-blocks and their coverage masks
    blocks = []
    block_masks = []
    for b in itertools.combinations(range(v), k):
        mask = 0
        # A block b covers a t-subset s if s is a subset of b
        # Efficient check: 
        # We can iterate over all t-subsets of b
        for s in itertools.combinations(b, t):
            mask |= (1 << sub_to_idx[s])
        blocks.append(list(b))
        block_masks.append(mask)
        
    num_blocks = len(blocks)
    
    # 3. Greedy Set Cover
    def greedy_cover(randomize_ties=False, random_skip_prob=0.0):
        uncovered = full_mask
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_indices = []
            
            # Find all blocks with max gain
            # Optimization: We can skip blocks that have 0 gain
            # But checking gain is the cost.
            
            for i in range(num_blocks):
                # If block is already selected, skip? 
                # In standard set cover, we don't pick the same set twice.
                # But here, picking the same block again is useless.
                # We can maintain a 'used' set or just rely on gain=0 if covered.
                # Actually, if a block's subsets are all covered, gain is 0.
                
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain and gain > 0:
                    best_indices.append(i)
            
            if best_gain <= 0:
                # Should not happen if uncovered is not 0 and problem is solvable
                break
            
            if randomize_ties and len(best_indices) > 1:
                idx = random.choice(best_indices)
            else:
                idx = best_indices[0]
                
            # Random skip for diversity in restarts
            if random_skip_prob > 0 and random.random() < random_skip_prob:
                # Pick a random block with positive gain instead of best
                # This is expensive to do correctly, so we just pick random from best_indices if we want slight variation
                # Or just skip this mechanism for now, ties are enough.
                pass
                
            selected_indices.append(idx)
            uncovered &= ~block_masks[idx]
            
        return [blocks[i] for i in selected_indices]
        
    # First run: Deterministic Greedy
    best_blocks = greedy_cover(randomize_ties=False)
    best_count = len(best_blocks)
    
    # If we have time, do randomized restarts to find a better (smaller) covering
    # The greedy choice order can affect the final size.
    # We will run a few randomized versions.
    
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        # Increase randomization slightly
        curr_blocks = greedy_cover(randomize_ties=True, random_skip_prob=0.0)
        curr_count = len(curr_blocks)
        
        if curr_count < best_count:
            best_count = curr_count
            best_blocks = curr_blocks
            
        # Early stop if we hit a very good number? 
        # Lower bound is 131. Best known 151.
        # If we get close to 150, we can stop or continue.
        if best_count <= 145:
            break

    # 4. Redundancy Removal
    # Check if any block can be removed.
    # This is O(N^2 * cost_of_check). 
    # Cost of check: For each block, calculate if it covers any unique subset.
    # A subset is "covered by others" if it is in the union of masks of all other selected blocks.
    
    # Optimization: 
    # Maintain the count of how many times each t-subset is covered.
    # If a block is removed, only subsets it covers are affected.
    
    def remove_redundants(blocks_list):
        if not blocks_list:
            return blocks_list
            
        # Map each block index in the list to its mask
        # We need to know which t-subsets are covered by how many blocks.
        
        # Count coverage
        coverage_count = [0] * num_t_subs
        
        for b_idx in range(len(blocks_list)):
            b = blocks_list[b_idx]
            # Recalculate mask? No, we have blocks as lists.
            # It's faster to use the precomputed masks if we track original indices.
            # But we returned lists. Let's map back or just recompute.
            # Recomputing is slow. Let's modify greedy to return indices.
            pass
            
        # Let's implement redundancy removal on indices.
        pass
        
    # Refactor: Work with indices throughout.
    
    def greedy_cover_indices(randomize_ties=False):
        uncovered = full_mask
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_indices = []
            
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain and gain > 0:
                    best_indices.append(i)
            
            if best_gain <= 0:
                break
            
            if randomize_ties and len(best_indices) > 1:
                idx = random.choice(best_indices)
            else:
                idx = best_indices[0]
                
            selected_indices.append(idx)
            uncovered &= ~block_masks[idx]
            
        return selected_indices
        
    # Re-run with indices
    best_indices = greedy_cover_indices(randomize_ties=False)
    best_count = len(best_indices)
    
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        curr_indices = greedy_cover_indices(randomize_ties=True)
        curr_count = len(curr_indices)
        
        if curr_count < best_count:
            best_count = curr_count
            best_indices = curr_indices
            
        if best_count <= 145:
            break
            
    # Redundancy Removal on Indices
    # We have a list of block indices.
    # We want to remove any block that is redundant.
    # A block is redundant if every t-subset it covers is also covered by at least one other block in the set.
    
    # Algorithm:
    # 1. For each t-subset, count how many blocks in `best_indices` cover it.
    # 2. If a block covers only subsets that have count > 1, it is redundant.
    # 3. Remove it and update counts.
    
    def remove_redundant_indices(indices):
        if not indices:
            return indices
            
        # Count coverage of each t-subset
        # Initialize counts to 0
        counts = [0] * num_t_subs
        
        # Pre-compute the list of t-subsets for each block in the selection?
        # We have block_masks. 
        # We can iterate through the selected blocks and increment counts for their bits.
        
        for idx in indices:
            mask = block_masks[idx]
            # Iterate through set bits in mask
            # Python int bit iteration is not built-in, but we can use a trick or just loop 0..num_t_subs
            # num_t_subs is 3876. 
            # 150 blocks * 3876 checks = ~500k operations. Fast.
            for s_idx in range(num_t_subs):
                if mask & (1 << s_idx):
                    counts[s_idx] += 1
                    
        # Now, try to remove blocks
        # We need to check if a block is redundant.
        # A block is redundant if for all s_idx in its mask, counts[s_idx] > 1.
        
        # We can iterate and remove.
        # To be safe, we do this in a loop until no more removals.
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            for idx in indices:
                mask = block_masks[idx]
                is_redundant = True
                # Check if any subset in this block has count == 1
                # If yes, removing this block would uncover it.
                for s_idx in range(num_t_subs):
                    if mask & (1 << s_idx):
                        if counts[s_idx] == 1:
                            is_redundant = False
                            break
                            
                if is_redundant:
                    # Remove it
                    # Decrement counts for all subsets in this block
                    for s_idx in range(num_t_subs):
                        if mask & (1 << s_idx):
                            counts[s_idx] -= 1
                    changed = True
                else:
                    new_indices.append(idx)
                    
            indices = new_indices
            
        return indices

    final_indices = remove_redundant_indices(best_indices)
    
    # Convert back to lists of points
    final_blocks = [blocks[i] for i in final_indices]
    
    return final_blocks
