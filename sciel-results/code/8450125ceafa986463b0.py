import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Solve C(v, k, t) using bitmask-accelerated greedy construction 
    followed by deletion and exchange local search.
    """
    start_time = time.time()
    # 150s limit. Use 140s for safety.
    deadline = start_time + 140.0
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute bitmask for each block
    # 2024 bits fits in a Python int (arbitrary precision, but bitwise ops are C-speed)
    block_masks = [0] * num_blocks
    # Also precompute the list of t-subset indices for each block for local search
    block_t_indices = [None] * num_blocks
    
    # Reverse index: t_subset_idx -> list of block indices covering it
    # This is useful for exchange moves: if we remove block B, we need to find 
    # blocks that cover the "lost" triples.
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, b in enumerate(all_blocks):
        mask = 0
        indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
            indices.append(idx)
            t_to_blocks[idx].append(i)
        block_masks[i] = mask
        block_t_indices[i] = indices
    
    def greedy_construct(seed):
        """
        Standard greedy set cover, but using bitmasks for speed.
        At each step, scan all blocks to find the one with max new coverage.
        """
        random.seed(seed)
        uncovered_mask = (1 << num_t_subsets) - 1
        selected = []
        
        # To add diversity, we can perturb the gain slightly or randomize tie-breaking.
        # Here we use strict max but rely on multiple restarts.
        
        while uncovered_mask != 0:
            if time.time() > deadline:
                return None
            
            best_block_idx = -1
            best_gain = 0
            
            # Scan all blocks
            # Optimization: If a block's mask is a subset of already covered, gain is 0.
            # We can skip blocks where (mask & uncovered_mask) == 0.
            
            for i in range(num_blocks):
                # Quick check: if this block covers no new items, skip
                # mask & uncovered_mask gives the new items
                # We want to count bits.
                # To avoid computing bit_count for all 346k blocks if gain is likely low,
                # we can check if (block_masks[i] & uncovered_mask) == 0 first.
                # But bit_count is fast.
                
                # A slight optimization: 
                # If we have a current best_gain, and the max possible gain for a block is 35,
                # we can't prune much unless best_gain is high.
                
                new_items = block_masks[i] & uncovered_mask
                if new_items == 0:
                    continue
                    
                gain = new_items.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    # If we found a block that covers ALL remaining uncovered items, 
                    # and gain is maximal possible? No, max gain is 35.
                    # If best_gain == 35, it's a perfect block for the current state.
                    # We could break early if we find a block that covers a specific 
                    # "critical" set, but standard greedy just takes the max.
                
                # Early termination for this step if we found a block covering 
                # a large portion? No, just find the max.
                
            if best_block_idx == -1:
                # Should not happen if uncovered_mask != 0
                return None
                
            selected.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
        return selected

    def local_search(blocks, time_limit):
        """
        Perform deletion and 1-for-1 exchange local search.
        Returns a list of block indices.
        """
        if not blocks:
            return blocks
        
        # Use a list of indices
        current_blocks = list(blocks)
        
        # Helper to check if a set of blocks covers all t-subsets
        # We can maintain a global mask of covered items.
        # But for local search, we need to check validity after each move.
        
        def is_valid(block_list):
            # Check if union of masks covers all bits
            # (1 << num_t_subsets) - 1
            total_mask = 0
            for b in block_list:
                total_mask |= block_masks[b]
            return total_mask == (1 << num_t_subsets) - 1
        
        # Initial validity check
        if not is_valid(current_blocks):
            return current_blocks
            
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            
            # Phase 1: Deletion
            i = 0
            while i < len(current_blocks) and time.time() < time_limit:
                b_idx = current_blocks[i]
                # Check if removing b_idx leaves the set valid
                # This is equivalent to: is there any t-subset covered ONLY by b_idx?
                # We can compute the mask of current_blocks without b_idx.
                # But that's O(N * 35) per check.
                # Better: maintain a count of how many blocks cover each t-subset?
                # That's hard to update dynamically in a simple loop.
                # Alternative: Just compute the union mask of the rest.
                # Since N is ~70, and each mask is an int, OR-ing 70 ints is fast.
                
                rest_mask = 0
                for j, other_b in enumerate(current_blocks):
                    if j != i:
                        rest_mask |= block_masks[other_b]
                        if rest_mask == (1 << num_t_subsets) - 1:
                            # If we've already covered everything with the rest, 
                            # we can break early? 
                            # Actually, if rest_mask is full, it's valid.
                            # But we need to know if it's FULL.
                            pass
                
                # Optimization: 
                # A block is redundant if its "unique" contribution is empty.
                # Unique contribution of b_idx = block_masks[b_idx] & ~ (OR of all other masks)
                # Let's compute OR of others.
                others_mask = 0
                for j, other_b in enumerate(current_blocks):
                    if j != i:
                        others_mask |= block_masks[other_b]
                        
                if (block_masks[b_idx] & ~others_mask) == 0:
                    # Redundant
                    current_blocks.pop(i)
                    changed = True
                    # Do not increment i, check the next block at the same index
                else:
                    i += 1
                    
            if time.time() > time_limit:
                break
                
            # Phase 2: Exchange
            # Try to swap a block with another block to potentially enable deletions later
            # or just to improve the "shape" of the covering.
            # A swap is valid if the new set is still a covering.
            # Heuristic: For each block b_i, try to replace it with b_j 
            # such that b_j covers the "critical" triples of b_i (triples only covered by b_i)
            # and b_j adds new coverage (or at least doesn't reduce it).
            
            # To make this efficient, we can focus on blocks that are "almost" redundant 
            # or have low "unique coverage".
            
            for i in range(len(current_blocks)):
                if time.time() > time_limit:
                    break
                    
                b_i = current_blocks[i]
                
                # Find unique triples of b_i
                # Unique = block_masks[b_i] & ~(OR of others)
                others_mask = 0
                for j, other_b in enumerate(current_blocks):
                    if j != i:
                        others_mask |= block_masks[other_b]
                            
                unique_mask = block_masks[b_i] & ~others_mask
                unique_count = unique_mask.bit_count()
                
                # If unique_count is high, swapping is hard (need a block covering all those unique triples).
                # If unique_count is 0, it would have been deleted in Phase 1.
                
                # We want to find a block b_j NOT in current_blocks 
                # such that b_j covers unique_mask.
                # And ideally, b_j covers many other things too.
                
                # Candidate selection:
                # Instead of scanning all 346k blocks, we can look at blocks that cover 
                # one of the unique triples.
                # Let's pick one unique triple index.
                if unique_mask == 0:
                    continue
                    
                # Get one unique triple index
                # Find the lowest set bit
                lsb = unique_mask & -unique_mask
                unique_idx = lsb.bit_length() - 1
                
                # Candidates are blocks in t_to_blocks[unique_idx]
                candidates = t_to_blocks[unique_idx]
                
                best_swap_idx = -1
                best_swap_gain = -1
                
                # Current total mask
                total_mask = 0
                for b in current_blocks:
                    total_mask |= block_masks[b]
                # total_mask should be full
                
                for b_j in candidates:
                    if b_j in current_blocks:
                        continue
                        
                    # Check if b_j covers ALL unique_mask
                    if (block_masks[b_j] & unique_mask) != unique_mask:
                        continue
                        
                    # If it covers unique_mask, the swap is valid 
                    # (because the unique parts are covered by b_j, and the rest were covered by others).
                    
                    # Calculate "gain" of the swap:
                    # New total mask = total_mask | block_masks[b_j]
                    # Since total_mask is full, new total is also full.
                    # So validity is guaranteed if unique_mask is covered.
                    
                    # Heuristic for quality:
                    # We want a swap that might allow future deletions.
                    # A block that covers many triples already covered by others is "wasteful" 
                    # but might be "compatible" with others.
                    # A better heuristic: 
                    # Count how many triples of b_j are covered by OTHERS (excluding b_i).
                    # This is "redundancy" of b_j with respect to the rest.
                    # High redundancy means b_j is "cheap" to keep, or rather, 
                    # it means b_j overlaps a lot with the existing structure.
                    
                    # Actually, let's look at the number of NEW triples b_j adds 
                    # to the set {others}.
                    # others_mask is the mask of all blocks except b_i.
                    # new_items = block_masks[b_j] & ~others_mask
                    # If new_items is large, b_j is "strong".
                    # If new_items is small, b_j is "redundant" with the rest.
                    
                    # We prefer swaps where b_j is "compatible", i.e., it doesn't introduce 
                    # too many new "fragile" points.
                    # However, standard local search for set cover often tries to minimize 
                    # the "unique load" of the new block.
                    
                    # Let's try a different metric:
                    # Score = - (number of unique triples of b_j w.r.t others)
                    # We want b_j to have few unique triples, so it's easier to delete later?
                    # Or we want b_j to cover many triples, so it's a "good" block.
                    
                    # Let's just pick the first valid swap to add diversity, 
                    # or the one with max overlap with others.
                    
                    overlap = (block_masks[b_j] & others_mask).bit_count()
                    if overlap > best_swap_gain:
                        best_swap_gain = overlap
                        best_swap_idx = b_j
                        
                if best_swap_idx != -1:
                    # Perform swap
                    current_blocks[i] = best_swap_idx
                    changed = True
                    # Break to restart phases, as the structure has changed
                    break
                    
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy restarts
    seeds = [42, 123, 999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        greedy_res = greedy_construct(seed)
        if greedy_res is None:
            continue
            
        # Local search with a time limit per instance
        # Reserve time for local search
        time_for_ls = min(10.0, max(1.0, (deadline - time.time()) / 5.0))
        
        processed_res = local_search(greedy_res, time.time() + time_for_ls)
        
        if len(processed_res) < best_len:
            best_len = len(processed_res)
            best_blocks = processed_res
            
        if best_len <= 69:
            break

    # If we have time, do more random restarts
    if time.time() < deadline:
        rng = random.Random(12345)
        while time.time() < deadline - 5.0:
            seed = rng.randint(0, 1000000)
            greedy_res = greedy_construct(seed)
            if greedy_res is None:
                continue
                
            # Shorter local search for later iterations
            time_for_ls = min(5.0, max(0.5, (deadline - time.time()) / 5.0))
            
            processed_res = local_search(greedy_res, time.time() + time_for_ls)
            
            if len(processed_res) < best_len:
                best_len = len(processed_res)
                best_blocks = processed_res
                
            if best_len <= 69:
                break

    if best_blocks is None:
        # Fallback
        return []
        
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    for b in final_blocks:
        b.sort()
        
    return final_blocks
