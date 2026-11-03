import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Pre-computation
    # Generate all t-subsets and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Generate all k-subsets and compute their bitmask of covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in all_blocks:
        mask = 0
        for triple in itertools.combinations(block, t):
            mask |= 1 << triple_to_idx[triple]
        block_masks.append(mask)
    
    num_blocks = len(all_blocks)
    
    # 2. Greedy Construction
    def greedy_construct(uncovered_mask):
        blocks = []
        current_uncovered = uncovered_mask
        # To speed up, we can keep a list of candidate blocks, but full scan is okay for this size
        # We need to avoid re-picking the same block. We can just check if it's in the list, 
        # or better, since blocks are distinct, we can track used indices.
        used = set()
        
        while current_uncovered:
            best_gain = 0
            best_block_idx = -1
            # Optimization: If we have many blocks, scanning all is O(N).
            # 346k blocks. 
            for i in range(num_blocks):
                if i in used:
                    continue
                gain = current_uncovered & block_masks[i]
                if gain:
                    count = gain.bit_count()
                    if count > best_gain:
                        best_gain = count
                        best_block_idx = i
            if best_block_idx == -1:
                break # Should not happen if valid
            blocks.append(best_block_idx)
            used.add(best_block_idx)
            current_uncovered &= ~block_masks[best_block_idx]
            
        return blocks

    # Initial Greedy
    initial_uncovered = (1 << num_triples) - 1
    current_blocks = greedy_construct(initial_uncovered)
    
    # Helper to calculate coverage counts
    def get_coverage_counts(blocks):
        counts = [0] * num_triples
        for b_idx in blocks:
            mask = block_masks[b_idx]
            # Iterate through set bits
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    # Local Search: Removal and Perturbation
    def try_remove_blocks(blocks):
        removed_any = True
        while removed_any:
            removed_any = False
            counts = get_coverage_counts(blocks)
            for i in range(len(blocks)):
                b_idx = blocks[i]
                mask = block_masks[b_idx]
                # Check if all triples in this block have count > 1
                can_remove = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        can_remove = False
                        break
                    m ^= lsb
                
                if can_remove:
                    # Remove block
                    blocks.pop(i)
                    removed_any = True
                    break # Restart loop to recalculate counts
        return blocks

    current_blocks = try_remove_blocks(current_blocks)
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # Iterated Local Search
    while time.time() < deadline:
        # Perturb: Pick a random block, remove it, and re-greedy fill
        if not current_blocks:
            break
            
        # Pick a random block to remove
        rand_idx = random.randint(0, len(current_blocks) - 1)
        removed_block = current_blocks[rand_idx]
        
        # Remove it
        current_blocks.pop(rand_idx)
        
        # Calculate new uncovered mask
        # We need to know what was covered ONLY by the removed block
        # Easier: Recalculate uncovered from scratch? No, expensive.
        # Let's just re-greedy from the current state.
        # We need the current uncovered mask.
        # We can maintain it, but for simplicity in perturbation, let's just re-greedy the whole thing?
        # No, that's too slow.
        
        # Let's maintain `current_uncovered` properly in the main loop?
        # It's hard to maintain exactly with removals.
        
        # Alternative Perturbation: 
        # 1. Remove a random block.
        # 2. Identify triples that are now uncovered.
        # 3. Greedy add blocks to cover them.
        
        # To do this, we need counts.
        counts = get_coverage_counts(current_blocks)
        
        # Find uncovered triples
        new_uncovered = 0
        for i in range(num_triples):
            if counts[i] == 0:
                new_uncovered |= 1 << i
                
        # Greedy fill for new_uncovered
        # We can reuse the greedy logic but only for the uncovered part
        # And we should prefer blocks already in the set? No, they are in the set.
        # We need to find NEW blocks.
        
        # Let's just run a mini-greedy
        while new_uncovered:
            best_gain = 0
            best_block_idx = -1
            # We can search all blocks, but exclude those already in current_blocks
            # Or just search all, and if we pick one already there, it's fine (it adds nothing new)
            # But we want to add NEW blocks.
            for i in range(num_blocks):
                if i in current_blocks:
                    continue
                gain = new_uncovered & block_masks[i]
                if gain:
                    count = gain.bit_count()
                    if count > best_gain:
                        best_gain = count
                        best_block_idx = i
            if best_block_idx == -1:
                break
            current_blocks.append(best_block_idx)
            new_uncovered &= ~block_masks[best_block_idx]
            
        # Now try to remove blocks again
        current_blocks = try_remove_blocks(current_blocks)
        
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # Check time
        if time.time() > deadline:
            break

    # Convert indices back to blocks
    result = [list(all_blocks[i]) for i in best_blocks]
    return result
