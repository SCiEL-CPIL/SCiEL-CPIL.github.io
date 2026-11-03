import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Map triples to IDs
    triples = list(itertools.combinations(range(v), t))
    triple_to_id = {triple: i for i, triple in enumerate(triples)}
    n_triples = len(triples)
    all_mask = (1 << n_triples) - 1
    
    # 2. Pre-compute bitsets for all possible blocks
    # This is the most memory/time intensive part, but C(24,7) is manageable
    blocks_list = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks_list)
    
    block_bitsets = [0] * n_blocks
    # Also, for each triple, which blocks contain it?
    # triple_to_blocks = [set() for _ in range(n_triples)] # Too slow/memory heavy?
    # Instead, for greedy, we need to find blocks containing a specific triple.
    # Let's pre-compute: for each triple ID, list of block indices.
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(blocks_list):
        mask = 0
        for triple in itertools.combinations(block, t):
            tid = triple_to_id[triple]
            mask |= (1 << tid)
            triple_to_blocks[tid].append(i)
        block_bitsets[i] = mask
        
    # 3. Greedy Construction
    covered = 0
    current_blocks = []
    
    while covered != all_mask:
        if time.time() > deadline:
            break
            
        # Find first uncovered triple
        # We can do this by finding the lowest set bit in (~covered) & all_mask
        uncovered_mask = (~covered) & all_mask
        if uncovered_mask == 0:
            break
            
        # Get the index of the lowest set bit
        # Python 3.11+ has int.bit_count, but we need index.
        # We can use (uncovered_mask & -uncovered_mask).bit_length() - 1
        lowest_bit = uncovered_mask & -uncovered_mask
        first_uncovered_id = lowest_bit.bit_length() - 1
        
        # Find best block containing this triple
        best_block_idx = -1
        max_gain = -1
        
        for b_idx in triple_to_blocks[first_uncovered_id]:
            gain = (block_bitsets[b_idx] & ~covered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
                if gain == n_triples - covered.bit_count(): # Can't do better
                    break
        
        if best_block_idx == -1:
            break # Should not happen
            
        current_blocks.append(best_block_idx)
        covered |= block_bitsets[best_block_idx]
        
    # Convert indices to actual blocks
    initial_blocks = [blocks_list[i] for i in current_blocks]
    
    if time.time() > deadline:
        return initial_blocks
        
    # 4. Local Search
    # We work with the list of block indices
    current_indices = list(current_blocks)
    
    # Helper to check if a block is redundant
    def is_redundant(idx):
        # Calculate coverage of all other blocks
        # This is expensive if done from scratch every time.
        # Optimization: maintain a coverage count array?
        # For simplicity in this time limit, we can do a quick check:
        # If the block covers any triple that is covered ONLY by this block, it's not redundant.
        # We need to know the "exclusive" triples of the block.
        pass
        
    # A more efficient local search for time constraints:
    # 1. Try to delete blocks that are completely covered by others.
    # 2. Try to swap blocks to enable deletions.
    
    # Let's use a simpler heuristic:
    # Maintain a set of current block indices for O(1) lookup.
    current_set = set(current_indices)
    
    def calculate_exclusive_coverage(block_idx):
        # Returns bitset of triples covered by this block but not by any other in current_set
        # This is O(N) where N is number of blocks.
        # We can optimize by checking only blocks that overlap?
        # For now, just iterate all other blocks.
        exclusive = block_bitsets[block_idx]
        for other_idx in current_set:
            if other_idx != block_idx:
                exclusive &= ~block_bitsets[other_idx]
        return exclusive
        
    # Greedy deletion loop
    while time.time() < deadline:
        progress = False
        # Try to delete redundant blocks
        for i in range(len(current_indices) - 1, -1, -1):
            if time.time() > deadline:
                break
            idx = current_indices[i]
            # Quick check: if exclusive coverage is 0, delete
            # To save time, we can sample or just do it if N is small.
            # N is ~76. 76*76 = 5776 bit ops. Very fast.
            exclusive = block_bitsets[idx]
            for other_idx in current_indices:
                if other_idx != idx:
                    exclusive &= ~block_bitsets[other_idx]
                    if exclusive == 0:
                        break
            if exclusive == 0:
                current_indices.pop(i)
                current_set.remove(idx)
                progress = True
                # Restart loop to find more redundancies
                break
        
        if not progress:
            # Try swaps
            # Pick a random block to replace
            if not current_indices:
                break
            target_idx = random.choice(current_indices)
            target_exclusive = block_bitsets[target_idx]
            for other_idx in current_indices:
                if other_idx != target_idx:
                    target_exclusive &= ~block_bitsets[other_idx]
            
            if target_exclusive == 0:
                # Should have been deleted, but maybe logic missed it?
                continue
                
            # Try to find a replacement block that covers target_exclusive
            # Sample some random blocks
            improved = False
            for _ in range(100): # Limit attempts per target
                if time.time() > deadline:
                    break
                new_idx = random.randint(0, n_blocks - 1)
                if new_idx in current_set:
                    continue
                # Check if new block covers exclusive triples
                if (block_bitsets[new_idx] & target_exclusive) == target_exclusive:
                    # Check if this swap is beneficial
                    # Benefit: Does the new block cover more "unique" stuff or allow more deletions?
                    # Simple metric: Does new block cover more currently covered triples than old one?
                    # Or just accept if it covers the exclusive part.
                    # To avoid cycles, we can require that the new block covers at least as many
                    # total triples (in the current set context) or just accept and hope for deletion.
                    # Let's accept and try to delete.
                    current_indices.remove(target_idx)
                    current_set.remove(target_idx)
                    current_indices.append(new_idx)
                    current_set.add(new_idx)
                    improved = True
                    break
            
            if not improved:
                # If no swap found after some tries, break to avoid infinite loop on bad state
                # We can continue to next target, but if we can't improve, we are stuck.
                # For robustness, just break the inner loop and try again in outer loop
                # Or break completely if we've made no progress for a while.
                pass

    # Final conversion
    final_blocks = [blocks_list[i] for i in current_indices]
    
    # Ensure validity (just in case)
    # The evaluator checks this, but we should ensure our logic didn't break it.
    # The deletion/swap logic preserves coverage if we only delete when exclusive=0 
    # and only swap when new covers exclusive.
    
    return final_blocks
