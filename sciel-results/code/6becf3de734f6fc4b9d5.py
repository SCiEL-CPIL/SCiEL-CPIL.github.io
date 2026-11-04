import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Bitsets for blocks
    block_bitsets = [0] * num_blocks
    # List of blocks containing each triple
    blocks_containing_triple = [[] for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        bitset = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            bitset |= (1 << idx)
            blocks_containing_triple[idx].append(i)
        block_bitsets[i] = bitset
        
    # 2. Greedy Construction
    uncovered = (1 << num_triples) - 1
    selected = set()
    selected_blocks = [] # list of indices
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(num_blocks):
            if i in selected:
                continue
            gain = (block_bitsets[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        selected.add(best_idx)
        selected_blocks.append(best_idx)
        uncovered &= ~block_bitsets[best_idx]
        
        if time.time() > deadline - 20:
            break
            
    # 3. Local Search
    # Helper to compute counts
    def get_counts(selected_idx_list):
        counts = [0] * num_triples
        for idx in selected_idx_list:
            # Iterate bits in bitset? Or iterate triples in block?
            # Iterating triples in block is C(k,t) = 35 ops.
            # Iterating bits in bitset requires finding set bits.
            # 35 is small.
            b = all_blocks[idx]
            for tri in itertools.combinations(b, t):
                counts[triple_to_idx[tri]] += 1
        return counts

    # Initial counts
    counts = get_counts(selected_blocks)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Pass 1: Remove Redundant
        # Check all selected blocks
        to_remove = []
        for idx in selected_blocks:
            b = all_blocks[idx]
            redundant = True
            for tri in itertools.combinations(b, t):
                if counts[triple_to_idx[tri]] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(idx)
                
        if to_remove:
            for idx in to_remove:
                selected_blocks.remove(idx)
                selected.remove(idx)
                b = all_blocks[idx]
                for tri in itertools.combinations(b, t):
                    counts[triple_to_idx[tri]] -= 1
            improved = True
            continue # Restart pass to catch new redundancies
            
        # Pass 2: Swap
        # Try to swap each block
        # To avoid O(N^2) complexity in the inner loop, we limit swaps per pass
        # or just iterate through all.
        
        # We need a list of selected blocks to iterate
        # If we modify selected_blocks during iteration, it's tricky.
        # Let's iterate over a copy or indices.
        
        swap_done = False
        for idx in list(selected_blocks):
            if time.time() > deadline - 10:
                break
                
            # Temporarily remove idx
            b = all_blocks[idx]
            # Find critical triples
            critical = []
            for tri in itertools.combinations(b, t):
                tri_idx = triple_to_idx[tri]
                if counts[tri_idx] == 1:
                    critical.append(tri_idx)
            
            if not critical:
                # This shouldn't happen if Pass 1 is clean, but just in case
                selected_blocks.remove(idx)
                selected.remove(idx)
                for tri in itertools.combinations(b, t):
                    counts[triple_to_idx[tri]] -= 1
                improved = True
                swap_done = True
                break
            
            # Find a block covering all critical triples
            # Pick the critical triple with fewest candidates
            # candidates for critical[0], critical[1], etc.
            # Intersection of candidates.
            # Since critical size is small (<=35), we can just check candidates of the first one?
            # No, must cover ALL.
            # Strategy: Pick the critical triple with the smallest `blocks_containing_triple` list.
            min_len = float('inf')
            min_tri_idx = -1
            for tri_idx in critical:
                if len(blocks_containing_triple[tri_idx]) < min_len:
                    min_len = len(blocks_containing_triple[tri_idx])
                    min_tri_idx = tri_idx
            
            candidates = blocks_containing_triple[min_tri_idx]
            
            found_swap = False
            for cand_idx in candidates:
                if cand_idx in selected:
                    continue
                
                # Check if cand covers all critical triples
                cand_bitset = block_bitsets[cand_idx]
                covers_all = True
                for tri_idx in critical:
                    if not (cand_bitset & (1 << tri_idx)):
                        covers_all = False
                        break
                
                if covers_all:
                    # Perform Swap
                    # Remove idx
                    selected_blocks.remove(idx)
                    selected.remove(idx)
                    for tri in itertools.combinations(b, t):
                        counts[triple_to_idx[tri]] -= 1
                    
                    # Add cand_idx
                    selected_blocks.append(cand_idx)
                    selected.add(cand_idx)
                    cand_b = all_blocks[cand_idx]
                    for tri in itertools.combinations(cand_b, t):
                        counts[triple_to_idx[tri]] += 1
                        
                    improved = True
                    found_swap = True
                    swap_done = True
                    break
                    
            if swap_done:
                break # Restart pass after a swap
                
    # Final cleanup: remove any remaining redundancies
    # (The loop above should handle it, but a final pass is safe)
    final_blocks = [all_blocks[idx] for idx in selected_blocks]
    
    return final_blocks
