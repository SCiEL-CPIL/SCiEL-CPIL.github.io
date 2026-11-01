import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Precompute coverage mask for each k-subset as a Python int bitset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    # 4. Initialize uncovered bitset (all 1s for n_tsubs bits)
    uncovered_mask = (1 << n_tsubs) - 1
    remaining = n_tsubs

    blocks = []
    used_blocks = set()

    # 5. Greedy selection loop
    while remaining > 0:
        if time.time() > deadline:
            break

        best_j = -1
        best_gain = 0

        for j in range(n_ksubs):
            if j in used_blocks:
                continue
            
            common = uncovered_mask & ksub_masks[j]
            gain = common.bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1 or best_gain == 0:
            break

        blocks.append(list(all_ksubs[best_j]))
        used_blocks.add(best_j)
        
        uncovered_mask &= ~ksub_masks[best_j]
        remaining -= best_gain

    # 6. Post-processing: Local Search with Swaps
    if time.time() < deadline - 5.0 and len(blocks) > 0:
        # Map selected blocks back to indices
        selected_indices = []
        for b in blocks:
            for idx, ks in enumerate(all_ksubs):
                if tuple(b) == ks:
                    selected_indices.append(idx)
                    break
        
        active_indices = set(selected_indices)
        
        # Helper to get unique coverage of a block in the active set
        def get_unique_mask(block_idx, active_set):
            other_mask = 0
            for idx in active_set:
                if idx != block_idx:
                    other_mask |= ksub_masks[idx]
            return ksub_masks[block_idx] & ~other_mask

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Calculate unique masks for all active blocks
            unique_masks = {}
            for idx in active_indices:
                unique_masks[idx] = get_unique_mask(idx, active_indices)
            
            # Find redundant blocks (unique mask is 0)
            to_remove = [idx for idx, um in unique_masks.items() if um == 0]
            if to_remove:
                for idx in to_remove:
                    active_indices.remove(idx)
                improved = True
                continue
            
            # Try to find a swap that enables a removal
            # Pick a block with small unique coverage to replace
            # Sort by unique coverage size to prioritize "weak" blocks
            sorted_indices = sorted(active_indices, key=lambda x: unique_masks[x].bit_count())
            
            for target_idx in sorted_indices:
                if time.time() > deadline - 2.0:
                    break
                
                target_unique = unique_masks[target_idx]
                if target_unique == 0:
                    continue
                
                # Find a replacement block that covers target_unique
                replacement_idx = -1
                # Iterate through all blocks not in active set
                # Optimization: We can break early if we find one
                for j in range(n_ksubs):
                    if j in active_indices:
                        continue
                    if (ksub_masks[j] & target_unique) == target_unique:
                        replacement_idx = j
                        break
                
                if replacement_idx != -1:
                    # Perform swap
                    active_indices.remove(target_idx)
                    active_indices.add(replacement_idx)
                    
                    # Check if this swap allows any removal
                    new_unique_masks = {}
                    for idx in active_indices:
                        new_unique_masks[idx] = get_unique_mask(idx, active_indices)
                    
                    new_redundants = [idx for idx, um in new_unique_masks.items() if um == 0]
                    if new_redundants:
                        for idx in new_redundants:
                            active_indices.remove(idx)
                        improved = True
                        break # Restart the main loop with the new set
                    else:
                        # If no immediate improvement, revert the swap to keep the set valid and potentially try another target?
                        # Or keep it? If we keep it, we might be stuck in a cycle.
                        # Let's revert if no count reduction, to ensure we don't waste time on non-improving moves.
                        active_indices.remove(replacement_idx)
                        active_indices.add(target_idx)
                        # Mark as improved? No, we didn't improve.
                        # We should continue looking for other targets or break if we've tried all.
                        # To avoid infinite loops, we only accept swaps that lead to immediate count reduction.
                        # This is a conservative heuristic.
                        pass

        # Final check and reconstruction
        if active_indices:
            final_blocks = [list(all_ksubs[idx]) for idx in active_indices]
            final_mask = 0
            for idx in active_indices:
                final_mask |= ksub_masks[idx]
            if final_mask == (1 << n_tsubs) - 1:
                blocks = final_blocks

    return blocks
