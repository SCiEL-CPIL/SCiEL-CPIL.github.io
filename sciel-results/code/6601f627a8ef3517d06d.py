import itertools
import time
import random
import sys

def construct(v, k, t):
    # Safety margin: 150s limit, aim for 115s to be safe
    deadline = time.time() + 115
    
    # If parameters are not the specific case, use a generic fallback (though task is specific)
    if v != 19 or k != 7 or t != 4:
        # Generic greedy fallback for other parameters if needed
        all_t_subs = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_blocks = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_blocks)
        for b in all_k_blocks:
            if not all_t_subs:
                break
            covered = set(itertools.combinations(b, t))
            if all_t_subs & covered:
                blocks.append(list(b))
                all_t_subs -= covered
        return blocks

    # Specific optimization for v=19, k=7, t=4
    rng = random.Random(42)
    
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Precompute masks for all k-blocks
    # mask[i] is a bitmask of t-subsets covered by block i
    block_masks = [0] * n_k_blocks
    # Precompute list of t-subset indices for each block for faster iteration later
    block_t_indices = [[] for _ in range(n_k_blocks)]
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices

    # 2. Greedy Initialization
    uncovered_mask = (1 << n_t_subs) - 1
    selected_blocks = []
    selected_set = set()
    counts = [0] * n_t_subs
    
    def add_block(idx):
        selected_blocks.append(idx)
        selected_set.add(idx)
        for sub_idx in block_t_indices[idx]:
            counts[sub_idx] += 1
            
    def remove_block(idx):
        selected_set.remove(idx)
        # Remove from list efficiently (swap with last)
        pos = selected_blocks.index(idx)
        selected_blocks[pos] = selected_blocks[-1]
        selected_blocks.pop()
        for sub_idx in block_t_indices[idx]:
            counts[sub_idx] -= 1
            
    def is_redundant(idx):
        for sub_idx in block_t_indices[idx]:
            if counts[sub_idx] == 1:
                return False
        return True

    # Greedy Loop
    while uncovered_mask != 0:
        if time.time() > deadline - 60: # Leave time for local search
            break
            
        best_idx = -1
        best_gain = -1
        
        # Optimization: We can skip blocks that are already selected
        # Iterate all blocks to find max gain
        for i in range(n_k_blocks):
            if i in selected_set:
                continue
            gain = (block_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            break
            
        add_block(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # 3. Initial Redundancy Removal
    changed = True
    while changed:
        changed = False
        for i in range(len(selected_blocks) - 1, -1, -1):
            if is_redundant(selected_blocks[i]):
                remove_block(selected_blocks[i])
                changed = True

    best_count = len(selected_blocks)
    best_blocks = selected_blocks[:]
    
    # 4. Local Search: Targeted Swaps (Combinatorial Auction Logic)
    # We try to swap a block with another that covers its unique subsets,
    # hoping to enable further redundancy removal.
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 5:
        if len(selected_blocks) < best_count:
            best_count = len(selected_blocks)
            best_blocks = selected_blocks[:]
            
        # Phase 1: Prune redundant blocks
        removed_any = False
        # Shuffle indices to randomize pruning order
        indices = list(range(len(selected_blocks)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            idx = selected_blocks[i]
            if is_redundant(idx):
                remove_block(idx)
                removed_any = True
                if len(selected_blocks) < best_count:
                    best_count = len(selected_blocks)
                    best_blocks = selected_blocks[:]
                # Restart pruning loop as counts changed
                break
                
        if removed_any:
            no_improve_count = 0
            continue
            
        # Phase 2: Targeted Swap
        if not selected_blocks:
            break
            
        found_swap = False
        # Pick a random block to potentially replace
        # We prioritize blocks with many unique subsets? Or just random?
        # Random is fine, but we can bias towards "critical" blocks.
        # Let's just pick random for now, as checking all is expensive.
        i = rng.randrange(len(selected_blocks))
        idx = selected_blocks[i]
        
        # Find unique subsets covered by idx
        unique_bits = []
        for sub_idx in block_t_indices[idx]:
            if counts[sub_idx] == 1:
                unique_bits.append(sub_idx)
                
        if not unique_bits:
            # It was redundant, but pruning missed it? (Shouldn't happen if pruning is correct)
            remove_block(idx)
            no_improve_count = 0
            continue
            
        # If a block has too many unique subsets, it's hard to replace.
        # If it has 0, it's redundant.
        # We look for a block that covers ALL unique_bits.
        
        # Optimization: If unique_bits is large, the intersection of candidate lists will be small.
        # We can precompute which blocks cover which subsets? 
        # That's 3876 subsets * 77520 blocks. Too big for memory if stored as lists of lists.
        # But we can check candidates by intersecting sets.
        
        # Let's try to find a candidate.
        # We iterate through all blocks NOT in selected_set.
        # To speed this up, we can sort unique_bits by the number of blocks covering them?
        # We don't have that precomputed.
        
        # Alternative: Just scan. 77k blocks is fast enough in Python if done a few times.
        # But we do this in a loop.
        
        # Let's try a few random candidates first? No, we need one that covers unique_bits.
        # Let's scan.
        
        # Early exit if unique_bits is too large?
        # If len(unique_bits) > 10, it's very specific.
        
        candidate_idx = -1
        # Scan for a block that covers all unique_bits
        # We can speed this up by checking the first unique bit's mask against others?
        # block_masks[j] & unique_mask == unique_mask
        
        unique_mask = 0
        for b in unique_bits:
            unique_mask |= (1 << b)
            
        for j in range(n_k_blocks):
            if j in selected_set:
                continue
            if (block_masks[j] & unique_mask) == unique_mask:
                candidate_idx = j
                break
                
        if candidate_idx != -1:
            # Perform swap
            remove_block(idx)
            add_block(candidate_idx)
            found_swap = True
            no_improve_count = 0
            
            # Aggressive pruning after swap
            changed = True
            while changed:
                changed = False
                for k in range(len(selected_blocks) - 1, -1, -1):
                    if is_redundant(selected_blocks[k]):
                        remove_block(selected_blocks[k])
                        changed = True
        else:
            no_improve_count += 1
            
        if not found_swap:
            # Perturbation if stuck
            if no_improve_count > max_no_improve:
                # Remove a few random blocks and re-greedy
                n_remove = min(3, len(selected_blocks))
                if n_remove > 0:
                    to_remove = rng.sample(selected_blocks, n_remove)
                    for idx in to_remove:
                        remove_block(idx)
                        
                    # Re-greedy to cover gaps
                    uncov_mask = 0
                    for s in range(n_t_subs):
                        if counts[s] == 0:
                            uncov_mask |= (1 << s)
                            
                    while uncov_mask:
                        if time.time() > deadline - 5:
                            break
                        best_idx = -1
                        best_gain = 0
                        for i in range(n_k_blocks):
                            if i in selected_set:
                                continue
                            gain = (block_masks[i] & uncov_mask).bit_count()
                            if gain > best_gain:
                                best_gain = gain
                                best_idx = i
                        if best_idx == -1:
                            break
                        add_block(best_idx)
                        uncov_mask &= ~block_masks[best_idx]
                        
                    no_improve_count = 0
                    max_no_improve = min(300, max_no_improve * 1.5)
                else:
                    break # Can't remove more

    # Final cleanup
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected_blocks) - 1, -1, -1):
            if is_redundant(selected_blocks[i]):
                remove_block(selected_blocks[i])
                changed = True

    # Return best found
    final_blocks = best_blocks if len(best_blocks) < len(selected_blocks) else selected_blocks
    
    # Convert indices to lists of points
    blocks = [list(all_k_blocks[i]) for i in final_blocks]
    
    return blocks

if __name__ == "__main__":
    # Quick test
    blocks = construct(19, 7, 4)
    print(f"Found covering with {len(blocks)} blocks")
