"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    """
    Greedy set cover with bitsets, followed by local search optimization.
    """
    deadline = time.time() + 115.0  # Safety margin for 120s limit (150s kill)
    rng = random.Random(42)

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Initialize uncovered bitset (all 1s)
    uncovered_mask = (1 << n_tsubs) - 1
    remaining = n_tsubs

    # 3. Generate all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 4. Precompute coverage mask for each k-subset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    # 5. Greedy selection loop
    blocks_idx = []  # indices into all_ksubs
    used_blocks = set()

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

        blocks_idx.append(best_j)
        used_blocks.add(best_j)
        uncovered_mask &= ~ksub_masks[best_j]
        remaining -= best_gain

    # If greedy didn't finish, we can't guarantee validity. 
    # For v=19, it should finish quickly.
    if remaining > 0:
        # Emergency: try to cover remaining with brute force (shouldn't happen)
        # Just return what we have; if invalid, score is 0.
        pass

    # 6. Local search: try to reduce block count
    #    For each block, check if removing it leaves all t-subsets covered.
    #    If not, try to find a replacement block that covers the "lost" t-subsets
    #    and allows more blocks to be removed.
    
    n_blocks = len(blocks_idx)
    
    # Build coverage count: how many blocks cover each t-subset
    # Using bitsets: for each block, OR its mask. Then check if all bits are set.
    # More efficient: maintain a "coverage" array using bitsets.
    
    # Let's use a different approach: 
    # coverage_mask = OR of all block masks. If coverage_mask == full_mask, valid.
    # To check if removing block i still covers everything:
    #   We need OR of all other block masks to still be full_mask.
    # This is expensive to recompute each time. Instead, use a count array.
    
    # For v=19, n_tsubs = 3876. We can use a list of ints for coverage counts.
    # But with ~150 blocks and 3876 t-subsets, a simple approach works.
    
    # Optimization: precompute for each block which t-subsets it covers (as a set of indices)
    # Then use a count array.
    
    block_tsub_sets = []
    for idx in blocks_idx:
        mask = ksub_masks[idx]
        # Extract set bits
        tsubs_in_block = set()
        m = mask
        while m:
            low = m & (-m)
            bit_idx = low.bit_length() - 1
            tsubs_in_block.add(bit_idx)
            m ^= low
        block_tsub_sets.append(tsubs_in_block)
    
    # Coverage count array
    cov_count = [0] * n_tsubs
    for tsub_set in block_tsub_sets:
        for ti in tsub_set:
            cov_count[ti] += 1
    
    # Local search: try removing each block if all t-subsets still covered
    # If removal fails, try swapping with a different block
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_blocks = len(blocks_idx)
        
        # Shuffle order to avoid bias
        order = list(range(n_blocks))
        rng.shuffle(order)
        
        for bi in order:
            if time.time() > deadline:
                break
            
            # Try removing block bi
            # Check if all t-subsets still have cov_count >= 1 after removal
            lost_tsubs = []
            for ti in block_tsub_sets[bi]:
                cov_count[ti] -= 1
                if cov_count[ti] == 0:
                    lost_tsubs.append(ti)
            
            if not lost_tsubs:
                # Safe to remove
                blocks_idx.pop(bi)
                block_tsub_sets.pop(bi)
                improved = True
                break  # Restart scan since indices changed
            else:
                # Restore coverage counts
                for ti in lost_tsubs:
                    cov_count[ti] += 1
                # Also need to restore for t-subsets that had cov_count > 1
                # Wait, we only decremented for all in block_tsub_sets[bi],
                # but only tracked those that hit 0. We need to restore ALL.
                # Let's redo: decrement all, check, then increment all back if not removing.
                # Actually, let's restructure.
                
                # Revert all decrements
                for ti in block_tsub_sets[bi]:
                    cov_count[ti] += 1
                
                # Try to find a replacement block that covers all lost_tsubs
                # and has high overlap with other blocks' coverage
                # For simplicity, try to find a block that covers all lost_tsubs
                lost_set = set(lost_tsubs)
                
                best_replace = -1
                best_overlap = -1
                
                for j in range(n_ksubs):
                    if j in used_blocks:
                        continue
                    # Check if ksub_masks[j] covers all lost_tsubs
                    mask_j = ksub_masks[j]
                    # Check if (lost_mask & ~mask_j) == 0
                    lost_mask = 0
                    for ti in lost_set:
                        lost_mask |= (1 << ti)
                    if (lost_mask & ~mask_j) == 0:
                        # This block covers all lost t-subsets
                        # Score by how much it overlaps with existing coverage
                        # (i.e., how many of its t-subsets are already covered by other blocks)
                        overlap = 0
                        mj = mask_j
                        while mj:
                            low = mj & (-mj)
                            bit_idx = low.bit_length() - 1
                            if cov_count[bit_idx] >= 2:  # Covered by at least 2 blocks (including bi)
                                overlap += 1
                            mj ^= low
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_replace = j
                
                if best_replace != -1:
                    # Replace block bi with best_replace
                    # Remove old block
                    for ti in block_tsub_sets[bi]:
                        cov_count[ti] -= 1
                    blocks_idx[bi] = best_replace
                    used_blocks.discard(blocks_idx[bi] if False else None)  # clean up
                    # Actually, update used_blocks
                    old_idx = blocks_idx[bi]
                    # Wait, we already popped? No, we're replacing in place.
                    # Let me redo this more carefully.
                    
                    # Revert: we haven't actually removed bi yet in the list
                    # Let's just do a clean swap:
                    # 1. Remove bi from blocks_idx and update cov_count
                    # 2. Add best_replace to blocks_idx and update cov_count
                    
                    # Remove bi
                    for ti in block_tsub_sets[bi]:
                        cov_count[ti] -= 1
                    blocks_idx.pop(bi)
                    block_tsub_sets.pop(bi)
                    used_blocks.discard(old_idx)
                    
                    # Add best_replace
                    blocks_idx.append(best_replace)
                    used_blocks.add(best_replace)
                    
                    # Compute new tsub set for best_replace
                    mask_new = ksub_masks[best_replace]
                    new_tsubs = set()
                    m = mask_new
                    while m:
                        low = m & (-m)
                        bit_idx = low.bit_length() - 1
                        new_tsubs.add(bit_idx)
                        m ^= low
                    block_tsub_sets.append(new_tsubs)
                    
                    # Update cov_count
                    for ti in new_tsubs:
                        cov_count[ti] += 1
                    
                    improved = True
                    break  # Restart scan
    
    # 7. Convert to final block list
    blocks = [list(all_ksubs[idx]) for idx in blocks_idx]
    
    # 8. Verify validity (safety check)
    # Check that all t-subsets are covered
    final_mask = 0
    for idx in blocks_idx:
        final_mask |= ksub_masks[idx]
    
    if final_mask != (1 << n_tsubs) - 1:
        # Invalid! Fallback: try to add blocks to cover missing t-subsets
        missing_mask = ((1 << n_tsubs) - 1) & ~final_mask
        # Find t-subsets not covered
        m = missing_mask
        while m:
            low = m & (-m)
            ti = low.bit_length() - 1
            tsub = all_tsubs[ti]
            # Find a k-subset containing this t-subset
            for j in range(n_ksubs):
                if j in used_blocks:
                    continue
                if ksub_masks[j] & (1 << ti):
                    blocks.append(list(all_ksubs[j]))
                    used_blocks.add(j)
                    final_mask |= ksub_masks[j]
                    break
            m ^= low
    
    return blocks
# EVOLVE-BLOCK-END
