import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute for faster gain calculation
# For each block, store the list of subset indices it covers
block_subsets = []
for mask in all_masks:
    subs = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bit_idx = lsb.bit_length() - 1
        subs.append(bit_idx)
        mm ^= lsb
    block_subsets.append(subs)

# Precompute overlap counts between all pairs of blocks (number of 4-subsets they share)
# This is expensive but done once. For N_BLOCKS=50388, this is ~2.5 billion pairs which is too slow.
# Instead, we'll compute overlaps on the fly using bit operations or use a different strategy.
# Let's precompute for each block, a list of (other_block_idx, overlap_count) for blocks with high overlap?
# Actually, for packing-first, we need to know how many new subsets a block covers given current set.
# We can approximate this by tracking covered subsets.

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            mm ^= lsb

    def remove_block(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            mm ^= lsb
        # Remove from lists
        current_blocks.pop(i)
        current_masks.pop(i)

    def is_redundant(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                return False
            mm ^= lsb
        return True

    def get_new_coverage(idx):
        """Calculate number of subsets covered by block idx that are not currently covered."""
        m = all_masks[idx]
        mm = m
        new_cov = 0
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 0:
                new_cov += 1
            mm ^= lsb
        return new_cov

    # Phase 1: Packing-First Greedy
    # Strategy: Build a core of blocks with minimal overlap.
    # We prefer blocks that cover many NEW subsets (standard greedy) but with a bias towards
    # "efficient" blocks. In a packing context, we want to avoid blocks that overlap heavily
    # with existing ones.
    # However, since we need to cover ALL subsets eventually, pure packing isn't enough.
    # The idea is: Start with a greedy that maximizes new coverage. 
    # To introduce "packing" flavor, we can penalize overlap.
    # But for simplicity and speed, let's try a modified greedy that prioritizes blocks
    # that are "disjoint" in terms of 4-subsets from recently added blocks?
    # Or simply: Run standard greedy, but if there's a tie, pick the one with least overlap 
    # with the rest of the current set? That's expensive.
    
    # Alternative Packing-First: 
    # 1. Generate a random permutation of all blocks.
    # 2. Iterate through them. If a block covers at least 1 new subset, add it? No, that's just greedy with random order.
    # 3. Better: Maximize (new_coverage - lambda * overlap_with_existing).
    # Since overlap calculation is expensive, let's stick to standard greedy for the core, 
    # but maybe run it multiple times with different random tie-breaks or perturbations early on.
    
    # Let's try a "Greedy with Diversity" approach for the initial core:
    # Keep a list of recently added blocks. When selecting a new block, 
    # among those with high gain, prefer those with low overlap with recent blocks.
    # This is heuristic and might be slow.
    
    # Let's stick to the current structure but enhance the local search, 
    # as the directive suggests investigating improvements within the family.
    # The "Packing-First" directive suggests starting from a high-efficiency structural core.
    # A good proxy for efficiency is: Start with a small set of blocks that cover many unique subsets.
    # Standard greedy does this. 
    # Perhaps the improvement is in the LOCAL SEARCH being more aggressive or using a better neighborhood.
    
    # Let's implement the standard greedy first, then a more powerful local search.
    
    # 1. Greedy Initialization
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        # Optimization: If we have many blocks, this loop is slow.
        # We can break early if best_gain == max possible? No, max possible varies.
        for i in range(N_BLOCKS):
            # Quick check: if block is already fully covered, skip? 
            # (all_masks[i] & uncovered_mask) == 0
            # But we need to find the MAX gain.
            gain = (all_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if best_gain == 35: # Max possible new coverage for a 7-block is C(7,4)=35
                    # Can't get better than this, but maybe another block also has 35?
                    # Doesn't matter which one we pick if gain is max.
                    break
        if best_idx == -1:
            break # Should not happen if uncovered_mask is not 0
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search with Perturbation (Enhanced)
    no_improve_count = 0
    max_no_improve = 50 # Reduced to allow more perturbations
    
    while time.time() < deadline:
        # Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline: break
            if is_redundant(i):
                remove_block(i)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                # After removing, continue checking for more redundancies
                # But we broke the loop in original code. Let's allow multiple removals.
                # Actually, the original code broke. Let's stick to that to be safe, 
                # or restart the scan?
                # Let's restart the scan to catch cascading redundancies.
                break
        
        if removed_any:
            no_improve_count = 0
            continue
            
        # If no redundant block found, try a swap
        if len(current_blocks) == 0: break
        
        # Pick a random block to remove
        i = random.randint(0, len(current_blocks) - 1)
        
        # Tentatively remove
        m_i = current_masks[i]
        uncovered_bits = []
        mm = m_i
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                uncovered_bits.append(bit_idx)
            mm ^= lsb
            
        if not uncovered_bits:
            # This block was redundant (should have been caught above, but maybe counts changed?)
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue
            
        # We need to cover uncovered_bits
        if len(uncovered_bits) > 5: # Increased threshold slightly
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: Add a random block, then try to prune
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                    no_improve_count = 0
                continue
        else:
            # Find a block that covers all uncovered_bits
            # Sort by rarity to speed up intersection
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = blocks_containing_subset[uncovered_bits[0]]
            
            for b in uncovered_bits[1:]:
                if not candidates: break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
                
            if candidates:
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    # Choose candidate that maximizes "new" coverage relative to current set (excluding i)
                    # This is expensive. Let's just pick one that covers the most total subsets?
                    # Or just random.
                    # Let's pick the one with the highest total coverage (popcount of mask) among candidates?
                    # Or the one that overlaps least with current set (excluding i)?
                    # For now, random choice is fast.
                    j = random.choice(valid_candidates)
                    remove_block(i)
                    add_block(j)
                    no_improve_count = 0
                    continue
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in current_blocks:
                        add_block(c)
                    no_improve_count = 0
                    continue

    # Final cleanup: remove any remaining redundant blocks from best solution
    # Reconstruct counts for best_blocks
    final_counts = [0] * N_SUBSETS
    for idx in best_blocks:
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            final_counts[bit_idx] += 1
            mm ^= lsb
    
    # Try to remove redundant blocks from best solution (multiple passes)
    final_blocks = best_blocks.copy()
    improved = True
    while improved:
        improved = False
        for i in range(len(final_blocks) - 1, -1, -1):
            m = all_masks[final_blocks[i]]
            mm = m
            redundant = True
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if final_counts[bit_idx] == 1:
                    redundant = False
                    break
                mm ^= lsb
            if redundant:
                # Remove it
                mm = m
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    final_counts[bit_idx] -= 1
                    mm ^= lsb
                final_blocks.pop(i)
                improved = True
                if time.time() > deadline:
                    break
    
    return [list(all_blocks[i]) for i in final_blocks]
