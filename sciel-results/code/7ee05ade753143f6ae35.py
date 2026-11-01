import itertools
import time
import random
import numpy as np
from collections import defaultdict

def _precompute(v, k, t):
    """
    Precompute mappings and bitmasks for efficient set cover operations.
    """
    tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubs)
    tsub_id = {ts: i for i, ts in enumerate(tsubs)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block tuple to index
    block_id = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute bitmasks for each block
    # Each block covers C(k, t) t-subsets.
    # We use an array of 64-bit integers.
    # n_tsubs / 64 = 11628 / 64 = 181.68 -> 182 words.
    n_words = (n_tsubs + 63) // 64
    
    # masks[i] is an array of n_words uint64s
    masks = np.zeros((n_blocks, n_words), dtype=np.uint64)
    
    # Helper to set a bit in a numpy array
    # We will fill this in a loop, but we can optimize by precomputing the t-subsets for each block type?
    # No, just loop over blocks. 75582 * 56 ops is ~4M ops. Fast enough.
    
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_id = tsub_id[ts]
            word_idx = ts_id >> 6
            bit_idx = ts_id & 63
            masks[i, word_idx] |= (1 << bit_idx)
            
    return tsubs, tsub_id, block_id, all_blocks, masks, n_words

def _greedy_cover(v, k, t, rng, tsubs, tsub_id, block_id, all_blocks, masks, n_words, deadline):
    """
    Vectorized Greedy Set Cover.
    """
    n_tsubs = len(tsubs)
    n_blocks = len(all_blocks)
    
    # uncovered_mask: array of n_words uint64s. Initially all 1s (for valid bits).
    # We need to mask out the unused bits in the last word.
    valid_bits = n_tsubs % 64
    if valid_bits == 0:
        last_mask = 0xFFFFFFFFFFFFFFFF
    else:
        last_mask = (1 << valid_bits) - 1
        
    uncovered = np.ones(n_words, dtype=np.uint64)
    uncovered[-1] = last_mask
    
    selected_indices = []
    
    # Precompute OR of all masks? No, we need dynamic uncovered.
    # To find the best block, we need to count how many uncovered bits each block covers.
    # popcount of (mask_block & uncovered)
    
    # We can do this in batches or vectorized.
    # masks is (n_blocks, n_words). uncovered is (n_words,).
    # intersection = masks & uncovered (broadcasting)
    # counts = np.sum(np.unpackbits(intersection).sum(axis=1), axis=1)
    # This is O(n_blocks * n_words * 8). 75k * 182 * 8 ~ 110M ops per step.
    # 110M ops in numpy is ~0.1s. So ~10 steps/sec. Too slow for 150s?
    # 150s * 10 = 1500 steps. Greedy usually needs ~300-500 steps.
    # So we can do ~3-5 full greedy runs.
    
    # Optimization: Use `np.unpackbits` on the intersection.
    # Actually, `np.bitwise_and` then `np.unpackbits` then `np.sum` is standard.
    
    while time.time() < deadline:
        # Calculate gain for all blocks
        # intersection shape: (n_blocks, n_words)
        intersection = np.bitwise_and(masks, uncovered)
        # Unpack bits: (n_blocks, n_words * 8)
        # Note: unpackbits returns uint8
        bits = np.unpackbits(intersection, axis=1)
        # Sum bits per block
        gains = bits.sum(axis=1)
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break # No block covers any new subset? Should not happen if uncovered not 0.
            
        selected_indices.append(best_idx)
        
        # Update uncovered
        # uncovered = uncovered & ~masks[best_idx]
        uncovered = np.bitwise_and(uncovered, np.bitwise_not(masks[best_idx]))
        
        # Check if all covered
        # If sum of unpacked uncovered bits is 0, done.
        # Faster check: if all words are 0.
        if not np.any(uncovered):
            break
            
    return [all_blocks[i] for i in selected_indices]

def _local_search(blocks, v, k, t, deadline, tsubs, tsub_id, block_id, all_blocks, masks, n_words):
    """
    Local search with bitmask acceleration.
    """
    # Convert blocks to indices
    block_indices = [block_id[b] for b in blocks]
    n_blocks = len(block_indices)
    
    # Precompute OR of all masks? 
    # For removal: Block i is redundant if OR(masks[j] for j != i) covers masks[i].
    # This is equivalent to: masks[i] & ~OR_others == 0.
    # OR_all = OR of all masks.
    # OR_others_i = OR_all if masks[i] is not the unique provider of any bit.
    # Actually, we can just check if (masks[i] & ~OR_others) == 0.
    
    # For swap: Replace i with j.
    # Valid if masks[j] covers all unique bits of i.
    # unique_i = masks[i] & ~OR_others_i
    # Condition: (masks[j] & unique_i) == unique_i.
    
    # We will maintain a list of indices and rebuild ORs periodically or use a segment tree?
    # Given n_blocks ~ 300-400, rebuilding OR of 300 arrays of 182 words is fast.
    # 300 * 182 * 8 bytes ~ 400KB. OR reduction is O(N * W).
    
    def get_or_except(idx_to_exclude):
        """Compute bitwise OR of all masks except the one at idx_to_exclude."""
        if len(block_indices) <= 1:
            return np.zeros(n_words, dtype=np.uint64)
        
        # Create a view of masks for current blocks
        # masks_subset = masks[np.array(block_indices)]
        # This copies data. 300 * 182 * 8 ~ 400KB. Acceptable.
        # But we can avoid copy if we just loop? No, numpy OR is fast.
        
        current_masks = masks[np.array(block_indices)]
        
        if idx_to_exclude is None:
            return np.bitwise_or.reduce(current_masks, axis=0)
        else:
            # Exclude idx_to_exclude
            # We can do OR of all, then "subtract"? No, OR is not invertible.
            # We must OR all others.
            # Create mask of indices to include
            # Or just slice.
            
            # Since we rebuild often, let's just do:
            # or_all = OR(current_masks)
            # This doesn't help for "except".
            
            # Efficient way:
            # Split into two halves? Or just loop?
            # With N=300, looping 300 times doing numpy OR is slow.
            # Vectorized:
            # Create a boolean mask for exclusion
            # But we need to OR a subset of rows.
            
            # Trick:
            # or_all = np.bitwise_or.reduce(current_masks, axis=0)
            # This is the OR of everything.
            # We need OR of everything EXCEPT i.
            # If we know which bits are ONLY in i, we can't just subtract.
            
            # Let's just do:
            # indices_to_or = [j for j in range(len(block_indices)) if j != idx_to_exclude]
            # This is O(N) Python overhead.
            
            # Better:
            # Use np.bitwise_or.reduce on a sliced array?
            # We can't easily slice rows by index without a copy or fancy indexing.
            # fancy indexing copies.
            
            # Alternative:
            # Maintain a prefix OR and suffix OR?
            # prefix[i] = OR of masks[0..i-1]
            # suffix[i] = OR of masks[i+1..N-1]
            # OR_except(i) = prefix[i] | suffix[i]
            
            # Let's compute prefix and suffix ORs once per local search step?
            # No, block_indices changes.
            
            # Given the small N, maybe just rebuilding is fine if we do it smartly.
            # Let's try a simple loop for OR_except if N is small?
            # No, numpy is better.
            
            # Let's compute OR of all, and then for each block, check redundancy differently?
            
            # Redundancy check:
            # Block i is redundant if for every bit b in masks[i], there exists j != i such that bit b is in masks[j].
            # This is equivalent to: masks[i] & ~OR_others == 0.
            
            # Let's compute OR_others for all i at once?
            # or_all = OR(all)
            # For each i, OR_others_i is not easily derived from or_all.
            
            # However, we can compute:
            # unique_i = masks[i] & ~OR_others_i
            # If we can't get OR_others_i fast, we are stuck.
            
            # Let's use the prefix/suffix method. It requires O(N*W) memory and O(N*W) time to build.
            # N=300, W=182. 300*182 = 54k ops. Very fast.
            
            current_masks = masks[np.array(block_indices)]
            N = len(block_indices)
            
            prefix_or = np.zeros((N, n_words), dtype=np.uint64)
            suffix_or = np.zeros((N, n_words), dtype=np.uint64)
            
            if N > 0:
                prefix_or[0] = current_masks[0]
                for i in range(1, N):
                    prefix_or[i] = np.bitwise_or(prefix_or[i-1], current_masks[i])
                
                suffix_or[N-1] = current_masks[N-1]
                for i in range(N-2, -1, -1):
                    suffix_or[i] = np.bitwise_or(suffix_or[i+1], current_masks[i])
                
                # OR_except(i) = prefix_or[i-1] | suffix_or[i+1]
                # Handle boundaries
                or_except = np.zeros((N, n_words), dtype=np.uint64)
                
                if N > 1:
                    # For i=0: suffix_or[1]
                    or_except[0] = suffix_or[1]
                    # For i=N-1: prefix_or[N-2]
                    or_except[N-1] = prefix_or[N-2]
                    # For 0 < i < N-1: prefix_or[i-1] | suffix_or[i+1]
                    if N > 2:
                        or_except[1:-1] = np.bitwise_or(prefix_or[:-2], suffix_or[2:])
                else:
                    or_except[0] = 0
                    
            return or_except, current_masks

    # Main Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_blocks = len(block_indices)
        
        if n_blocks == 0:
            break
            
        or_except, current_masks = get_or_except(None) # This returns (N, W) array of ORs excluding each i
        
        # Pass 1: Removal
        # Check if masks[i] & ~or_except[i] == 0
        # unique_bits = current_masks & ~or_except
        # if all unique_bits[i] are 0, block i is removable.
        
        unique_bits = np.bitwise_and(current_masks, np.bitwise_not(or_except))
        
        # Find indices where unique_bits are all zero
        # unique_bits is (N, W). We need to check if any word is non-zero.
        # any_nonzero = unique_bits.any(axis=1)
        # removable = ~any_nonzero
        
        removable_mask = ~unique_bits.any(axis=1)
        
        removable_indices = np.where(removable_mask)[0]
        
        if len(removable_indices) > 0:
            # Remove the first one found (or all? Removing one might make others redundant.
            # Removing one is safer for index management in this simple loop structure.
            # But removing all is faster. Let's remove all currently redundant.
            # Wait, if we remove multiple, the ORs change.
            # Let's just remove one to be safe and restart the loop.
            idx_to_remove = removable_indices[0]
            
            # Remove from list
            block_indices.pop(idx_to_remove)
            improved = True
            continue # Restart to recompute ORs
            
        # Pass 2: Swap
        # Try to replace block i with block j.
        # Condition: masks[j] covers unique_i.
        # unique_i = current_masks[i] & ~or_except[i]
        # We already have unique_bits[i].
        
        # For each i, we want to find a j (not in current blocks) such that
        # (masks[j] & unique_bits[i]) == unique_bits[i]
        # AND replacing i with j allows us to potentially remove j later?
        # No, standard 1-swap: remove i, add j.
        # This is valid if j covers all unique bits of i.
        # Does it reduce block count? No, count stays same.
        # But it might enable a removal in the next iteration.
        # Or, we can look for a "2-swap" or "1-add 1-remove" that leads to a net reduction?
        # The directive says "Removal, 1-Block Replacement".
        # Replacement usually implies changing the set of blocks.
        # If we swap i for j, and j is "better" (covers more non-unique stuff?), it doesn't help count.
        # BUT, if we swap i for j, and then j becomes redundant (because i covered something j doesn't, but wait, j covers unique_i, so j is at least as good for those bits. But i might cover bits that are covered by others, so i is not unique there.
        # Actually, if we swap i for j, the new unique bits for j are (masks[j] & ~OR_new_others).
        # This is complex.
        
        # Simpler heuristic:
        # Look for a block j NOT in current set such that it covers unique_i.
        # If we find one, we swap.
        # To ensure progress, we should prefer swaps that allow immediate removal?
        # That's a 2-step move.
        
        # Let's stick to:
        # 1. Remove redundant blocks.
        # 2. If no removal, try to swap a block i with a block j (not in set) such that:
        #    a) j covers unique_i.
        #    b) The new block j is "more central" or we just do it randomly to escape local minima?
        #    Actually, random swap might increase size? No, size is constant.
        #    But it might allow a removal in the next step.
        
        # Let's try to find a swap that makes the new block j have FEWER unique bits than i did?
        # Or just any valid swap.
        
        # Optimization:
        # Precompute for each block i, the list of candidate blocks j that cover unique_i.
        # This is expensive to compute on the fly.
        
        # Alternative:
        # Just iterate through a random subset of all blocks as candidates for j.
        
        if time.time() > deadline - 5:
            break
            
        # Pick a random block i to try to swap
        i = rng.randint(0, n_blocks - 1)
        u_i = unique_bits[i]
        
        if not np.any(u_i):
            # Should have been removed
            continue
            
        # Find a candidate j
        # We can check all blocks? 75k blocks.
        # For each j, check (masks[j] & u_i) == u_i.
        # This is a vectorized check against u_i.
        
        # u_i is (W,). masks is (N_total, W).
        # intersection = np.bitwise_and(masks, u_i)
        # match = np.all(intersection == u_i, axis=1)
        
        # This is O(N_total * W). 75k * 182 ~ 13M ops. ~0.01s.
        # We can do this for a few i's.
        
        found_swap = False
        # Try a few random i's
        for _ in range(10):
            i = rng.randint(0, n_blocks - 1)
            u_i = unique_bits[i]
            if not np.any(u_i):
                continue
                
            # Find all j that cover u_i
            # We must exclude j that are already in block_indices
            # First find all valid j
            intersection = np.bitwise_and(masks, u_i)
            # Check if intersection == u_i for all words
            # Note: u_i might have 0s. intersection must match u_i exactly.
            # Actually, we just need intersection to have 1s where u_i has 1s.
            # So (intersection & u_i) == u_i is sufficient.
            # Since intersection is subset of masks, and we AND with u_i, it is subset of u_i.
            # So we check if (intersection == u_i).
            
            valid_js = np.where(np.all(intersection == u_i, axis=1))[0]
            
            if len(valid_js) == 0:
                continue
                
            # Filter out j already in block_indices
            current_set = set(block_indices)
            candidates = [j for j in valid_js if j not in current_set]
            
            if not candidates:
                continue
                
            # Pick a random candidate
            j = rng.choice(candidates)
            
            # Perform swap
            block_indices[i] = j
            found_swap = True
            improved = True
            break
            
        if not found_swap:
            # No swap found in this batch
            # To avoid infinite loop, we might need to break or do a random perturbation
            # But if we can't remove and can't swap, we are stuck.
            # We should break to save time?
            # Or try more swaps?
            # Let's break to avoid wasting time if stuck.
            break
            
    return [all_blocks[i] for i in block_indices]

def construct(v, k, t):
    deadline = time.time() + 145
    rng = random.Random(42)
    
    # Precompute
    tsubs, tsub_id, block_id, all_blocks, masks, n_words = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multiple Greedy Restarts
    # Each greedy run takes ~0.1s - 0.5s.
    # We can do ~10-20 runs.
    while time.time() < deadline - 60:
        blocks = _greedy_cover(v, k, t, rng, tsubs, tsub_id, block_id, all_blocks, masks, n_words, deadline - 5)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
            
    # Phase 2: Local Search on the best blocks
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, v, k, t, deadline - 2, tsubs, tsub_id, block_id, all_blocks, masks, n_words)
        
    return best_blocks
