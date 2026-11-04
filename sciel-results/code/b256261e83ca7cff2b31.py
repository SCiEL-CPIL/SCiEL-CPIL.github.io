import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    # Precompute bitsets for all possible k-blocks
    # Each block is represented by a bitset of length n_triplets
    # Number of blocks is C(v, k)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # We will store the bitsets in a NumPy array of shape (n_blocks, n_bits)
    # n_triplets = 2024 for v=24, k=7, t=3. 2024 bits = 32 uint64s.
    n_uint64 = (n_triplets + 63) // 64
    block_bitsets = np.zeros((n_blocks, n_uint64), dtype=np.uint64)
    
    # Map triplet ID to bit position
    # tid -> (tid // 64, tid % 64)
    
    # Fill bitsets
    # This is the expensive part. 
    # Optimization: Precompute for each point p, the list of triplet IDs containing p?
    # Or just iterate?
    # C(24,7) = 346104. Each block has C(7,3)=35 triplets.
    # Total operations: 346104 * 35 ~ 12 million. This is fast in Python if not doing heavy lifting per op.
    # But we are setting bits in a NumPy array.
    
    # To speed up, we can precompute the triplet IDs for each block?
    # Actually, iterating 12M times in Python is ~1-2 seconds.
    for b_idx, block in enumerate(all_blocks):
        block_set = set(block)
        # Find all triplets in this block
        for triplet in itertools.combinations(block, t):
            tid = get_id(triplet)
            word_idx = tid // 64
            bit_idx = tid % 64
            block_bitsets[b_idx, word_idx] |= (1 << bit_idx)

    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        # Current uncovered mask
        # Initially all 1s (all uncovered)
        uncovered_mask = np.ones(n_uint64, dtype=np.uint64)
        # Set the last few bits to 0 if n_triplets is not multiple of 64
        if n_triplets % 64 != 0:
            remainder = n_triplets % 64
            # Mask for the last word: bits 0 to remainder-1 are valid
            # We want 1s for valid bits, 0s for invalid.
            # Since we treat 1 as "uncovered", we want 1s for valid bits.
            # The ones() already has 1s everywhere.
            # We need to clear the bits >= remainder in the last word.
            last_word_mask = (1 << remainder) - 1
            uncovered_mask[-1] &= last_word_mask
            
        blocks_selected = []
        used_blocks = np.zeros(n_blocks, dtype=bool)
        
        while True:
            if time.time() > deadline - 20:
                break
            
            # Check if any uncovered
            if not np.any(uncovered_mask):
                break
                
            # Calculate coverage: block_bitsets & uncovered_mask
            # We want to find the block with max sum of bits
            # Summing 32 uint64s for 346k blocks is expensive if done naively.
            # But NumPy can do it.
            
            # Optimization: We can maintain a list of candidate blocks?
            # Or just compute for all. 346k * 32 operations is ~11M ops.
            # NumPy should handle this in < 0.1s.
            
            # Intersection
            # Note: block_bitsets is (n_blocks, n_uint64)
            # uncovered_mask is (n_uint64,)
            # Broadcasting works.
            
            # We need the count of 1s in the intersection.
            # np.bitwise_and(block_bitsets, uncovered_mask)
            # Then count bits.
            
            # Counting bits in NumPy is not directly vectorized for uint64 without lookup tables or loops.
            # However, we can use a trick or just loop over the words if n_uint64 is small (32).
            
            # Let's use a lookup table for bit counting for 8-bit chunks?
            # Or just use `np.unpackbits`? That expands to 8x size. 346k * 32 * 8 = 88M bytes. Manageable.
            # But `np.unpackbits` is slow.
            
            # Alternative: Since n_uint64 is small (32), we can loop over the 32 words.
            # But doing it in a Python loop for 32 words, where each word op is vectorized over 346k blocks, is fast.
            
            scores = np.zeros(n_blocks, dtype=np.int32)
            for w in range(n_uint64):
                # Intersect and count bits
                # We can use a precomputed lookup table for 16-bit or 8-bit?
                # Or just use `np.bit_count` if available (NumPy 2.0+? No, standard is older).
                # Let's assume we can use a simple loop or a trick.
                # Actually, `np.unpackbits` is the most portable "vectorized" way, but slow.
                
                # Let's try a different approach for speed:
                # We only need to find the MAX.
                # We can maintain a running max? No, mask changes.
                
                # Let's use `np.unpackbits` for simplicity first, if it's too slow, we optimize.
                # Actually, 346104 * 32 * 8 bytes = 88 MB.
                # unpackbits creates this array.
                # Then sum along axis 1.
                # This might be ~100ms per iteration.
                # If we do ~70 iterations, that's 7 seconds. Acceptable.
                
                # However, we can be smarter.
                # We can precompute the bit counts for 16-bit chunks?
                # Let's stick to the simplest NumPy approach for now.
                
                # To speed up bit counting, we can use a lookup table for bytes.
                # But let's try `np.unpackbits` first.
                
                # Wait, `np.bitwise_and` returns an array.
                # `np.unpackbits` works on uint8.
                # We have uint64. We need to view as uint8?
                # `block_bitsets[:, w].view(np.uint8)` -> shape (n_blocks, 8)
                # `uncovered_mask[w].view(np.uint8)` -> shape (8,)
                
                inter = np.bitwise_and(block_bitsets[:, w], uncovered_mask[w])
                # View as uint8
                inter_bytes = inter.view(np.uint8)
                # unpackbits
                bits = np.unpackbits(inter_bytes, axis=1)
                # Sum
                scores += bits.sum(axis=1)
                
            # Find best block
            # Add small random noise to break ties?
            # scores += rng.random(n_blocks) * 0.1 ?
            # No, strict max is better for greedy.
            
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                # No block covers any new triplet? Should not happen if uncovered_mask is not 0
                # unless all blocks are used?
                break
            
            blocks_selected.append(all_blocks[best_idx])
            used_blocks[best_idx] = True
            
            # Update uncovered_mask
            # uncovered_mask &= ~block_bitsets[best_idx]
            # We need to invert the block bitset.
            # ~block_bitsets[best_idx] might have 1s in the padding bits.
            # We must mask with the valid bits mask.
            
            # valid_mask is 1s for valid triplet IDs, 0s for padding.
            # We constructed uncovered_mask initially with padding 0s.
            # So we can just do:
            uncovered_mask &= ~block_bitsets[best_idx]
            # But ~block_bitsets[best_idx] has 1s in padding.
            # If uncovered_mask has 0s in padding, 0 & 1 = 0. So it's safe.
            
            # Ensure padding stays 0
            if n_triplets % 64 != 0:
                 uncovered_mask[-1] &= (1 << (n_triplets % 64)) - 1

        return [list(b) for b in blocks_selected]

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds
    # The standard greedy is deterministic if we don't add noise.
    # To diversify, we can add random noise to scores or randomize tie-breaking.
    # Or just run once? The local search is the main improver.
    # Let's run a few with different tie-breaking strategies if we implement them.
    # For now, let's just run the deterministic one, maybe with a few random restarts of the *order* of evaluation?
    # Actually, the standard greedy is quite strong.
    
    # To allow for "improvements within its family", we can try a "weighted" greedy where we penalize blocks that overlap heavily with previous ones?
    # That's hard to do with the global bitset approach easily without O(N^2) checks.
    
    # Let's just run the standard greedy once, and rely on local search.
    # But the prompt suggests "Standard greedy set cover ... finds a denser initial solution".
    # So one good run + local search should be better than many bad incremental runs.
    
    # Let's do 3 seeds with slight randomization in tie-breaking to see if we get lucky.
    for seed in range(5):
        if time.time() > deadline - 30:
            break
        
        # We need to modify greedy_set_cover to accept a seed and use it for tie-breaking
        # Let's create a wrapper or modify the function.
        
        # For now, let's just run the deterministic one.
        # If we want randomness, we can perturb scores.
        
        # Let's modify greedy_set_cover to take a seed and add noise to ties.
        # But editing the function definition is easier.
        
        pass # We will define a new function below or edit the existing one.

    # Let's redefine greedy_set_cover to support noise
    def greedy_set_cover_noisy(seed=0):
        rng = random.Random(seed)
        uncovered_mask = np.ones(n_uint64, dtype=np.uint64)
        if n_triplets % 64 != 0:
            remainder = n_triplets % 64
            uncovered_mask[-1] &= (1 << remainder) - 1
            
        blocks_selected = []
        
        while True:
            if time.time() > deadline - 20:
                break
            if not np.any(uncovered_mask):
                break
                
            scores = np.zeros(n_blocks, dtype=np.int32)
            for w in range(n_uint64):
                inter = np.bitwise_and(block_bitsets[:, w], uncovered_mask[w])
                inter_bytes = inter.view(np.uint8)
                bits = np.unpackbits(inter_bytes, axis=1)
                scores += bits.sum(axis=1)
                
            # Add noise to break ties
            # Only add noise to blocks with the max score?
            # Or add small noise to all?
            # Adding noise to all might pick a suboptimal block.
            # Better: find max score, then pick randomly from those with max score.
            
            max_score = np.max(scores)
            if max_score == 0:
                break
            
            # Find indices with max score
            candidates = np.where(scores == max_score)[0]
            if len(candidates) > 0:
                best_idx = candidates[rng.randint(0, len(candidates)-1)]
            else:
                break # Should not happen
            
            blocks_selected.append(all_blocks[best_idx])
            
            uncovered_mask &= ~block_bitsets[best_idx]
            if n_triplets % 64 != 0:
                 uncovered_mask[-1] &= (1 << (n_triplets % 64)) - 1

        return [list(b) for b in blocks_selected]

    for seed in range(10):
        if time.time() > deadline - 40:
            break
        blocks = greedy_set_cover_noisy(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_set_cover_noisy(0)

    current_blocks = list(best_blocks)
    
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                count[get_id(triplet)] += delta

    update_counts(current_blocks, 1)

    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_id(triplet)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_merge(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        for i in range(n):
            if time.time() > deadline - 3:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = set()
            for triplet in itertools.combinations(b1, t):
                tid = get_id(triplet)
                ids_b1.add(tid)
                
            unique_b1 = [tid for tid in ids_b1 if count[tid] == 1]
            
            if not unique_b1:
                continue

            for j in range(i + 1, n):
                if time.time() > deadline - 3:
                    return blocks, False
                b2 = blocks[j]
                ids_b2 = set()
                for triplet in itertools.combinations(b2, t):
                    tid = get_id(triplet)
                    ids_b2.add(tid)
                    
                unique_b2 = [tid for tid in ids_b2 if count[tid] == 1]
                
                all_unique_ids = set(unique_b1) | set(unique_b2)
                
                points_needed = set()
                for tid in all_unique_ids:
                    points_needed.update(all_triplets[tid])
                    
                if len(points_needed) <= k:
                    new_block = list(points_needed)
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = []
                    for idx, b in enumerate(blocks):
                        if idx != i and idx != j:
                            new_blocks.append(b)
                    new_blocks.append(new_block)
                    
                    return new_blocks, True

        return blocks, False

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        block_unique = []
        for b in blocks:
            ids_b = set()
            for triplet in itertools.combinations(b, t):
                tid = get_id(triplet)
                ids_b.add(tid)
            unique_b = [tid for tid in ids_b if count[tid] == 1]
            block_unique.append(unique_b)

        for i in range(n):
            if time.time() > deadline - 3:
                return blocks, False
            if not block_unique[i]:
                continue
            for j in range(i + 1, n):
                if time.time() > deadline - 3:
                    return blocks, False
                if not block_unique[j]:
                    continue

                all_unique = set(block_unique[i]) | set(block_unique[j])
                points_needed = set()
                for tid in all_unique:
                    points_needed.update(all_triplets[tid])

                if len(points_needed) <= k:
                    new_block = list(points_needed)
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()

                    update_counts([blocks[i], blocks[j]], -1)
                    update_counts([new_block], 1)

                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    def try_replace_1_for_1(blocks):
        n = len(blocks)
        if n < 1:
            return blocks, False

        block_unique = []
        for b in blocks:
            ids_b = set()
            for triplet in itertools.combinations(b, t):
                tid = get_id(triplet)
                ids_b.add(tid)
            unique_b = [tid for tid in ids_b if count[tid] == 1]
            block_unique.append(unique_b)

        for i in range(n):
            if time.time() > deadline - 3:
                return blocks, False
            unique_b = block_unique[i]
            if not unique_b:
                continue

            points_needed = set()
            for tid in unique_b:
                points_needed.update(all_triplets[tid])

            if len(points_needed) > k:
                continue

            current_block = list(blocks[i])
            current_block_set = set(current_block)
            
            current_triplet_ids = set()
            for triplet in itertools.combinations(current_block, t):
                tid = get_id(triplet)
                current_triplet_ids.add(tid)

            new_block = list(points_needed)
            new_block_set = set(new_block)
            
            while len(new_block) < k:
                candidates = [p for p in range(v) if p not in new_block_set]
                if not candidates:
                    break
                
                best_point = -1
                best_gain = 0
                pairs_in_block = list(itertools.combinations(new_block, t - 1))
                
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if count[tid] >= 2:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point == -1 or best_gain == 0:
                    best_point = random.choice(candidates)
                
                new_block.append(best_point)
                new_block_set.add(best_point)
            
            new_block.sort()
            
            new_triplet_ids = set()
            for triplet in itertools.combinations(new_block, t):
                tid = get_id(triplet)
                new_triplet_ids.add(tid)
            
            if not set(unique_b).issubset(new_triplet_ids):
                continue

            old_covered = len(current_triplet_ids)
            new_covered = len(new_triplet_ids)
            
            if new_covered > old_covered:
                update_counts([blocks[i]], -1)
                update_counts([new_block], 1)
                new_blocks = blocks[:i] + [new_block] + blocks[i+1:]
                return new_blocks, True

        return blocks, False

    progress_made = True
    while progress_made and time.time() < deadline - 5:
        progress_made = False
        
        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True

        if not progress_made:
            current_blocks, merged = try_swap_2_for_1(current_blocks)
            if merged:
                progress_made = True

        if not progress_made:
            current_blocks, replaced = try_replace_1_for_1(current_blocks)
            if replaced:
                progress_made = True

    return current_blocks

if __name__ == "__main__":
    pass
