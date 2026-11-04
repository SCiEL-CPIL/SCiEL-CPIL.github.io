import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets and map to IDs
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute bitsets for all blocks
    block_triplets = [0] * n_blocks
    for idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_triplets[idx] = mask
        
    all_mask = (1 << n_triplets) - 1

    best_blocks = []
    best_len = float('inf')

    # Run multiple randomized greedy constructions
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 60:
            break
            
        uncovered = all_mask
        blocks = []
        block_indices = []
        
        # Greedy Construction with Randomized Tie-Breaking
        while uncovered:
            if time.time() > deadline - 30:
                break
                
            # Calculate gains
            # Using map and lambda is faster than a for loop in Python
            gains = list(map(lambda m: (m & uncovered).bit_count(), block_triplets))
            
            # Find max gain
            max_gain = max(gains)
            
            # Collect all blocks with max gain (or close to max)
            # To speed up, we can just pick randomly from those with gain > 0.9 * max_gain
            threshold = int(max_gain * 0.9)
            candidates = [i for i, g in enumerate(gains) if g >= threshold]
            
            if not candidates:
                candidates = [i for i, g in enumerate(gains) if g > 0]
            
            if not candidates:
                break # Should not happen if uncovered != 0
                
            # Pick random candidate
            best_idx = random.choice(candidates)
            
            blocks.append(all_blocks[best_idx])
            block_indices.append(best_idx)
            uncovered &= ~block_triplets[best_idx]
            
        # Update best
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()

    # Local Search on the best solution
    if not best_blocks:
        return []

    blocks = best_blocks
    
    # Maintain count of how many times each triplet is covered
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks_list, delta=1):
        for b in blocks_list:
            for triplet in itertools.combinations(b, t):
                tid = triplet_to_id[triplet]
                count[tid] += delta

    # Initialize counts
    update_counts(blocks, 1)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            if count[tid] == 1:
                return False
        return True

    def try_remove_redundant(blocks_list):
        removed = False
        new_blocks = []
        for b in blocks_list:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def get_unique_triplets(block):
        """Return list of triplet IDs that are only covered by this block."""
        unique = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            if count[tid] == 1:
                unique.append(tid)
        return unique

    def try_merge_2_for_1(blocks_list):
        n = len(blocks_list)
        if n < 2:
            return blocks_list, False
            
        # Precompute unique triplets for each block
        block_unique = [get_unique_triplets(b) for b in blocks_list]
        
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks_list, False
            if not block_unique[i]:
                continue
                
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks_list, False
                if not block_unique[j]:
                    continue
                
                # Union of unique triplets
                all_unique = set(block_unique[i]) | set(block_unique[j])
                
                # Check if there is a block that covers all these unique triplets
                # The union of points in these triplets must be <= k
                points_needed = set()
                for tid in all_unique:
                    points_needed.update(all_triplets[tid])
                    
                if len(points_needed) <= k:
                    # We can form a block containing points_needed
                    # Fill the rest with arbitrary points to make size k
                    new_block = list(points_needed)
                    while len(new_block) < k:
                        # Pick a point not in new_block
                        # Simple random or sequential
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    # Update counts
                    update_counts([blocks_list[i], blocks_list[j]], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks_list) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    
                    return new_blocks, True
                    
        return blocks_list, False

    progress_made = True
    while progress_made and time.time() < deadline - 10:
        progress_made = False
        
        blocks, removed = try_remove_redundant(blocks)
        if removed:
            progress_made = True
            
        if not progress_made:
            blocks, merged = try_merge_2_for_1(blocks)
            if merged:
                progress_made = True

    return blocks

if __name__ == "__main__":
    pass
