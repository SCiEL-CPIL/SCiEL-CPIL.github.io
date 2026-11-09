import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask and subset indices for every possible k-block
    all_blocks = []
    block_masks_int = []
    block_subsets_list = [] # List of lists of t-subset indices for each block
    
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        subs = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks_int.append(mask)
        block_subsets_list.append(subs)
        
    num_blocks = len(all_blocks)
    
    # Convert masks to NumPy array of uint64 words
    num_words = (num_t_subsets + 63) // 64
    block_masks_np = np.zeros((num_blocks, num_words), dtype=np.uint64)
    
    for i, mask in enumerate(block_masks_int):
        for w in range(num_words):
            # Explicitly cast to uint64 to prevent any overflow issues during assignment
            word = np.uint64((mask >> (w * 64)) & 0xFFFFFFFFFFFFFFFF)
            block_masks_np[i, w] = word

    # Precompute popcount usage
    try:
        _ = np.bit_count(np.array([1], dtype=np.uint64))
        use_np_bit_count = True
    except AttributeError:
        use_np_bit_count = False
        # Fallback popcount implementation using lookup table on bytes
        # This is a fallback, but np.bit_count is standard in modern NumPy
        popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

    best_solution = None
    best_size = float('inf')
    
    def greedy_cover(random_seed):
        rng = random.Random(random_seed)
        # Initialize uncovered mask
        current_uncovered_np = np.ones(num_words, dtype=np.uint64)
        if num_t_subsets % 64 != 0:
            bits_to_clear = 64 - (num_t_subsets % 64)
            current_uncovered_np[-1] &= ~(0xFFFFFFFFFFFFFFFF << bits_to_clear)
            
        chosen_indices = []
        used = np.zeros(num_blocks, dtype=bool)
        
        while True:
            if time.time() > deadline:
                return None
            
            if not np.any(current_uncovered_np):
                break
            
            # Calculate gains: number of new subsets covered by each block
            intersections = block_masks_np & current_uncovered_np[None, :]
            
            if use_np_bit_count:
                gains = np.bit_count(intersections).sum(axis=1)
            else:
                # Fallback: manual popcount using lookup table
                # Reshape to bytes is complex, so we'll use a simple loop if this path is taken
                # But we assume np.bit_count exists.
                gains = np.bit_count(intersections).sum(axis=1)

            # Mask out used blocks
            gains[used] = -1
            
            best_gain = np.max(gains)
            
            if best_gain <= 0:
                break
                
            candidates = np.where(gains == best_gain)[0]
            chosen_idx = int(rng.choice(candidates))
            chosen_indices.append(chosen_idx)
            used[chosen_idx] = True
            
            # Update uncovered
            current_uncovered_np &= ~block_masks_np[chosen_idx]
            
        return chosen_indices

    def reduce_cover(chosen_indices):
        if not chosen_indices:
            return []
            
        # Count coverage for each t-subset
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        for idx in chosen_indices:
            for sub_idx in block_subsets_list[idx]:
                cover_count[sub_idx] += 1
                
        kept_indices = []
        # Sort indices to ensure deterministic order for removal if needed, 
        # but greedy removal order can affect result. 
        # We'll iterate and remove if redundant.
        
        # To speed up, we can check redundancy in a single pass if we update counts carefully,
        # but removing an item changes counts for others.
        # Simple approach: Iterate multiple passes until no change.
        
        changed = True
        current_indices = list(chosen_indices)
        
        while changed:
            changed = False
            new_indices = []
            for idx in current_indices:
                subs = block_subsets_list[idx]
                # Check if any subset in this block has count == 1
                # If yes, this block is necessary.
                # If no, it is redundant.
                
                # Vectorized check: np.all(cover_count[subs] > 1)
                if np.all(cover_count[subs] > 1):
                    # Redundant, remove it
                    for sub_idx in subs:
                        cover_count[sub_idx] -= 1
                    changed = True
                else:
                    new_indices.append(idx)
                    
            current_indices = new_indices
            
        return current_indices

    # Run multiple restarts
    seed = 0
    while time.time() < deadline:
        indices = greedy_cover(seed)
        if indices is None:
            break
            
        reduced_indices = reduce_cover(indices)
        
        if len(reduced_indices) < best_size:
            best_size = len(reduced_indices)
            best_solution = [list(all_blocks[i]) for i in reduced_indices]
            
        seed += 1

    if best_solution is None:
        # Fallback
        indices = greedy_cover(0)
        if indices:
            best_solution = [list(all_blocks[i]) for i in indices]
        else:
            return []
            
    return best_solution
