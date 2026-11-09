import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    all_blocks = []
    block_masks = []
    
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(all_blocks)
    
    # Convert block_masks to numpy array of uint64 for vectorized operations
    # Each mask fits in a single uint64 if num_t_subsets <= 64, otherwise use multiple
    # For v=19, t=4: C(19,4) = 3876, so we need multiple uint64s
    # Use list of uint64 arrays, one per 64-bit chunk
    num_chunks = (num_t_subsets + 63) // 64
    block_masks_np = np.zeros((num_blocks, num_chunks), dtype=np.uint64)
    
    for i, mask in enumerate(block_masks):
        for chunk_idx in range(num_chunks):
            start_bit = chunk_idx * 64
            end_bit = min(start_bit + 64, num_t_subsets)
            chunk_mask = (mask >> start_bit) & ((1 << (end_bit - start_bit)) - 1)
            block_masks_np[i, chunk_idx] = chunk_mask
    
    # Full uncovered mask as numpy array
    full_uncovered = np.zeros(num_chunks, dtype=np.uint64)
    for chunk_idx in range(num_chunks):
        start_bit = chunk_idx * 64
        end_bit = min(start_bit + 64, num_t_subsets)
        full_uncovered[chunk_idx] = (1 << (end_bit - start_bit)) - 1
    
    def count_bits_numpy(mask_arr):
        """Count bits in a numpy array of uint64 chunks."""
        total = 0
        for chunk in mask_arr:
            total += bin(chunk).count('1')
        return total
    
    def greedy_cover(rng, deadline):
        """Run a greedy set cover with random tie-breaking."""
        current_uncovered = full_uncovered.copy()
        chosen_indices = []
        used = np.zeros(num_blocks, dtype=bool)
        
        while True:
            if time.time() > deadline:
                return None
            
            # Check if all covered
            if not np.any(current_uncovered):
                break
            
            # Compute gains for all unused blocks
            # gain[i] = popcount(block_masks[i] & current_uncovered)
            best_gain = -1
            best_candidates = []
            
            # Vectorized gain computation
            for i in range(num_blocks):
                if used[i]:
                    continue
                # Compute intersection
                intersection = np.bitwise_and(block_masks_np[i], current_uncovered)
                gain = count_bits_numpy(intersection)
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain <= 0:
                break
            
            # Randomly pick from top candidates
            chosen_idx = rng.choice(best_candidates)
            chosen_indices.append(chosen_idx)
            used[chosen_idx] = True
            current_uncovered = np.bitwise_and(current_uncovered, ~block_masks_np[chosen_idx])
            # Fix: numpy bitwise not on uint64 gives large negative, need to mask
            for chunk_idx in range(num_chunks):
                start_bit = chunk_idx * 64
                end_bit = min(start_bit + 64, num_t_subsets)
                mask_bits = (1 << (end_bit - start_bit)) - 1
                current_uncovered[chunk_idx] &= mask_bits
        
        return chosen_indices
    
    def reduce_cover(chosen_indices, deadline):
        """Remove redundant blocks using coverage counts."""
        if not chosen_indices:
            return []
        
        # Count how many times each t-subset is covered
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        for idx in chosen_indices:
            mask = block_masks[idx]
            for bit in range(num_t_subsets):
                if mask & (1 << bit):
                    cover_count[bit] += 1
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i in range(len(chosen_indices)):
                if time.time() > deadline:
                    break
                
                idx = chosen_indices[i]
                mask = block_masks[idx]
                
                # Check if any t-subset in this block has cover_count == 1
                is_redundant = True
                for bit in range(num_t_subsets):
                    if mask & (1 << bit):
                        if cover_count[bit] == 1:
                            is_redundant = False
                            break
                
                if is_redundant:
                    # Remove this block
                    for bit in range(num_t_subsets):
                        if mask & (1 << bit):
                            cover_count[bit] -= 1
                    chosen_indices.pop(i)
                    improved = True
                    break
        
        return chosen_indices
    
    # Run multiple randomized greedy passes and keep the best
    best_solution = None
    best_size = float('inf')
    
    # Determine how many restarts we can afford
    # Estimate time per greedy run
    rng = random.Random(42)
    
    num_restarts = 20  # Try multiple diverse solutions
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        seed = restart * 12345 + 67890
        rng = random.Random(seed)
        
        chosen_indices = greedy_cover(rng, deadline - 10.0)
        if chosen_indices is None:
            continue
        
        # Reduce the cover
        reduced = reduce_cover(chosen_indices, deadline - 5.0)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_solution = reduced
    
    # If no solution found, do a final greedy without reduction
    if best_solution is None:
        rng = random.Random(0)
        chosen_indices = greedy_cover(rng, deadline - 1.0)
        if chosen_indices is not None:
            best_solution = chosen_indices
    
    # Convert indices to blocks
    if best_solution is None:
        return []
    
    return [list(all_blocks[idx]) for idx in best_solution]
