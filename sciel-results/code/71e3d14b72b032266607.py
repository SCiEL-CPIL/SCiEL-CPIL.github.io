import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a randomized greedy set cover approach 
    followed by a redundancy removal phase.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Set random seed for reproducibility
    random.seed(42)

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    # There are C(19, 7) = 50388 blocks
    # Each block covers C(7, 4) = 35 t-subsets
    
    block_masks = []
    all_blocks = []
    
    # Generate all k-blocks and their masks
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Convert to numpy array for faster operations
    block_masks_np = np.array(block_masks, dtype=object)
    
    # Helper for bit count
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')
    
    # Best solution found so far
    best_solution = None
    best_size = float('inf')
    
    # Try multiple randomized greedy runs
    while time.time() < deadline - 5.0:
        current_uncovered = (1 << num_t_subsets) - 1
        
        chosen_block_indices = []
        chosen_block_masks = []
        
        # Randomized greedy set cover
        while current_uncovered != 0:
            if time.time() > deadline - 5.0:
                break
                
            # Compute gains for all blocks
            gains = []
            for i in range(len(block_masks)):
                intersection = block_masks[i] & current_uncovered
                gain = count_bits(intersection)
                gains.append(gain)
            
            # Find blocks with maximum gain
            max_gain = max(gains)
            
            if max_gain == 0:
                break
            
            # Get all blocks with maximum gain
            max_gain_indices = [i for i, g in enumerate(gains) if g == max_gain]
            
            # Randomly select from max gain blocks with some probability of selecting near-max
            # Use softmax-like selection: probability proportional to exp(gain / temperature)
            temperature = 0.1
            weights = []
            for i in range(len(gains)):
                if gains[i] > 0:
                    weights.append(np.exp(gains[i] / temperature))
                else:
                    weights.append(0)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                break
            
            probs = [w / total_weight for w in weights]
            
            # Select a block
            best_block_idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
            
            chosen_block_indices.append(best_block_idx)
            chosen_block_masks.append(block_masks[best_block_idx])
            
            current_uncovered &= ~block_masks[best_block_idx]
        
        # Check if we have a complete solution
        if current_uncovered == 0:
            solution_size = len(chosen_block_indices)
            if solution_size < best_size:
                best_size = solution_size
                best_solution = chosen_block_indices.copy()
        
        # If we've reached the known best, stop
        if best_size <= 151:
            break
    
    # Use the best solution found
    if best_solution is None:
        # Fallback: run deterministic greedy
        current_uncovered = (1 << num_t_subsets) - 1
        chosen_block_indices = []
        
        while current_uncovered != 0:
            best_gain = -1
            best_block_idx = -1
            
            for i in range(len(block_masks)):
                intersection = block_masks[i] & current_uncovered
                gain = count_bits(intersection)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            
            if best_block_idx == -1:
                break
            
            chosen_block_indices.append(best_block_idx)
            current_uncovered &= ~block_masks[best_block_idx]
        
        best_solution = chosen_block_indices
        
    # 4. Post-processing: Remove redundant blocks from best solution
    n_chosen = len(best_solution)
    chosen_block_masks = [block_masks[idx] for idx in best_solution]
    
    changed = True
    while changed:
        changed = False
        for i in range(n_chosen):
            if time.time() > deadline:
                break
            
            mask_i = chosen_block_masks[i]
            
            # Calculate union of others
            union_others = 0
            for j in range(n_chosen):
                if i != j:
                    union_others |= chosen_block_masks[j]
            
            # Check if mask_i is subset of union_others
            if (mask_i & ~union_others) == 0:
                # Remove block i
                best_solution.pop(i)
                chosen_block_masks.pop(i)
                n_chosen -= 1
                changed = True
                break # Restart loop as indices shifted
        
        if time.time() > deadline:
            break

    # Return the blocks
    return [list(all_blocks[idx]) for idx in best_solution]
