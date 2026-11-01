import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Finds a covering design C(v, k, t) using a randomized greedy set cover approach.
    """
    # For this specific task v=24, k=7, t=3.
    # We can hardcode optimizations for this specific case if needed, 
    # but the generic algorithm should work well.
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for faster lookup
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # We need to evaluate candidate blocks.
    # Generating all C(24, 7) = 346,104 blocks upfront might be memory heavy if we store all their coverage sets,
    # but it's feasible. However, calculating gain dynamically is better.
    
    # A better strategy: 
    # 1. Maintain a set of uncovered t-subsets.
    # 2. At each step, find the block that covers the most uncovered t-subsets.
    #    To do this efficiently, we can iterate through potential blocks? No, too slow.
    #    Instead, we can use a heuristic: 
    #    Pick a random uncovered t-subset. Generate random k-sets containing it?
    #    Or, use a "Greedy with Randomization" where we pick the best block from a pool of candidates.
    
    # Let's use a standard Greedy Set Cover with random restarts.
    # To speed up "find best block", we can precompute the list of all k-subsets?
    # C(24,7) is ~346k. For each block, precomputing its t-subsets is 35 per block.
    # Total triples stored: 346k * 35 ~ 12 million ints. This is fine.
    
    # However, checking gain for every block at every step is O(NumBlocks * t) per step.
    # NumBlocks ~ 346k. Steps ~ 76. Total ops ~ 26M * 3 = 78M. This is fast enough in Python/NumPy if vectorized?
    # Maybe not pure Python. Let's try to optimize.
    
    # Alternative: 
    # 1. Generate a pool of candidate blocks.
    # 2. Use a priority queue or simply scan the pool to find the max gain.
    
    # Let's try a robust approach:
    # 1. Precompute all k-subsets and their constituent t-subsets (as indices).
    # 2. Use a loop with random restarts.
    # 3. In each run, maintain an array `uncovered` of booleans.
    # 4. At each step, find the block with max count of True in its t-subset list.
    
    # Precomputation
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # For each k-subset, store the indices of the t-subsets it covers
    # This is a list of lists of ints.
    block_cover_indices = []
    for block in all_k_subsets:
        t_subsets_in_block = list(itertools.combinations(block, t))
        indices = [t_sub_index[ts] for ts in t_subsets_in_block]
        block_cover_indices.append(indices)
        
    # Convert to numpy array for faster summing? 
    # block_cover_indices is a list of lists of varying length? No, all length C(k,t).
    # C(7,3) = 35. So it's a fixed size.
    # Let's make it a 2D numpy array: shape (n_k_subsets, C(k,t))
    c_k_t = len(block_cover_indices[0])
    cover_matrix = np.array(block_cover_indices, dtype=np.int32)
    
    best_blocks = []
    best_count = float('inf')
    
    start_time = time.time()
    timeout = 115  # Leave 5s margin
    
    for restart in range(200):
        if time.time() - start_time > timeout:
            break
            
        random.seed(restart)
        
        # 1. Greedy Construction
        uncovered = np.ones(n_t_subsets, dtype=bool)
        chosen_indices = []
        
        while np.any(uncovered):
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[random.randint(0, len(candidates)-1)]
            
            chosen_indices.append(chosen_idx)
            uncovered[cover_matrix[chosen_idx]] = False
        
        # 2. Local Search: Redundancy Removal
        # We want to remove any block that is not essential.
        # A block is essential if it is the *only* block covering some t-subset.
        # If for all t-subsets in the block, count > 1, the block is redundant.
        
        if chosen_indices:
            counts = np.zeros(n_t_subsets, dtype=np.int32)
            for idx in chosen_indices:
                counts[cover_matrix[idx]] += 1
            
            changed = True
            while changed:
                changed = False
                # Iterate over a copy because we might modify the list
                for i in range(len(chosen_indices)):
                    idx = chosen_indices[i]
                    # Check if all t-subsets in this block have count > 1
                    if np.all(counts[cover_matrix[idx]] > 1):
                        # Remove this block
                        counts[cover_matrix[idx]] -= 1
                        chosen_indices.pop(i)
                        changed = True
                        break # Restart scan with updated counts
        
        # Convert indices back to blocks
        current_blocks = [all_k_subsets[i] for i in chosen_indices]
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
    return best_blocks
