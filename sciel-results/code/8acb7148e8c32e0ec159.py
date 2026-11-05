import itertools
import random
import time

def construct(v, k, t):
    points = list(range(v))
    
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(points, t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    # Precompute for each possible k-block, the indices of t-subsets it contains
    all_k_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_k_blocks)
    block_to_subsets = []
    for block in all_k_blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(subset_to_idx[sub])
        block_to_subsets.append(subs)
        
    # Inverted Index: subset_idx -> list of block_indices that cover it
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)

    def fast_greedy(uncovered_set):
        """
        Greedily add blocks to cover uncovered_set.
        Returns a list of new block indices added.
        """
        if not uncovered_set:
            return []
            
        added = []
        # We maintain a set of uncovered indices
        # We will recalculate gains each step for simplicity, but optimize the max finding
        
        while uncovered_set:
            gains = [0] * num_blocks
            
            # Calculate gains
            for s_idx in uncovered_set:
                for b_idx in subset_to_blocks[s_idx]:
                    gains[b_idx] += 1
            
            # Find max gain block
            # Using max with key is slow. 
            # max_val = max(gains)
            # best_b = gains.index(max_val)
            
            # This is faster:
            best_b = -1
            max_gain = -1
            for i, g in enumerate(gains):
                if g > max_gain:
                    max_gain = g
                    best_b = i
                    
            if max_gain == 0:
                break
                
            added.append(best_b)
            
            # Update uncovered_set
            # Optimization: Only remove if present
            for s_idx in block_to_subsets[best_b]:
                if s_idx in uncovered_set:
                    uncovered_set.remove(s_idx)
                    
        return added

    # Initialize with empty solution
    current_blocks = []
    
    # Run initial greedy
    initial_uncovered = set(range(num_subsets))
    new_blocks = fast_greedy(initial_uncovered)
    current_blocks.extend(new_blocks)
    
    def reduce_blocks(blocks_list):
        """
        Try to remove redundant blocks.
        Returns new list of blocks and boolean if any removed.
        """
        if not blocks_list:
            return [], False

        # Count coverage
        counts = [0] * num_subsets
        for b_idx in blocks_list:
            for s_idx in block_to_subsets[b_idx]:
                counts[s_idx] += 1
        
        changed = False
        new_list = []
        for b_idx in blocks_list:
            # Check if this block is necessary
            necessary = False
            for s_idx in block_to_subsets[b_idx]:
                if counts[s_idx] == 1:
                    necessary = True
                    break
            
            if not necessary:
                changed = True
                for s_idx in block_to_subsets[b_idx]:
                    counts[s_idx] -= 1
            else:
                new_list.append(b_idx)
                
        return new_list, changed

    # Run reduction phase
    while True:
        current_blocks, changed = reduce_blocks(current_blocks)
        if not changed:
            break
            
    best_blocks = current_blocks[:]
    best_size = len(current_blocks)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # VNS Loop
    # To avoid getting stuck in a slow repair loop, we limit the number of VNS iterations
    # and check time frequently.
    
    while time.time() < deadline:
        if not best_blocks:
            break
            
        # Try a few perturbations from the current best
        improved_in_this_iter = False
        for _ in range(10): # 10 attempts per outer loop
            if time.time() > deadline:
                break
                
            perturbed = best_blocks[:]
            if len(perturbed) <= 1:
                continue
                
            # Remove 1 random block
            idx_to_remove = random.randint(0, len(perturbed) - 1)
            perturbed.pop(idx_to_remove)
            
            # Find what is now uncovered
            counts = [0] * num_subsets
            for b_idx in perturbed:
                for s_idx in block_to_subsets[b_idx]:
                    counts[s_idx] += 1
            
            uncovered_set = set()
            for i in range(num_subsets):
                if counts[i] == 0:
                    uncovered_set.add(i)
            
            # Repair with Greedy
            if uncovered_set:
                added = fast_greedy(uncovered_set)
                perturbed.extend(added)
            
            # Reduce
            while True:
                perturbed, changed = reduce_blocks(perturbed)
                if not changed:
                    break
                
            if len(perturbed) < best_size:
                best_blocks = perturbed
                best_size = len(perturbed)
                improved_in_this_iter = True
                break # Restart with new best
                
    # Convert block indices back to lists of points
    result = [list(all_k_blocks[b_idx]) for b_idx in best_blocks]
    return result
