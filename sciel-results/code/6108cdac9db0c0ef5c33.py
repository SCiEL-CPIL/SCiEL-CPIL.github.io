import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    # Map t-subsets to integer IDs for faster lookup and counting
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)
    
    # Precompute for each point x and each (t-1)-subset, which triples are formed
    # For efficiency: for a given block of size b, we need to compute gain for each candidate x
    # gain(x) = number of (t-1)-subsets P of block such that (x, P) forms an uncovered triple
    
    # Precompute: for each point x (0..v-1) and each (t-1)-subset s, the triple index
    # This is too large, so we'll compute on the fly but vectorize
    
    best_blocks = None
    best_len = float('inf')
    
    def random_greedy_cover():
        # uncovered[i] is 1 if triple i is uncovered
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []
        
        # Precompute all (t-1)-subsets for quick reference
        all_t_minus_1 = list(itertools.combinations(all_points, t - 1))
        
        while True:
            # Check if all covered
            if not np.any(uncovered):
                break
            
            # Pick the most constrained uncovered triple (fewest candidate blocks)
            # For speed, pick randomly but weight toward uncovered
            seed_idx = None
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = random.choice(uncovered_indices.tolist())
            
            seed_triple = all_t_subsets[seed_idx]
            block = list(seed_triple)
            block_set = set(block)
            
            # Extend block to size k using vectorized gain computation
            while len(block) < k:
                if time.time() > deadline - 5.0:
                    break
                
                block_pairs = list(itertools.combinations(block, t - 1))
                candidates = [x for x in all_points if x not in block_set]
                
                if not candidates:
                    break
                
                # Compute gain for each candidate x
                # For each x, gain = sum over (t-1)-subsets P of block of uncovered[triple(x, P)]
                gains = np.zeros(len(candidates), dtype=np.int32)
                
                for x_i, x in enumerate(candidates):
                    for pair in block_pairs:
                        triple = tuple(sorted((x,) + pair))
                        idx = triple_to_idx[triple]
                        gains[x_i] += uncovered[idx]
                
                # Pick the candidate with highest gain, break ties randomly
                max_gain = np.max(gains)
                if max_gain <= 0:
                    # No gain, pick randomly
                    block.append(random.choice(candidates))
                else:
                    best_indices = np.where(gains == max_gain)[0]
                    chosen = random.choice(best_indices.tolist())
                    block.append(candidates[chosen])
                
                block_set.add(block[-1])
            
            if len(block) < k:
                # Fill with random points if we ran out of time
                block_set = set(block)
                while len(block) < k:
                    available = [x for x in all_points if x not in block_set]
                    if not available:
                        break
                    block.append(random.choice(available))
                    block_set.add(block[-1])
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                uncovered[idx] = 0
        
        return blocks
    
    def local_search_remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # triple_count[idx] = number of blocks covering triple idx
        triple_count = np.zeros(num_triples, dtype=np.int32)
        
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                triple_count[idx] += 1
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[tuple(triple)]
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = []
                for i, block in enumerate(blocks):
                    if i not in remove_set:
                        new_blocks.append(block)
                
                # Update counts
                for i in to_remove:
                    block = blocks[i]
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[tuple(triple)]
                        triple_count[idx] -= 1
                
                blocks = new_blocks
                changed = True
                
        return blocks
    
    def local_search_swap(blocks, triple_count):
        """Try to remove a block and replace it with a better one, or just remove if possible."""
        if len(blocks) < 2:
            return blocks
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            for i in range(len(blocks)):
                if time.time() > deadline - 10.0:
                    break
                
                # Check if block i is redundant
                is_redundant = True
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[tuple(triple)]
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[tuple(triple)]
                        triple_count[idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break
        
        return blocks
    
    # Main loop
    while time.time() < deadline - 5.0:
        blocks = random_greedy_cover()
        if not blocks:
            continue
        
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
    return best_blocks if best_blocks is not None else []
