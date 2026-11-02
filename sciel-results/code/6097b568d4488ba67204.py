import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_masks = [0] * num_blocks
    blocks_containing_triple = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
            blocks_containing_triple[idx].append(b_idx)
        block_masks[b_idx] = mask
        
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple times within time limit
    start_time = time.time()
    max_time = 100  # Leave 20s buffer
    
    run = 0
    while time.time() - start_time < max_time:
        random.seed(run)
        uncovered_mask = (1 << num_triples) - 1
        uncovered_set = set(range(num_triples))
        blocks = []
        
        while uncovered_set:
            # Pick random seed
            # Converting set to list is O(N), but N is small (2024)
            seed_idx = random.choice(list(uncovered_set))
            
            candidates = blocks_containing_triple[seed_idx]
            
            # Sample candidates to speed up
            # If fewer than 2000, use all. Else sample 2000.
            sample_size = min(2000, len(candidates))
            sampled = random.sample(candidates, sample_size)
            
            best_b_idx = -1
            max_covered = -1
            
            for b_idx in sampled:
                covered = block_masks[b_idx] & uncovered_mask
                count = covered.bit_count()
                if count > max_covered:
                    max_covered = count
                    best_b_idx = b_idx
            
            if best_b_idx == -1:
                break
                
            blocks.append(list(all_blocks[best_b_idx]))
            
            # Update uncovered
            covered_mask = block_masks[best_b_idx] & uncovered_mask
            uncovered_mask &= ~covered_mask
            
            # Update set: remove bits in covered_mask
            temp = covered_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                uncovered_set.remove(idx)
                temp ^= lsb
                
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        run += 1
        
    return best_blocks
