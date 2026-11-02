import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin

    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    num_words = (num_triples + 63) // 64
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_bits = np.zeros((num_blocks, num_words), dtype=np.uint64)
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            word_idx = t_idx // 64
            bit_idx = t_idx % 64
            block_bits[b_idx, word_idx] |= (1 << bit_idx)
            triple_to_blocks[t_idx].append(b_idx)
            
    # Full mask
    full_mask = np.zeros(num_words, dtype=np.uint64)
    for i in range(num_words):
        full_mask[i] = np.uint64(0xFFFFFFFFFFFFFFFF)
    extra_bits = num_words * 64 - num_triples
    if extra_bits > 0:
        full_mask[-1] &= (np.uint64(0xFFFFFFFFFFFFFFFF) >> extra_bits)

    best_blocks = None
    best_count = float('inf')
    
    def reduce_blocks(blocks_list):
        if not blocks_list:
            return []
        current_blocks = list(blocks_list)
        
        # Simple reduction: check each block
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                # Check if removing b_idx leaves all triples covered
                # Compute OR of all other blocks
                temp_coverage = np.zeros(num_words, dtype=np.uint64)
                for j in range(len(current_blocks)):
                    if j != i:
                        temp_coverage |= block_bits[current_blocks[j]]
                        
                if np.array_equal(temp_coverage & full_mask, full_mask):
                    current_blocks.pop(i)
                    changed = True
                    break
        
        return current_blocks

    # Greedy with Sampling
    num_restarts = 0
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        uncovered = full_mask.copy()
        blocks_used = []
        
        while True:
            # Find random uncovered triple
            # We can pick a random word with non-zero bits, then a random bit in it
            target_t_idx = -1
            while target_t_idx == -1:
                w_idx = random.randint(0, num_words - 1)
                if uncovered[w_idx] != 0:
                    # Pick a random set bit in this word
                    bits = int(uncovered[w_idx])
                    # Get list of set bit positions
                    # Or just pick a random position and check?
                    # Faster: convert to list of indices
                    # Since 64 bits, we can just iterate
                    # Or use a trick:
                    # random_pos = random.randint(0, 63)
                    # if (bits >> random_pos) & 1:
                    #     target_t_idx = w_idx * 64 + random_pos
                    # This might loop if few bits set.
                    
                    # Better: get the list of set bits
                    set_bits = []
                    for b in range(64):
                        if (bits >> b) & 1:
                            set_bits.append(b)
                    if set_bits:
                        target_t_idx = w_idx * 64 + random.choice(set_bits)
                    else:
                        target_t_idx = -1
                        break
                else:
                    target_t_idx = -1
                    # Retry with another word? 
                    # If we picked a word with 0 bits, we should pick another.
                    # The while loop above handles this by retrying if target_t_idx remains -1?
                    # No, if uncovered[w_idx] is 0, we set target_t_idx = -1 and break the inner while?
                    # No, the inner while condition is `target_t_idx == -1`.
                    # If we pick a word with 0 bits, we set target_t_idx = -1 and continue the while loop?
                    # Yes, it will pick a new w_idx.
                    
            if target_t_idx == -1:
                break # All covered
                
            candidates = triple_to_blocks[target_t_idx]
            
            # Sample candidates
            # If len(candidates) < 200, use all
            if len(candidates) <= 200:
                sampled = candidates
            else:
                sampled = random.sample(candidates, 200)
                
            best_gain = -1
            best_b_idx = -1
            
            for b_idx in sampled:
                mask = block_bits[b_idx] & uncovered
                # Count bits
                gain = 0
                for w in mask:
                    gain += int(w).bit_count()
                    
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
                    
            if best_b_idx == -1:
                # Should not happen if uncovered is not empty
                break
                
            blocks_used.append(best_b_idx)
            uncovered &= ~block_bits[best_b_idx]
            
        if target_t_idx == -1: # Covered
            reduced = reduce_blocks(blocks_used)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced
                
        num_restarts += 1

    if best_blocks is None:
        # Fallback: return a valid covering from the last run (even if not reduced)
        # Or just return empty? No.
        # If we never completed a run, we have a problem.
        # But the first run should complete quickly.
        # Let's assume best_blocks is set.
        pass

    result = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return result
