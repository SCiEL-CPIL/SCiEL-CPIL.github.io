import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    # 110 seconds deadline to be safe
    deadline = start_time + 110.0
    
    # Precomputation
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    blocks_list = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks_list)
    
    block_masks = [0] * num_blocks
    block_triples = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(blocks_list):
        mask = 0
        b_tris = []
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
            b_tris.append(idx)
            triple_to_blocks[idx].append(b_idx)
        block_masks[b_idx] = mask
        block_triples[b_idx] = b_tris

    # Fallback bit_count for older Python
    if not hasattr(int, "bit_count"):
        def popcount(x):
            return bin(x).count("1")
    else:
        def popcount(x):
            return x.bit_count()

    def greedy_run(seed):
        random.seed(seed)
        covered_mask = 0
        used_blocks = []
        uncovered_indices = list(range(num_triples))
        
        while uncovered_indices:
            if time.time() > deadline:
                return None
            
            # Pick a random uncovered triple
            # We pick a random index from the current uncovered list
            # This list is small (< 2024)
            rand_pos = random.randint(0, len(uncovered_indices) - 1)
            tri_idx = uncovered_indices[rand_pos]
            
            best_gain = -1
            best_block_idx = -1
            
            # Find best block covering this triple
            # Iterate through all blocks that contain this triple
            for b_idx in triple_to_blocks[tri_idx]:
                # Calculate gain: number of new triples covered
                # new_covered = block_mask & ~covered_mask
                # gain = popcount(new_covered)
                
                # Optimization: If the block is already fully covered, gain is 0.
                # But we are picking an uncovered triple, so gain is at least 1.
                
                new_covered = block_masks[b_idx] & (~covered_mask)
                gain = popcount(new_covered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                elif gain == best_gain and gain > 0:
                    # Random tie breaking
                    if random.random() < 0.5:
                        best_block_idx = b_idx
                        
            if best_block_idx == -1:
                # Should not happen if uncovered_indices is correct
                return None
            
            used_blocks.append(best_block_idx)
            covered_mask |= block_masks[best_block_idx]
            
            # Update uncovered_indices
            # Keep only those not covered by the new mask
            # We can do this by checking each index in the list
            uncovered_indices = [i for i in uncovered_indices if not (covered_mask & (1 << i))]
            
        return used_blocks

    def polish(block_indices):
        current_blocks = list(block_indices)
        changed = True
        while changed:
            changed = False
            # Count coverage of each triple
            triple_cov = [0] * num_triples
            for b_idx in current_blocks:
                for t_idx in block_triples[b_idx]:
                    triple_cov[t_idx] += 1
            
            new_blocks = []
            for b_idx in current_blocks:
                can_remove = True
                for t_idx in block_triples[b_idx]:
                    if triple_cov[t_idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for t_idx in block_triples[b_idx]:
                        triple_cov[t_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
            
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    seed = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        raw_blocks = greedy_run(seed)
        if raw_blocks is not None:
            polished = polish(raw_blocks)
            if len(polished) < best_len:
                best_len = len(polished)
                best_blocks = polished
        seed += 1
        
    if best_blocks is None:
        # If no complete covering was found (should be very rare), 
        # we might return an empty list or a trivial one, but the task requires validity.
        # Given the small size, greedy should always find a solution quickly.
        # If we hit the deadline in the first run, it's a problem.
        # But 80 steps * 100k checks is fast.
        return []
        
    return [list(blocks_list[b_idx]) for b_idx in best_blocks]
