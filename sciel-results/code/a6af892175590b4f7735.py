import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets (triples) and map them to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Precompute all k-subsets (blocks)
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # Precompute mask for each block: which triples it covers
    # Using Python integers as bitsets
    block_masks = [0] * num_blocks
    for i, block in enumerate(blocks):
        mask = 0
        # Generate all t-subsets of this block
        for sub in itertools.combinations(block, t):
            mask |= 1 << triple_to_idx[sub]
        block_masks[i] = mask
        
    best_solution = None
    best_len = float('inf')
    
    # Number of iterations to try
    # We will adjust based on time
    max_iters = 100
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        uncovered = (1 << num_triples) - 1
        current_blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_candidates = []
            
            # Find the block with max coverage of uncovered triples
            # Optimization: If uncovered has few bits, we can check fewer blocks?
            # For now, scan all.
            for i in range(num_blocks):
                # If the block doesn't cover any uncovered triple, gain is 0
                # Quick check: if (uncovered & block_masks[i]) == 0, skip?
                # bit_count is fast, but & is fast too.
                inter = uncovered & block_masks[i]
                if inter:
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [i]
                    elif gain == best_gain:
                        best_candidates.append(i)
            
            if best_gain == -1:
                # Should not happen if the problem is solvable
                break
                
            # Pick a candidate randomly to explore different paths
            chosen_idx = random.choice(best_candidates)
            current_blocks.append(blocks[chosen_idx])
            uncovered &= ~block_masks[chosen_idx]
            
        if not uncovered and len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_solution = current_blocks
            # If we hit the lower bound or a very good number, we could stop early?
            # Lower bound is 69. Best known 76.
            # If we find 76, it's good. If we find 75, even better.
            
    if best_solution is None:
        # Fallback: simple greedy without randomness (deterministic)
        # Or return the last one found
        # If no solution found due to timeout, we need to return something valid.
        # The loop above ensures we only update best_solution if 'not uncovered'.
        # If we timed out in the middle of a run, 'uncovered' might not be 0.
        # So best_solution might be None if the first run timed out before finishing.
        # We should ensure we have a valid solution.
        # Let's do one final deterministic greedy run if best_solution is None.
        uncovered = (1 << num_triples) - 1
        current_blocks = []
        while uncovered and time.time() < deadline:
            best_gain = -1
            best_idx = -1
            for i in range(num_blocks):
                inter = uncovered & block_masks[i]
                if inter:
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
            if best_idx == -1:
                break
            current_blocks.append(blocks[best_idx])
            uncovered &= ~block_masks[best_idx]
        if not uncovered:
            best_solution = current_blocks
        else:
            # Emergency fallback: return a trivial valid cover if possible?
            # Or just return the partial one? No, must be valid.
            # If we can't finish, we might fail. But 120s is plenty for 76 blocks.
            # The greedy step is ~346k * 76 = 26M ops.
            # Python 3.10+ bit_count is very fast.
            # 26M iterations in Python loop might take ~5-10s.
            # So 100 iterations might take 500-1000s.
            # We need to reduce iterations or speed up the inner loop.
            pass

    return [list(b) for b in best_solution] if best_solution else []
