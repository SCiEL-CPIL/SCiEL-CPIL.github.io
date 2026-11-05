import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    # Parameters for the specific task: v=24, k=7, t=3
    # But the function signature is generic, so we handle v, k, t.
    # However, the precomputation strategy is highly dependent on small v.
    # If v is large, this won't work. But the task is specifically C(24,7,3).
    # We will assume v, k, t are the ones from the task description or small enough.
    # The prompt says: "The evaluator passes this task's parameters v, k and t."
    # So v=24, k=7, t=3.
    
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Precomputation
    # 1. Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 2. Generate all triples
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # 3. Map blocks to triples they cover
    # block_triples[j] = list of triple indices covered by block j
    block_triples = []
    # triple_blocks[i] = list of block indices covering triple i
    triple_blocks = [[] for _ in range(num_triples)]
    
    for j, block in enumerate(all_blocks):
        b_set = set(block)
        covered_triples = []
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            covered_triples.append(idx)
            triple_blocks[idx].append(j)
        block_triples.append(covered_triples)
        
    # Heuristic: Greedy with Randomization + Deletion Pass
    
    best_blocks = None
    best_size = float('inf')
    
    # Number of restarts depends on time, but let's try a fixed number or time-based loop
    restarts = 0
    
    while time.time() < deadline:
        restarts += 1
        # 1. Greedy Construction
        uncovered = np.ones(num_triples, dtype=bool)
        blocks = []
        
        # To speed up, we can pick a random uncovered triple at each step
        # instead of finding the block with max global gain (which is expensive to compute from scratch)
        # This is a "Greedy Set Cover" variant.
        
        while np.any(uncovered):
            # Find indices of uncovered triples
            uncov_indices = np.where(uncovered)[0]
            if len(uncov_indices) == 0:
                break
            
            # Pick a random uncovered triple
            # Picking one that is "hard" to cover (few blocks cover it) might be better?
            # Or just random to diversify.
            # Let's pick a random one.
            target_triple_idx = uncov_indices[random.randint(0, len(uncov_indices) - 1)]
            
            # Get all blocks covering this triple
            candidates = triple_blocks[target_triple_idx]
            
            best_block_idx = -1
            max_gain = -1
            
            # Evaluate candidates
            for b_idx in candidates:
                # Count how many uncovered triples this block covers
                # block_triples[b_idx] is a list of triple indices
                gain = 0
                for t_idx in block_triples[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1
                
                # Random tie-breaking
                if gain > max_gain or (gain == max_gain and (best_block_idx == -1 or random.random() < 0.5)):
                    max_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                # Should not happen if covering is possible
                break
                
            blocks.append(best_block_idx)
            
            # Update uncovered mask
            for t_idx in block_triples[best_block_idx]:
                uncovered[t_idx] = False
        
        # 2. Deletion Pass
        # Calculate coverage counts
        coverage = np.zeros(num_triples, dtype=int)
        for b_idx in blocks:
            for t_idx in block_triples[b_idx]:
                coverage[t_idx] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks:
                # Check if this block is essential
                # Essential if it covers any triple with coverage == 1
                essential = False
                for t_idx in block_triples[b_idx]:
                    if coverage[t_idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove block
                    for t_idx in block_triples[b_idx]:
                        coverage[t_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            
            blocks = new_blocks
        
        # 3. Check validity and size
        if len(blocks) < best_size:
            # Verify validity (optional, but good for safety)
            # The deletion pass ensures every triple has coverage >= 1 if we started with a valid cover
            # and only removed non-essential blocks.
            # But we must ensure the initial greedy cover was valid.
            # Greedy adds blocks until all are covered, so it is valid.
            
            # Convert indices to lists of points
            # We need to return list of blocks (lists of points)
            # current 'blocks' is a list of indices into all_blocks
            final_blocks = [list(all_blocks[b]) for b in blocks]
            
            # Sort points in each block for consistency (not strictly required but good)
            for b in final_blocks:
                b.sort()
                
            best_blocks = final_blocks
            best_size = len(blocks)
            
            # Early exit if we reach the theoretical lower bound?
            # Lower bound is 69. If we hit 69, we can stop.
            if best_size <= 69:
                break

    if best_blocks is None:
        # Fallback: return a simple valid cover if something went wrong
        # This should not happen
        return []
        
    return best_blocks
