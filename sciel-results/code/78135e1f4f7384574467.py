import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate all blocks and triples
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_triples = list(itertools.combinations(points, t))
    
    n_blocks = len(all_blocks)
    n_triples = len(all_triples)
    
    # Map triples to indices
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    
    # Build coverage matrix M: M[i, j] = 1 if block i covers triple j
    # Using uint8 to save memory and allow fast summing
    M = np.zeros((n_blocks, n_triples), dtype=np.uint8)
    
    # Fill M
    for i, block in enumerate(all_blocks):
        for tr in itertools.combinations(block, t):
            idx = triple_to_idx[tr]
            M[i, idx] = 1
            
    # Initial gain: number of triples covered by each block
    gain = M.sum(axis=1).astype(np.int32)
    
    # covered: boolean array for triples
    covered = np.zeros(n_triples, dtype=bool)
    
    # used: boolean array for blocks
    used = np.zeros(n_blocks, dtype=bool)
    
    selected_blocks = []
    
    while True:
        if time.time() > deadline:
            # If we run out of time, return what we have (will be invalid)
            break
            
        # Find the block with max gain among unused
        valid_gain = gain.copy()
        valid_gain[used] = -1
        
        best_idx = np.argmax(valid_gain)
        
        if valid_gain[best_idx] <= 0:
            break
            
        # Select best_idx
        used[best_idx] = True
        selected_blocks.append(all_blocks[best_idx])
        
        # Determine newly covered triples
        block_coverage = M[best_idx]
        newly_covered_mask = block_coverage & (~covered)
        newly_covered_indices = np.where(newly_covered_mask)[0]
        
        if len(newly_covered_indices) == 0:
            continue
            
        # Update covered
        covered[newly_covered_indices] = True
        
        # Update gain for all blocks
        # Subtract the count of newly covered triples for each block
        for idx in newly_covered_indices:
            gain -= M[:, idx]
            
    return [list(b) for b in selected_blocks]
