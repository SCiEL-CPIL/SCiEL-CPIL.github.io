import itertools
import numpy as np
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using randomized greedy search.
    """
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    
    # M[x, i] = 1.0 if point x is in subset i
    M = np.zeros((v, num_subsets), dtype=np.float32)
    for i, s in enumerate(all_subsets):
        for x in s:
            M[x, i] = 1.0
            
    def run_greedy(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_subsets, dtype=bool)
        blocks = []
        
        while uncovered.any():
            block = []
            for _ in range(k):
                if len(block) >= t - 1:
                    block_mask = np.zeros(v, dtype=np.float32)
                    for x in block:
                        block_mask[x] = 1.0
                    
                    counts = M.T.dot(block_mask)
                    target_mask = uncovered & (counts == t - 1)
                    
                    if target_mask.any():
                        gains = M.dot(target_mask)
                    else:
                        gains = M.dot(uncovered)
                else:
                    gains = M.dot(uncovered)
                
                for x in block:
                    gains[x] = -1.0
                    
                max_gain = np.max(gains)
                if max_gain < 0:
                    break
                    
                candidates = np.where(gains == max_gain)[0]
                x = candidates[rng.randint(0, len(candidates) - 1)]
                block.append(int(x))
                
            block.sort()
            blocks.append(block)
            
            block_mask = np.zeros(v, dtype=np.float32)
            for x in block:
                block_mask[x] = 1.0
            
            counts = M.T.dot(block_mask)
            covered_now = (counts == t)
            uncovered &= ~covered_now
            
        return blocks

    start_time = time.time()
    deadline = 110.0
    
    best_blocks = None
    best_len = float('inf')
    seed = 0
    
    while True:
        current_time = time.time()
        if current_time > deadline:
            break
        
        blocks = run_greedy(seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
    if best_blocks is None:
        return []
        
    return best_blocks
