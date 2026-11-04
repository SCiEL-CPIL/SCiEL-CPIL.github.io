import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin under the 150s kill
    
    # Precomputation
    # 1. List of all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_to_idx = {c: i for i, c in enumerate(all_t)}
    
    # 2. List of all k-blocks
    all_k = list(itertools.combinations(range(v), k))
    n_k = len(all_k)
    
    # 3. Map each t-subset to the list of k-blocks containing it
    blocks_by_tsub = [[] for _ in range(n_t)]
    
    # 4. Map each k-block to the list of t-subsets it contains
    block_tsubs = [[] for _ in range(n_k)]
    
    # Efficient Precomputation
    # Iterate through all k-blocks and find the t-subsets they contain
    for j, block in enumerate(all_k):
        for tsub in itertools.combinations(block, t):
            idx = t_to_idx[tsub]
            blocks_by_tsub[idx].append(j)
            block_tsubs[j].append(idx)
            
    def greedy_construct(seed):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        used_blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None
                
            # Pick a random uncovered t-subset to guide the search
            # Converting to list is O(N), but N is small (<= 2024)
            current_tsub = rng.choice(list(uncovered))
            
            best_block = -1
            best_gain = -1
            
            # Evaluate all blocks containing current_tsub
            for b_idx in blocks_by_tsub[current_tsub]:
                # Calculate how many uncovered t-subsets this block covers
                gain = 0
                for t_idx in block_tsubs[b_idx]:
                    if t_idx in uncovered:
                        gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                # Should not happen
                break
                
            used_blocks.append(all_k[best_block])
            
            # Update uncovered set
            for t_idx in block_tsubs[best_block]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    
        return used_blocks

    best_blocks = None
    attempt = 0
    
    while time.time() < deadline - 5.0:
        blocks = greedy_construct(attempt)
        if blocks is not None:
            if best_blocks is None or len(blocks) < len(best_blocks):
                best_blocks = blocks
        attempt += 1
        
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(range(n_t))
        used_blocks = []
        while uncovered:
            current_tsub = min(uncovered)
            best_block = -1
            best_gain = -1
            for b_idx in blocks_by_tsub[current_tsub]:
                gain = 0
                for t_idx in block_tsubs[b_idx]:
                    if t_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            used_blocks.append(all_k[best_block])
            for t_idx in block_tsubs[best_block]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
        best_blocks = used_blocks

    return best_blocks
