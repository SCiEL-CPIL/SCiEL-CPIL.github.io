import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # 1. Generate all targets (t-subsets) and blocks (k-subsets)
    points = range(v)
    targets = list(itertools.combinations(points, t))
    blocks = list(itertools.combinations(points, k))
    
    num_targets = len(targets)
    num_blocks = len(blocks)
    
    # Map targets to indices
    target_index = {t: i for i, t in enumerate(targets)}
    
    # 2. Precompute coverage relationships
    # block_to_targets[j] = list of target indices covered by block j
    block_to_targets = [[] for _ in range(num_blocks)]
    # target_to_blocks[i] = list of block indices covering target i
    target_to_blocks = [[] for _ in range(num_targets)]
    
    for j, block in enumerate(blocks):
        for target in itertools.combinations(block, t):
            i = target_index[target]
            block_to_targets[j].append(i)
            target_to_blocks[i].append(j)
            
    # 3. Greedy Set Cover
    uncovered = [True] * num_targets
    counts = [0] * num_blocks
    
    # Initialize counts: each block covers C(k, t) targets
    # Since all are initially uncovered, counts[j] = len(block_to_targets[j])
    for j in range(num_blocks):
        counts[j] = len(block_to_targets[j])
        
    selected_blocks = []
    covered_count = 0
    
    while covered_count < num_targets:
        if time.time() > deadline:
            # If we run out of time, return what we have? 
            # The problem requires a valid covering. 
            # If we stop early, it's invalid. 
            # However, the greedy algorithm is very fast for this size.
            # 150 iterations * 16000 updates = 2.4M ops. Should take < 1s.
            pass
        
        # Find block with max count
        max_count = -1
        best_block_idx = -1
        for j in range(num_blocks):
            if counts[j] > max_count:
                max_count = counts[j]
                best_block_idx = j
                
        if best_block_idx == -1 or max_count == 0:
            # Should not happen if a covering exists
            break
            
        selected_blocks.append(blocks[best_block_idx])
        
        # Update counts
        for i in block_to_targets[best_block_idx]:
            if uncovered[i]:
                uncovered[i] = False
                covered_count += 1
                # Update all blocks covering this target
                for k in target_to_blocks[i]:
                    counts[k] -= 1
                    
    return [list(b) for b in selected_blocks]
