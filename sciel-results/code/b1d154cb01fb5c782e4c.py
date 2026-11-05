import itertools
import random
import math
import time
import numpy as np

def construct(v, k, t):
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all blocks and their covered subset indices
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
    # 1. Initialization: Randomized Greedy Set Cover
    # Precompute reverse index: subset -> list of block indices
    blocks_covering_subset = [[] for _ in range(n_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)
            
    random.seed(42)
    selected = []
    cov_count = np.zeros(n_subsets, dtype=np.int32)
    
    while True:
        # Find uncovered subsets
        uncovered_idx = np.where(cov_count == 0)[0]
        if len(uncovered_idx) == 0:
            break
        
        # Pick a random uncovered subset
        target_subset = random.choice(uncovered_idx)
        
        # Pick the block from candidates that covers the most OTHER uncovered subsets
        candidates = blocks_covering_subset[target_subset]
        best_b = -1
        best_gain = -1
        
        for b_idx in candidates:
            gain = 0
            for s in block_subsets[b_idx]:
                if cov_count[s] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_b = b_idx
                
        selected.append(best_b)
        for s in block_subsets[best_b]:
            cov_count[s] += 1
            
    # 2. Local Search: Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            b_idx = selected[i]
            redundant = True
            for s in block_subsets[b_idx]:
                if cov_count[s] == 1:
                    redundant = False
                    break
            
            if redundant:
                for s in block_subsets[b_idx]:
                    cov_count[s] -= 1
                selected.pop(i)
                changed = True
                break # Restart loop
                
    # 3. Iterated Local Search (Swaps)
    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    selected_set = set(selected)
    
    while time.time() - start_time < max_time:
        if len(selected) == 0: break
        
        i = random.randint(0, len(selected) - 1)
        old_b = selected[i]
        
        new_b = random.randint(0, n_blocks - 1)
        
        if new_b == old_b or new_b in selected_set:
            continue
            
        # Check validity of swap
        # Critical subsets are those in old_b with count 1
        critical = []
        for s in block_subsets[old_b]:
            if cov_count[s] == 1:
                critical.append(s)
                
        if not critical:
            # Valid swap
            for s in block_subsets[old_b]:
                cov_count[s] -= 1
            for s in block_subsets[new_b]:
                cov_count[s] += 1
            selected[i] = new_b
            selected_set.remove(old_b)
            selected_set.add(new_b)
            
            # Try to remove redundant blocks
            changed = True
            while changed:
                changed = False
                for j in range(len(selected)):
                    b_idx = selected[j]
                    redundant = True
                    for s in block_subsets[b_idx]:
                        if cov_count[s] == 1:
                            redundant = False
                            break
                    if redundant:
                        for s in block_subsets[b_idx]:
                            cov_count[s] -= 1
                        selected.pop(j)
                        selected_set.remove(b_idx)
                        changed = True
                        break
                        
        else:
            # Check if new_b covers all critical subsets
            new_b_set = set(block_subsets[new_b])
            if all(s in new_b_set for s in critical):
                # Valid swap
                for s in block_subsets[old_b]:
                    cov_count[s] -= 1
                for s in block_subsets[new_b]:
                    cov_count[s] += 1
                selected[i] = new_b
                selected_set.remove(old_b)
                selected_set.add(new_b)
                
                # Try to remove redundant blocks
                changed = True
                while changed:
                    changed = False
                    for j in range(len(selected)):
                        b_idx = selected[j]
                        redundant = True
                        for s in block_subsets[b_idx]:
                            if cov_count[s] == 1:
                                redundant = False
                                break
                        if redundant:
                            for s in block_subsets[b_idx]:
                                cov_count[s] -= 1
                            selected.pop(j)
                            selected_set.remove(b_idx)
                            changed = True
                            break
                            
    # Final cleanup: Remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            b_idx = selected[i]
            redundant = True
            for s in block_subsets[b_idx]:
                if cov_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                for s in block_subsets[b_idx]:
                    cov_count[s] -= 1
                selected.pop(i)
                changed = True
                break
                
    blocks = [list(all_blocks[i]) for i in selected]
    return blocks
