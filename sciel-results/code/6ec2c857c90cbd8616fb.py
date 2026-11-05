import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_covers[i]: list of subset indices covered by block i
    block_covers = [None] * n_blocks
    # subset_to_blocks[j]: set of block indices covering subset j
    subset_to_blocks = [set() for _ in range(n_subsets)]
    
    for i, block in enumerate(all_blocks):
        covered = []
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            covered.append(idx)
            subset_to_blocks[idx].add(i)
        block_covers[i] = covered

    # 2. Greedy Initialization
    cover_counts = np.zeros(n_subsets, dtype=np.int32)
    current_blocks = set()
    
    # Greedy: pick block covering most uncovered subsets
    for _ in range(200):
        if time.time() > deadline:
            break
        
        best_block = -1
        best_gain = 0
        
        for i in range(n_blocks):
            if i in current_blocks:
                continue
            gain = 0
            for j in block_covers[i]:
                if cover_counts[j] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = i
        
        if best_gain == 0:
            break
            
        current_blocks.add(best_block)
        for j in block_covers[best_block]:
            cover_counts[j] += 1
            
        if np.all(cover_counts > 0):
            break

    best_size = len(current_blocks)
    best_blocks = set(current_blocks)
    
    # 3. Local Search
    random.seed(42)
    
    while time.time() < deadline:
        # Try removal
        removed = False
        blocks_list = list(current_blocks)
        random.shuffle(blocks_list)
        
        for b in blocks_list:
            if time.time() > deadline:
                break
                
            redundant = True
            for j in block_covers[b]:
                if cover_counts[j] <= 1:
                    redundant = False
                    break
            
            if redundant:
                for j in block_covers[b]:
                    cover_counts[j] -= 1
                current_blocks.remove(b)
                removed = True
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = set(current_blocks)
                break
        
        if not removed:
            # Try swap
            if not current_blocks:
                break
            b1 = random.choice(list(current_blocks))
            
            # Find subsets that become uncovered if b1 removed
            uncovered_if_removed = []
            for j in block_covers[b1]:
                if cover_counts[j] == 1:
                    uncovered_if_removed.append(j)
                    
            if not uncovered_if_removed:
                continue

            # Find candidate blocks covering ALL uncovered_if_removed
            candidates = set(subset_to_blocks[uncovered_if_removed[0]])
            for s in uncovered_if_removed[1:]:
                candidates.intersection_update(subset_to_blocks[s])
                
            # Remove current blocks from candidates
            candidates.difference_update(current_blocks)
            
            if candidates:
                b2 = random.choice(list(candidates))
                
                # Perform swap
                for j in block_covers[b1]:
                    cover_counts[j] -= 1
                current_blocks.remove(b1)
                
                for j in block_covers[b2]:
                    cover_counts[j] += 1
                current_blocks.add(b2)
                
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = set(current_blocks)

    return [list(all_blocks[i]) for i in sorted(best_blocks)]
