"""Improved covering design solver with multiple strategies and local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Use numpy for coverage tracking
    covered = np.zeros(n_subsets, dtype=np.int32)
    
    def add_block(block):
        """Add a block and update coverage. Returns number of newly covered subsets."""
        gain = 0
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            if covered[idx] == 0:
                covered[idx] = 1
                gain += 1
            else:
                covered[idx] += 1
        return gain
    
    def remove_block(block):
        """Remove a block from coverage. Returns True if valid (no subset becomes uncovered)."""
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            if covered[idx] <= 1:
                return False
            covered[idx] -= 1
        return True
    
    def is_valid_covering():
        return np.all(covered >= 1)
    
    def greedy_construct(weight_mode='uniform'):
        """Greedy construction with different weighting strategies."""
        nonlocal covered
        covered = np.zeros(n_subsets, dtype=np.int32)
        blocks = []
        uncovered = set(range(n_subsets))
        
        while uncovered:
            if time.time() > deadline - 20:
                break
            
            # Pick seed: random uncovered subset for diversity
            seed_idx = random.choice(list(uncovered))
            seed = list(all_t_subsets[seed_idx])
            block = seed[:]
            
            while len(block) < k:
                if time.time() > deadline - 15:
                    break
                
                best_point = None
                best_gain = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        idx = subset_to_idx[sub]
                        if idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None or best_gain <= 0:
                    for x in range(v):
                        if x not in block:
                            best_point = x
                            break
                
                block.append(best_point)
            
            if len(block) < k:
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.append(x)
            
            block.sort()
            blocks.append(block)
            
            # Update coverage
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                if covered[idx] == 0:
                    covered[idx] = 1
                    uncovered.discard(idx)
                else:
                    covered[idx] += 1
            
            if not uncovered:
                break
        
        return blocks
    
    # Try multiple greedy strategies
    best_blocks = None
    best_count = float('inf')
    
    strategies = ['uniform', 'inverse_freq', 'random_seed']
    
    for strategy in strategies:
        if time.time() > deadline - 30:
            break
        
        random.seed(hash(strategy) % 10000)
        blocks = greedy_construct(strategy)
        
        # Verify and count
        if is_valid_covering():
            count = len(blocks)
            if count < best_count:
                best_count = count
                best_blocks = blocks[:]
    
    # If no valid covering yet, do emergency fill
    if best_blocks is None or not is_valid_covering():
        best_blocks = greedy_construct('uniform')
        # Emergency fill
        while not is_valid_covering() and time.time() < deadline - 5:
            uncovered_idx = np.where(covered == 0)[0]
            if len(uncovered_idx) == 0:
                break
            seed_idx = uncovered_idx[0]
            seed = list(all_t_subsets[seed_idx])
            block = seed[:]
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            best_blocks.append(block)
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                covered[idx] += 1
    
    # Local search: try to remove redundant blocks
    if time.time() < deadline - 10:
        best_blocks = _local_search_improved(best_blocks, v, k, t, deadline, covered, subset_to_idx)
    
    return best_blocks


def _local_search_improved(blocks, v, k, t, deadline, covered, subset_to_idx):
    """Aggressive local search: remove redundant blocks iteratively."""
    # Work on a copy of covered
    local_covered = covered.copy()
    local_blocks = blocks[:]
    
    # Rebuild coverage from scratch to ensure consistency
    local_covered = np.zeros(len(subset_to_idx), dtype=np.int32)
    for block in local_blocks:
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            local_covered[idx] += 1
    
    # Iteratively remove non-critical blocks
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(local_blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            # Check if block i is critical
            is_critical = False
            for sub in itertools.combinations(local_blocks[i], t):
                idx = subset_to_idx[sub]
                if local_covered[idx] == 1:
                    is_critical = True
                    break
            
            if not is_critical:
                # Remove block i
                for sub in itertools.combinations(local_blocks[i], t):
                    idx = subset_to_idx[sub]
                    local_covered[idx] -= 1
                local_blocks.pop(i)
                changed = True
    
    return local_blocks
# EVOLVE-BLOCK-END
