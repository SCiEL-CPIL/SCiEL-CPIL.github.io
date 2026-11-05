import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    # State
    blocks = []
    coverage = [0] * num_subsets
    
    # Helper: Add a block and update coverage
    def add_block(block):
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            coverage[subset_to_idx[ts]] += 1
            
    # Helper: Remove a block and update coverage
    def remove_block(idx):
        block = blocks[idx]
        for ts in itertools.combinations(block, t):
            coverage[subset_to_idx[ts]] -= 1
        blocks.pop(idx)
        
    # Helper: Find redundant blocks
    def find_redundant_blocks():
        redundant_indices = []
        for i, b in enumerate(blocks):
            is_redundant = True
            for ts in itertools.combinations(b, t):
                if coverage[subset_to_idx[ts]] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_indices.append(i)
        return redundant_indices
        
    # Helper: Greedy construction of a single block to cover a specific subset
    # and maximize coverage of low-coverage subsets
    def build_block(start_subset):
        block = list(start_subset)
        block_set = set(block)
        
        while len(block) < k:
            best_p = -1
            best_score = -1
            for p in range(v):
                if p in block_set:
                    continue
                score = 0
                # Score based on how many low-coverage subsets this point helps
                # Specifically, subsets that contain p and the rest of the block (t-1 elements)
                # If we add p, we cover subsets (comb + p) where comb is (t-1)-subset of block
                for comb in itertools.combinations(block, t-1):
                    ts = tuple(sorted(comb + (p,)))
                    idx = subset_to_idx[ts]
                    # Prioritize subsets with lower coverage
                    # If coverage is 1, covering it might allow redundancy removal
                    # If coverage is 0 (shouldn't happen if we maintain validity), it's critical
                    score += (2 - min(coverage[idx], 1)) # Heuristic weight
                if score > best_score:
                    best_score = score
                    best_p = p
            if best_p == -1:
                # Fallback
                for p in range(v):
                    if p not in block_set:
                        best_p = p
                        break
            block.append(best_p)
        block.sort()
        return block

    # Initial Construction: Greedy
    # Start with empty coverage
    # We need to cover all subsets.
    # Simple greedy: pick uncovered subset, build block.
    uncovered_indices = set(range(num_subsets))
    
    while uncovered_indices:
        # Pick random uncovered
        idx = random.choice(list(uncovered_indices))
        start_sub = all_t_subsets[idx]
        new_block = build_block(start_sub)
        add_block(new_block)
        
        # Update uncovered
        # This is O(N) per block. N is number of blocks.
        # We can just recalculate coverage and find zeros?
        # Or update uncovered set directly.
        for ts in itertools.combinations(new_block, t):
            i = subset_to_idx[ts]
            if coverage[i] == 1: # Just became covered
                uncovered_indices.discard(i)
                
    best_blocks = list(blocks)
    best_size = len(blocks)
    
    # Local Search Loop
    while time.time() < deadline:
        # 1. Add a block
        # Pick a subset with low coverage (1 or 2) to try to create redundancy
        # If we pick a subset with coverage 1, and add a block covering it,
        # its coverage becomes 2. Then the old block covering it *might* become redundant.
        
        # Select target subset:
        # Randomly pick from subsets with coverage == 1
        low_cov_indices = [i for i, c in enumerate(coverage) if c == 1]
        if not low_cov_indices:
            # All covered >= 2, try to find redundancy
            redundant = find_redundant_blocks()
            for r_idx in redundant:
                remove_block(r_idx)
            continue
            
        target_idx = random.choice(low_cov_indices)
        target_sub = all_t_subsets[target_idx]
        
        new_block = build_block(target_sub)
        add_block(new_block)
        
        # 2. Remove redundant blocks
        redundant = find_redundant_blocks()
        # Remove in reverse order to keep indices valid
        for r_idx in sorted(redundant, reverse=True):
            remove_block(r_idx)
            
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
            
    return best_blocks
