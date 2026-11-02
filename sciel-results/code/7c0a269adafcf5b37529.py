import itertools
import time
import random
import numpy as np
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    n_t_subsets = len(all_t_subsets)
    
    # Helper to get triple indices for a block
    def get_block_triple_indices(block):
        # block is a list or tuple of points
        return [t_subset_to_idx[tuple(sorted(sub))] for sub in itertools.combinations(block, t)]
    
    # Initial Greedy Construction
    coverage = np.zeros(n_t_subsets, dtype=np.int32)
    blocks = []
    
    uncovered_count = n_t_subsets
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        # Find an uncovered triple
        min_idx = np.argmin(coverage)
        if coverage[min_idx] > 0:
            break # All covered
            
        target_triple = all_t_subsets[min_idx]
        
        # Generate candidate blocks containing this triple
        remaining_points = [p for p in range(v) if p not in target_triple]
        
        best_block = None
        best_gain = -1
        
        # Determine candidate strategy
        n_candidates = math.comb(v - t, k - t)
        
        if n_candidates > 5000:
            # Use random sampling to speed up
            candidates = []
            for _ in range(200):
                combo = random.sample(remaining_points, k - t)
                block = tuple(sorted(target_triple + combo))
                candidates.append(block)
        else:
            candidates = [tuple(sorted(target_triple + combo)) for combo in itertools.combinations(remaining_points, k - t)]
        
        for block in candidates:
            # Calculate gain: number of uncovered triples in this block
            gain = 0
            for sub in itertools.combinations(block, t):
                idx = t_subset_to_idx[sub]
                if coverage[idx] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = list(block)
                
        if best_block is None:
            break
            
        # Add the best block
        blocks.append(best_block)
        
        # Update coverage
        block_indices = get_block_triple_indices(best_block)
        for idx in block_indices:
            if coverage[idx] == 0:
                uncovered_count -= 1
            coverage[idx] += 1

    # Local Search / Optimization Phase
    
    def try_remove_redundant(blocks, coverage):
        removed_any = False
        new_blocks = []
        for b in blocks:
            b_indices = get_block_triple_indices(b)
            redundant = True
            for idx in b_indices:
                if coverage[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                for idx in b_indices:
                    coverage[idx] -= 1
                removed_any = True
            else:
                new_blocks.append(b)
        
        return new_blocks, removed_any

    # Initial cleanup
    blocks, _ = try_remove_redundant(blocks, coverage)
    
    best_blocks = blocks[:]
    best_len = len(blocks)
    
    # Local Search Loop
    while time.time() < deadline:
        # Phase 1: Remove Redundant
        blocks, removed = try_remove_redundant(blocks, coverage)
        if len(blocks) < best_len:
            best_blocks = blocks[:]
            best_len = len(blocks)
        
        # Phase 2: Swap
        if not blocks:
            break
            
        idx_to_swap = random.randint(0, len(blocks) - 1)
        old_block = blocks[idx_to_swap]
        
        # Generate a candidate new block
        old_triples = list(itertools.combinations(old_block, t))
        target_triple = random.choice(old_triples)
        remaining_points = [p for p in range(v) if p not in target_triple]
        
        # Random block containing target_triple
        new_block = list(target_triple)
        new_block.extend(random.sample(remaining_points, k - t))
        new_block.sort()
        
        # Evaluate the swap on a copy
        sim_blocks = blocks[:]
        sim_blocks[idx_to_swap] = new_block
        sim_coverage = coverage.copy()
        
        # Apply swap to sim_coverage
        old_indices = get_block_triple_indices(old_block)
        new_indices = get_block_triple_indices(new_block)
        for idx in old_indices:
            sim_coverage[idx] -= 1
        for idx in new_indices:
            sim_coverage[idx] += 1
            
        # Redundancy removal on sim_blocks
        final_sim_blocks = []
        for b in sim_blocks:
            b_indices = get_block_triple_indices(b)
            redundant = True
            for idx in b_indices:
                if sim_coverage[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in b_indices:
                    sim_coverage[idx] -= 1
            else:
                final_sim_blocks.append(b)
                
        new_len = len(final_sim_blocks)
        
        # Acceptance
        if new_len < best_len:
            blocks = final_sim_blocks
            coverage = sim_coverage
            best_blocks = blocks[:]
            best_len = new_len
        elif new_len == best_len and random.random() < 0.3:
            # Accept to explore
            blocks = final_sim_blocks
            coverage = sim_coverage
            best_blocks = blocks[:]
            
    return best_blocks
