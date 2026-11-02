"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 110.0  # Safety margin
    
    # 1. Precompute all t-subsets and map them to indices
    t_sets = list(itertools.combinations(range(v), t))
    t_set_index = {ts: i for i, ts in enumerate(t_sets)}
    num_t_sets = len(t_sets)
    
    # 2. Precompute all k-blocks and the t-sets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Use sets for faster intersection
    block_cover_sets = []
    for block in all_blocks:
        indices = [t_set_index[ts] for ts in itertools.combinations(block, t)]
        block_cover_sets.append(set(indices))
    
    # Precompute t_set_to_blocks
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx, cover_set in enumerate(block_cover_sets):
        for ts_idx in cover_set:
            t_set_to_blocks[ts_idx].append(b_idx)
    
    best_solution = None
    best_size = float('inf')
    
    # Strategies: (beam_width, seed)
    # Beam width 1 is the original random pivot.
    # Higher beam widths consider multiple pivots and pick the best block among them.
    strategies = [
        (1, 42),
        (5, 101),
        (10, 102),
        (20, 103),
        (50, 104),
        (10, 201),
        (20, 202),
        (50, 203),
    ]
    
    for beam_width, seed in strategies:
        if time.time() > deadline - 10.0:
            break
            
        sol = _greedy_cover_beam(v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets, t_set_to_blocks, deadline, 
                                 rng_seed=seed, beam_width=beam_width)
        if sol:
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
                
    if best_solution is None:
        return []
        
    return best_solution

def _greedy_cover_beam(v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets, t_set_to_blocks, deadline, rng_seed=42, beam_width=1):
    rng = random.Random(rng_seed)
    uncovered = set(range(num_t_sets))
    selected_blocks = []
    
    while uncovered:
        if time.time() > deadline - 2.0:
            return None
        
        candidate_blocks = set()
        
        # Pick beam_width random t-sets from uncovered
        if len(uncovered) <= beam_width:
            targets = list(uncovered)
        else:
            targets = rng.sample(list(uncovered), beam_width)
            
        for target_tset_idx in targets:
            # Find best block covering this target
            best_block_idx = -1
            max_gain = 0
            
            for b_idx in t_set_to_blocks[target_tset_idx]:
                gain = len(block_cover_sets[b_idx] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                candidate_blocks.add(best_block_idx)
        
        # Pick the best block among candidates
        final_best_block_idx = -1
        final_max_gain = 0
        
        for b_idx in candidate_blocks:
            gain = len(block_cover_sets[b_idx] & uncovered)
            if gain > final_max_gain:
                final_max_gain = gain
                final_best_block_idx = b_idx
        
        if final_best_block_idx == -1:
            break
            
        selected_blocks.append(all_blocks[final_best_block_idx])
        
        # Update uncovered
        uncovered -= block_cover_sets[final_best_block_idx]
        
    return selected_blocks
# EVOLVE-BLOCK-END
