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
        block_cover_sets.append(frozenset(indices))
    
    # Precompute t_set_to_blocks
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx, cover_set in enumerate(block_cover_sets):
        for ts_idx in cover_set:
            t_set_to_blocks[ts_idx].append(b_idx)
    
    best_solution = None
    best_size = float('inf')
    
    # Strategies: (beam_width, seed)
    # Dual-guided greedy: prioritize hard-to-cover t-sets
    strategies = [
        (1, 42),
        (3, 101),
        (5, 102),
        (10, 103),
        (20, 104),
        (5, 201),
        (10, 202),
        (20, 203),
        (50, 301),
        (100, 302),
    ]
    
    for beam_width, seed in strategies:
        if time.time() > deadline - 15.0:
            break
            
        sol = _greedy_cover_beam(v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets, t_set_to_blocks, deadline, 
                                 rng_seed=seed, beam_width=beam_width)
        if sol:
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
                
    # Try local improvement: remove redundant blocks
    if best_solution and time.time() < deadline - 20.0:
        improved = _local_improve(best_solution, block_cover_sets, t_set_to_blocks, num_t_sets, deadline)
        if improved and len(improved) < best_size:
            best_solution = improved
            best_size = len(improved)
        
    if best_solution is None:
        return []
        
    return best_solution

def _greedy_cover_beam(v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets, t_set_to_blocks, deadline, rng_seed=42, beam_width=1):
    rng = random.Random(rng_seed)
    uncovered = set(range(num_t_sets))
    selected_blocks = []
    
    # Dual weights: track how "hard" each t-set is to cover
    # Higher weight = more uncovered blocks that could cover it (easier) 
    # We want to prioritize t-sets with FEWER covering options (higher difficulty)
    # dual[i] = 1 / (number of blocks covering t-set i that haven't been selected)
    # Simplified: just track count of available blocks per t-set
    
    # Precompute initial block counts per t-set
    tset_block_counts = [len(t_set_to_blocks[i]) for i in range(num_t_sets)]
    # Track remaining blocks per t-set (decrement when a block is selected)
    remaining_counts = tset_block_counts[:]
    
    while uncovered:
        if time.time() > deadline - 2.0:
            return None
        
        candidate_blocks = set()
        
        # Pick beam_width t-sets from uncovered, weighted by difficulty
        # Difficulty = 1 / remaining_counts (higher for t-sets with fewer options)
        if len(uncovered) <= beam_width:
            targets = list(uncovered)
        else:
            # Sample with probability proportional to difficulty
            weights = [1.0 / max(remaining_counts[i], 1) for i in uncovered]
            # Normalize
            total_w = sum(weights)
            if total_w == 0:
                targets = rng.sample(list(uncovered), beam_width)
            else:
                probs = [w / total_w for w in weights]
                targets = rng.choices(list(uncovered), weights=probs, k=beam_width)
                # Deduplicate
                targets = list(set(targets))
                if len(targets) < beam_width:
                    # Fill with random
                    remaining_uncovered = list(uncovered - set(targets))
                    if remaining_uncovered:
                        targets.extend(rng.sample(remaining_uncovered, min(beam_width - len(targets), len(remaining_uncovered))))
            
        for target_tset_idx in targets:
            # Find best block covering this target, using weighted gain
            best_block_idx = -1
            max_weighted_gain = 0
            
            for b_idx in t_set_to_blocks[target_tset_idx]:
                # Weighted gain: sum of dual weights of uncovered t-sets covered by this block
                weighted_gain = 0.0
                for ts_idx in block_cover_sets[b_idx]:
                    if ts_idx in uncovered:
                        # Weight by difficulty of this t-set
                        weighted_gain += 1.0 / max(remaining_counts[ts_idx], 1)
                if weighted_gain > max_weighted_gain:
                    max_weighted_gain = weighted_gain
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                candidate_blocks.add(best_block_idx)
        
        # Pick the best block among candidates using weighted gain
        final_best_block_idx = -1
        final_max_weighted_gain = 0
        
        for b_idx in candidate_blocks:
            weighted_gain = 0.0
            for ts_idx in block_cover_sets[b_idx]:
                if ts_idx in uncovered:
                    weighted_gain += 1.0 / max(remaining_counts[ts_idx], 1)
            if weighted_gain > final_max_weighted_gain:
                final_max_weighted_gain = weighted_gain
                final_best_block_idx = b_idx
        
        if final_best_block_idx == -1:
            break
            
        selected_blocks.append(all_blocks[final_best_block_idx])
        
        # Update uncovered and remaining counts
        covered = block_cover_sets[final_best_block_idx]
        for ts_idx in covered:
            if ts_idx in uncovered:
                remaining_counts[ts_idx] -= 1
        uncovered -= covered
        
    return selected_blocks

def _local_improve(solution, block_cover_sets, t_set_to_blocks, num_t_sets, deadline):
    """Try to remove redundant blocks."""
    if not solution:
        return None
    
    # Convert solution to frozensets for fast lookup
    sol_sets = [frozenset(b) for b in solution]
    
    # First pass: remove any block that doesn't cover a unique t-set
    # Use efficient approach: for each t-set, count how many blocks cover it
    # A block is redundant if all its t-sets are covered by other blocks
    
    # Build t-set to solution-block-index mapping
    tset_to_sol_blocks = {}
    for i, s in enumerate(sol_sets):
        for ts_idx in s:
            if ts_idx not in tset_to_sol_blocks:
                tset_to_sol_blocks[ts_idx] = []
            tset_to_sol_blocks[ts_idx].append(i)
    
    # A block is essential if it's the only one covering at least one t-set
    essential = [True] * len(sol_sets)
    for ts_idx, block_indices in tset_to_sol_blocks.items():
        if len(block_indices) == 1:
            essential[block_indices[0]] = True
    
    # Remove non-essential blocks
    new_sol = [s for s, e in zip(sol_sets, essential) if e]
    
    if len(new_sol) == len(sol_sets):
        return None  # No improvement
    
    return [list(s) for s in new_sol]
# EVOLVE-BLOCK-END
