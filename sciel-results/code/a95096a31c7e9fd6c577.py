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
    strategies = [
        (1, 42),
        (5, 101),
        (10, 102),
        (20, 103),
        (50, 104),
        (10, 201),
        (20, 202),
        (50, 203),
        (100, 301),
        (200, 302),
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
                
    # Try local improvement: replace blocks to reduce count
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

def _local_improve(solution, block_cover_sets, t_set_to_blocks, num_t_sets, deadline):
    """Try to remove redundant blocks or replace pairs with single blocks."""
    if not solution:
        return None
    
    # Convert solution to frozensets for fast lookup
    sol_sets = [frozenset(b) for b in solution]
    sol_cover = set()
    for s in sol_sets:
        sol_cover |= set(s)
    
    # First pass: remove any block that doesn't cover a unique t-set
    essential = [True] * len(sol_sets)
    for i, s in enumerate(sol_sets):
        # Check if this block covers any t-set not covered by others
        others_cover = set()
        for j, sj in enumerate(sol_sets):
            if j != i:
                others_cover |= set(sj)
        unique = set(s) - others_cover
        if not unique:
            essential[i] = False
    
    # Remove non-essential blocks
    new_sol = [s for s, e in zip(sol_sets, essential) if e]
    
    if len(new_sol) == len(sol_sets):
        return None  # No improvement
    
    # Second pass: try to replace two blocks with one
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        n = len(new_sol)
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            for j in range(i+1, n):
                # Check if union of blocks i and j can be covered by a single block
                union = set(new_sol[i]) | set(new_sol[j])
                # Find if any single block covers all of union
                for b_idx in range(len(t_set_to_blocks[0])):
                    if set(block_cover_sets[b_idx]) >= union:
                        # Replace i and j with this block
                        new_sol = [s for idx, s in enumerate(new_sol) if idx not in (i, j)]
                        new_sol.append(list(block_cover_sets[b_idx]))
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    if len(new_sol) >= len(solution):
        return None
    
    return [list(s) for s in new_sol]
# EVOLVE-BLOCK-END
