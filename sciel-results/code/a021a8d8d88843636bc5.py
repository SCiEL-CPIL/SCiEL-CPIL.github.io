"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin (150s limit)
    
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
        if time.time() > deadline - 15.0:
            break
            
        sol = _greedy_cover_beam(v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets, t_set_to_blocks, deadline, 
                                 rng_seed=seed, beam_width=beam_width)
        if sol:
            # Try to improve by removing redundant blocks
            sol = _remove_redundant(sol, t_set_index, all_blocks, block_cover_sets, num_t_sets, deadline)
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
                
    if best_solution is None:
        return []
        
    # Final polish: try to improve the best solution further
    if time.time() < deadline - 5.0:
        best_solution = _local_search_polish(best_solution, v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets, t_set_to_blocks, deadline)
    
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

def _remove_redundant(blocks, t_set_index, all_blocks, block_cover_sets, num_t_sets, deadline):
    """Remove blocks that are not needed for coverage."""
    if not blocks:
        return blocks
    
    # Convert to set of block tuples for fast lookup
    block_set = set(blocks)
    
    # Compute coverage count for each t-set
    cover_count = [0] * num_t_sets
    for block in blocks:
        b_idx = all_blocks.index(block) if block in all_blocks else None
        if b_idx is None:
            # Find the index
            for i, ab in enumerate(all_blocks):
                if ab == block:
                    b_idx = i
                    break
        if b_idx is not None:
            for ts_idx in block_cover_sets[b_idx]:
                cover_count[ts_idx] += 1
    
    # Try to remove blocks that are redundant
    changed = True
    while changed and time.time() < deadline - 3.0:
        changed = False
        for i, block in enumerate(list(blocks)):
            # Check if removing this block still maintains coverage
            b_idx = None
            for j, ab in enumerate(all_blocks):
                if ab == block:
                    b_idx = j
                    break
            if b_idx is None:
                continue
                
            # Check if all t-sets in this block are covered by other blocks
            redundant = True
            for ts_idx in block_cover_sets[b_idx]:
                if cover_count[ts_idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove the block
                blocks.remove(block)
                for ts_idx in block_cover_sets[b_idx]:
                    cover_count[ts_idx] -= 1
                changed = True
    
    return blocks

def _local_search_polish(blocks, v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets, t_set_to_blocks, deadline):
    """Try to improve the solution by swapping or removing blocks."""
    if not blocks:
        return blocks
    
    current_blocks = list(blocks)
    current_size = len(current_blocks)
    
    # Try multiple random restarts of local search
    rng = random.Random(12345)
    
    for restart in range(5):
        if time.time() > deadline - 3.0:
            break
            
        # Start from current best
        candidate = list(current_blocks)
        
        # Try to remove redundant blocks
        candidate = _remove_redundant(candidate, t_set_index, all_blocks, block_cover_sets, num_t_sets, deadline)
        
        if len(candidate) < current_size:
            current_blocks = candidate
            current_size = len(candidate)
        
        # Try swap moves: remove one block, add a better one
        for _ in range(20):
            if time.time() > deadline - 3.0:
                    break
            if not candidate:
                break
                
            # Remove a random block
            idx = rng.randint(0, len(candidate) - 1)
            removed_block = candidate.pop(idx)
            
            # Check if coverage is still valid
            if _is_valid_cover(candidate, v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets):
                # Valid without the block, keep it removed
                current_blocks = candidate
                current_size = len(candidate)
            else:
                # Need to add back or find replacement
                # Find uncovered t-sets
                uncovered = _find_uncovered(candidate, v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets)
                
                if uncovered:
                    # Find best block to cover most uncovered
                    best_block = None
                    best_gain = 0
                    for ts_idx in list(uncovered)[:10]:  # Limit search
                        for b_idx in t_set_to_blocks[ts_idx]:
                            gain = len(block_cover_sets[b_idx] & uncovered)
                            if gain > best_gain:
                                best_gain = gain
                                best_block = all_blocks[b_idx]
                    
                    if best_block is not None and best_block not in candidate:
                        candidate.append(best_block)
                    else:
                        candidate.append(removed_block)
                else:
                    candidate.append(removed_block)
                    
        if len(candidate) < current_size:
            current_blocks = candidate
            current_size = len(candidate)
    
    return current_blocks

def _is_valid_cover(blocks, v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets):
    """Check if the given blocks form a valid cover."""
    if not blocks:
        return num_t_sets == 0
        
    covered = set()
    for block in blocks:
        b_idx = None
        for j, ab in enumerate(all_blocks):
            if ab == block:
                b_idx = j
                break
        if b_idx is not None:
            covered |= block_cover_sets[b_idx]
            
    return len(covered) == num_t_sets

def _find_uncovered(blocks, v, k, t, t_set_index, all_blocks, block_cover_sets, num_t_sets):
    """Find uncovered t-sets."""
    covered = set()
    for block in blocks:
        b_idx = None
        for j, ab in enumerate(all_blocks):
            if ab == block:
                b_idx = j
                break
        if b_idx is not None:
            covered |= block_cover_sets[b_idx]
            
    return set(range(num_t_sets)) - covered
# EVOLVE-BLOCK-END
