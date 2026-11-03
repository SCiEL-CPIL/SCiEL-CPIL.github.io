"""Rewritten program for covering design C(19,8,5) using Lagrangian-guided greedy."""
import itertools
import time
import random
import sys
import math

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
    
    # Run multiple strategies with different Lagrangian parameters
    strategies = [
        (1, 42, 1.0, 0.1),      # (beam_width, seed, lambda_init, lambda_step)
        (5, 101, 1.0, 0.1),
        (10, 102, 2.0, 0.2),
        (20, 103, 1.5, 0.15),
        (50, 104, 1.0, 0.05),
        (10, 201, 3.0, 0.3),
        (20, 202, 2.0, 0.2),
        (50, 203, 1.5, 0.1),
        (1, 301, 5.0, 0.5),     # Higher lambda for stronger guidance
        (5, 302, 4.0, 0.4),
        (10, 303, 3.0, 0.3),
        (20, 304, 2.5, 0.25),
    ]
    
    for beam_width, seed, lambda_init, lambda_step in strategies:
        if time.time() > deadline - 15.0:
            break
            
        sol = _lagrangian_greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_sets, 
                                        num_t_sets, t_set_to_blocks, deadline,
                                        rng_seed=seed, beam_width=beam_width,
                                        lambda_init=lambda_init, lambda_step=lambda_step)
        if sol:
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
                
    if best_solution is None:
        return []
        
    # Post-processing: remove redundant blocks
    best_solution = _remove_redundant_blocks(best_solution, block_cover_sets, all_blocks, deadline)
    
    return best_solution

def _lagrangian_greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_sets, 
                              num_t_sets, t_set_to_blocks, deadline,
                              rng_seed=42, beam_width=1, lambda_init=1.0, lambda_step=0.1):
    """
    Lagrangian-relaxation-guided greedy set cover.
    
    We maintain penalty weights w[i] for each t-set i. Initially uniform.
    At each step, we pick a block maximizing sum_{i in block} w[i] * [i uncovered].
    After picking, we update weights: increase w[i] for uncovered i, decrease for covered.
    This guides the greedy toward globally important regions.
    """
    rng = random.Random(rng_seed)
    uncovered = set(range(num_t_sets))
    selected_blocks = []
    
    # Lagrangian multipliers / penalty weights
    w = [lambda_init] * num_t_sets
    
    while uncovered:
        if time.time() > deadline - 3.0:
            return None
        
        candidate_blocks = set()
        
        # Pick beam_width random t-sets from uncovered to guide the search
        if len(uncovered) <= beam_width:
            targets = list(uncovered)
        else:
            targets = rng.sample(list(uncovered), beam_width)
            
        for target_tset_idx in targets:
            # Find best block covering this target, weighted by Lagrangian multipliers
            best_block_idx = -1
            max_score = -1.0
            
            for b_idx in t_set_to_blocks[target_tset_idx]:
                # Compute weighted gain: sum of w[i] for i in block_cover_sets[b_idx] ∩ uncovered
                score = 0.0
                for i in block_cover_sets[b_idx]:
                    if i in uncovered:
                        score += w[i]
                if score > max_score:
                    max_score = score
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                candidate_blocks.add(best_block_idx)
        
        # Pick the best block among candidates by weighted gain
        final_best_block_idx = -1
        final_max_score = -1.0
        
        for b_idx in candidate_blocks:
            score = 0.0
            for i in block_cover_sets[b_idx]:
                if i in uncovered:
                    score += w[i]
            if score > final_max_score:
                final_max_score = score
                final_best_block_idx = b_idx
        
        if final_best_block_idx == -1:
            break
            
        selected_blocks.append(all_blocks[final_best_block_idx])
        
        # Update uncovered
        covered = block_cover_sets[final_best_block_idx]
        uncovered -= covered
        
        # Update Lagrangian multipliers
        # Increase weights for still-uncovered t-sets, decrease for newly covered
        for i in list(uncovered):
            w[i] *= (1.0 + lambda_step)
        for i in covered:
            if i not in uncovered:
                w[i] *= (1.0 - lambda_step * 0.5)
                # Floor to prevent negative weights
                w[i] = max(w[i], 0.1)
        
    return selected_blocks

def _remove_redundant_blocks(blocks, block_cover_sets, all_blocks, deadline):
    """Remove blocks that don't cover any t-set uniquely."""
    if not blocks:
        return blocks
    
    # Map each block to its cover set
    block_to_cover = {}
    for block in blocks:
        # Find the index of this block in all_blocks
        # For efficiency, use a set-based lookup
        block_set = frozenset(block)
        cover = None
        for i, b in enumerate(all_blocks):
            if frozenset(b) == block_set:
                cover = block_cover_sets[i]
                break
        if cover is not None:
            block_to_cover[frozenset(block)] = cover
    
    # Count how many blocks cover each t-set
    t_set_count = {}
    for block_set, cover in block_to_cover.items():
        for i in cover:
            t_set_count[i] = t_set_count.get(i, 0) + 1
    
    # Remove blocks where all their t-sets are covered by other blocks
    kept = []
    for block in blocks:
        block_set = frozenset(block)
        cover = block_to_cover.get(block_set)
        if cover is None:
            kept.append(block)
            continue
        
        redundant = True
        for i in cover:
            if t_set_count.get(i, 0) == 1:
                redundant = False
                break
        
        if not redundant:
            kept.append(block)
    
    return kept
