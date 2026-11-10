"""Covering design C(19,8,5) with optimized greedy and local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0


def construct(v, k, t):
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each block (k-subset), precompute which t-subsets it covers
    # We'll use a dictionary mapping block tuple -> mask of covered t-subsets
    # But C(19,8) = 75582, and each mask is 11628 bits. That's manageable.
    
    # Actually, let's think about this differently.
    # C(19,5) = 11628 t-subsets
    # C(19,8) = 75582 possible blocks
    # Each block covers C(8,5) = 56 t-subsets
    
    # Strategy: 
    # 1. Greedy construction with good heuristic
    # 2. Local search to improve
    
    # Precompute for each point x and each (t-1)-subset R, the t-subset R ∪ {x}
    # This helps in gain computation
    
    # For gain computation: given a partial block B and candidate x,
    # gain = number of uncovered t-subsets S such that S ⊆ B ∪ {x} and x ∈ S
    # This means S = R ∪ {x} where R is a (t-1)-subset of B
    # So we need to count how many (t-1)-subsets R of B have R ∪ {x} uncovered
    
    # Precompute: for each (t-1)-subset R and point x not in R, the index of R ∪ {x}
    # There are C(19,4) = 3876 (t-1)-subsets and 15 choices for x, so ~58140 entries
    
    # Actually, let's just compute gain on the fly but efficiently
    
    def compute_gain_fast(block, x, uncovered_count_by_tsubset):
        """Count uncovered t-subsets that would be covered by adding x to block."""
        gain = 0
        # For each (t-1)-subset R of block, check if R ∪ {x} is uncovered
        for rest in itertools.combinations(block, t - 1):
            s = tuple(sorted(rest + (x,)))
            si = idx_map.get(s, -1)
            if si >= 0 and uncovered_count_by_tsubset[si] > 0:
                gain += 1
        return gain
    
    # Main greedy construction
    uncovered = np.ones(n_t, dtype=np.int32)  # count of blocks covering each t-subset
    blocks = []
    
    # Precompute all k-subsets? No, too many. Instead, build blocks greedily.
    
    def greedy_build_block(uncovered, seed_subset=None):
        """Greedily build a block of size k maximizing coverage of uncovered t-subsets."""
        if seed_subset is not None:
            block = list(seed_subset)
        else:
            # Find the t-subset with minimum coverage (should be 0 if we're being greedy)
            min_count = uncovered.min()
            candidates = np.where(uncovered == min_count)[0]
            if len(candidates) == 0:
                return None
            start_idx = candidates[random.randint(0, len(candidates) - 1)]
            block = list(all_t_subsets[start_idx])
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = compute_gain_fast(block, x, uncovered)
                if gain > best_gain or (gain == best_gain and best_x is not None and x < best_x):
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain <= 0:
                break
            block.append(best_x)
        
        block.sort()
        return block
    
    # Phase 1: Initial greedy construction
    while uncovered.any():
        if time.time() - start_time > DEADLINE_SECONDS - 20:
            break
        
        block = greedy_build_block(uncovered)
        if block is None:
            break
        
        # Update uncovered counts
        for comb in itertools.combinations(block, t):
            idx = idx_map[tuple(comb)]
            uncovered[idx] += 1
        
        blocks.append(block)
    
    # Phase 2: Local search - try to remove blocks
    # If a block is redundant (all its t-subsets are covered by other blocks), remove it
    improved = True
    max_local_iters = 50
    iter_count = 0
    while improved and iter_count < max_local_iters:
        improved = False
        iter_count += 1
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
        
        # Recompute coverage
        coverage = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            for comb in itertools.combinations(b, t):
                idx = idx_map[tuple(comb)]
                coverage[idx] += 1
        
        # Try to remove each block
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() - start_time > DEADLINE_SECONDS - 10:
                break
            
            # Check if block i is redundant
            block = blocks[i]
            redundant = True
            for comb in itertools.combinations(block, t):
                idx = idx_map[tuple(comb)]
                if coverage[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove this block
                blocks.pop(i)
                improved = True
    
    # Phase 3: Try to improve by replacing blocks
    # For each block, try swapping one point with another to see if we can maintain coverage
    # with fewer total blocks or better structure
    
    if time.time() - start_time < DEADLINE_SECONDS - 15:
        improved = True
        while improved and time.time() - start_time < DEADLINE_SECONDS - 15:
            improved = False
            
            # Recompute coverage
            coverage = np.zeros(n_t, dtype=np.int32)
            for b in blocks:
                for comb in itertools.combinations(b, t):
                    idx = idx_map[tuple(comb)]
                    coverage[idx] += 1
            
            for bi in range(len(blocks)):
                if time.time() - start_time > DEADLINE_SECONDS - 15:
                    break
                
                block = list(blocks[bi])
                for si in range(len(block)):
                    for x in range(v):
                        if x in block:
                            continue
                        
                        new_block = block[:]
                        new_block[si] = x
                        new_block.sort()
                        
                        # Check if new block is valid (distinct from other blocks)
                        new_key = tuple(new_block)
                        if new_key in [tuple(b) for b in blocks if b != blocks[bi]]:
                            continue
                        
                        # Check if swapping maintains coverage
                        # Old block covers some t-subsets, new block covers others
                        # We need: for every t-subset covered only by old block,
                        # it must still be covered by new block or other blocks
                        
                        old_covered = set()
                        for comb in itertools.combinations(block, t):
                            old_covered.add(idx_map[tuple(comb)])
                        
                        new_covered = set()
                        for comb in itertools.combinations(new_block, t):
                            new_covered.add(idx_map[tuple(comb)])
                        
                        # T-subsets that were covered by old but not new
                        lost = old_covered - new_covered
                        # Check if all lost t-subsets are covered by other blocks
                        valid_swap = True
                        for idx in lost:
                            if coverage[idx] <= 1:
                                valid_swap = False
                                break
                        
                        if valid_swap:
                            # Also check if new block covers any previously uncovered t-subsets
                            # (this would be a bonus, but we're just maintaining)
                            blocks[bi] = new_block
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Final check: ensure all t-subsets are covered
    coverage = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        for comb in itertools.combinations(b, t):
            idx = idx_map[tuple(comb)]
            coverage[idx] += 1
    
    # If any t-subset is not covered, add more blocks
    if not (coverage >= 1).all():
        uncovered = (coverage == 0).astype(np.int32)
        while uncovered.any():
            if time.time() - start_time > DEADLINE_SECONDS - 2:
                break
            block = greedy_build_block(uncovered)
            if block is None:
                break
            for comb in itertools.combinations(block, t):
                idx = idx_map[tuple(comb)]
                uncovered[idx] += 1
                coverage[idx] += 1
            blocks.append(block)
    
    # Remove duplicates
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks
# EVOLVE-BLOCK-END
