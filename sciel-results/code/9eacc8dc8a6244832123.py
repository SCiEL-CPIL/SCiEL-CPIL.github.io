"""Covering design C(24,7,3): vectorized greedy with randomized weighting and local search."""
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def _precompute(v, k, t):
    """Precompute all t-subsets, their indices, and per-point coverage masks."""
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_index = {triple: i for i, triple in enumerate(all_triples)}
    
    # For each point, bitmask of triples containing it
    # Use numpy bitarray for efficiency
    point_mask = np.zeros((v, num_triples), dtype=np.uint8)
    for i, triple in enumerate(all_triples):
        for pt in triple:
            point_mask[pt, i] = 1
    
    return all_triples, triple_index, num_triples, point_mask


def _compute_gain(block, x, uncovered, triple_index, t):
    """Compute gain of adding x to block: number of uncovered triples covered."""
    if x in block:
        return 0
    gain = 0
    # For t=3: need pairs from block, check if (pair + x) is an uncovered triple
    for rest in itertools.combinations(block, t - 1):
        candidate = tuple(sorted(rest + (x,)))
        idx = triple_index.get(candidate)
        if idx is not None and idx in uncovered:
            gain += 1
    return gain


def _build_greedy(v, k, t, rng, all_triples, triple_index, num_triples, point_mask):
    """
    Build a greedy covering using vectorized gain computation.
    Uses randomized weighting to break ties.
    """
    # uncovered: boolean array of length num_triples
    uncovered = np.ones(num_triples, dtype=bool)
    blocks = []
    
    while uncovered.any():
        # Pick a random uncovered triple to start
        uncovered_indices = np.where(uncovered)[0]
        start_idx = int(rng.choice(uncovered_indices))
        start_triple = all_triples[start_idx]
        block = list(start_triple)
        block_set = set(block)
        
        while len(block) < k:
            best_x = -1
            best_score = -1.0
            best_ties = []
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = _compute_gain(block, x, set(np.where(uncovered)[0]), triple_index, t)
                score = gain * 1000.0 + rng.random()
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_ties = [x]
                elif score == best_score:
                    best_ties.append(x)
            
            chosen = rng.choice(best_ties) if best_ties else best_x
            block.append(chosen)
            block_set.add(chosen)
        
        block.sort()
        blocks.append(block)
        
        # Remove all t-subsets of this block from uncovered
        for comb in itertools.combinations(block, t):
            idx = triple_index.get(tuple(comb))
            if idx is not None:
                uncovered[idx] = False
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, triple_index):
    """Try to remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    new_blocks = list(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(len(new_blocks)):
            block_i = new_blocks[i]
            new_blocks.pop(i)
            
            # Check if all t-subsets are still covered
            covered = set()
            for block in new_blocks:
                for comb in itertools.combinations(block, t):
                    covered.add(tuple(comb))
            
            all_covered = True
            for comb in itertools.combinations(block_i, t):
                if tuple(comb) not in covered:
                    all_covered = False
                    break
            
            if all_covered:
                changed = True
            else:
                new_blocks.insert(i, block_i)
    
    return new_blocks


def _local_search(blocks, v, k, t, rng, all_triples, triple_index, num_triples, point_mask, time_limit=10):
    """Local search: try replacing blocks with better ones."""
    if not blocks:
        return blocks
    
    # Build uncovered set
    covered = set()
    for block in blocks:
        for comb in itertools.combinations(block, t):
            covered.add(tuple(comb))
    
    uncovered = set()
    for i, triple in enumerate(all_triples):
        if triple not in covered:
            uncovered.add(i)
    
    if not uncovered:
        return blocks
    
    best_blocks = list(blocks)
    best_uncovered = set(uncovered)
    
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < time_limit and iterations < 100:
        iterations += 1
        if not best_uncovered:
            break
        
        # Try to replace a random block
        idx = rng.randint(0, len(best_blocks) - 1)
        old_block = best_blocks[idx]
        
        # Build a new greedy block starting from an uncovered triple
        start_idx = rng.choice(list(best_uncovered))
        start_triple = all_triples[start_idx]
        new_block = list(start_triple)
        block_set = set(new_block)
        
        while len(new_block) < k:
            best_x = -1
            best_score = -1.0
            best_ties = []
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = _compute_gain(new_block, x, best_uncovered, triple_index, t)
                score = gain * 1000.0 + rng.random()
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_ties = [x]
                elif score == best_score:
                    best_ties.append(x)
            
            chosen = rng.choice(best_ties) if best_ties else best_x
            new_block.append(chosen)
            block_set.add(chosen)
        
        new_block.sort()
        
        # Check if replacing old_block with new_block improves coverage
        old_covered = set(tuple(c) for c in itertools.combinations(old_block, t))
        new_covered = set(tuple(c) for c in itertools.combinations(new_block, t))
        
        temp_uncovered = set(best_uncovered) | old_covered
        temp_uncovered -= new_covered
        
        if len(temp_uncovered) < len(best_uncovered):
            best_blocks[idx] = new_block
            best_uncovered = temp_uncovered
    
    best_blocks = _remove_redundant_blocks(best_blocks, v, k, t, triple_index)
    
    return best_blocks


def construct(v, k, t):
    """Main entry point: try multiple greedy builds with local search."""
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Precompute
    all_triples, triple_index, num_triples, point_mask = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    # Do several greedy builds with different random seeds
    max_builds = 20
    for build_idx in range(max_builds):
        if time.time() > deadline - 15:
            break
        
        blocks = _build_greedy(v, k, t, rng, all_triples, triple_index, num_triples, point_mask)
        
        # Local search to improve
        remaining_time = deadline - 15 - time.time()
        if remaining_time > 5:
            blocks = _local_search(blocks, v, k, t, rng, all_triples, triple_index, num_triples, point_mask, time_limit=min(10, remaining_time))
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we've reached a good number, stop early
        if best_len <= 76:
            break
    
    if best_blocks is None:
        best_blocks = _build_greedy(v, k, t, random.Random(0), all_triples, triple_index, num_triples, point_mask)
    
    return best_blocks
