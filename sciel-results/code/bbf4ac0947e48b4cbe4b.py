"""Optimized covering design C(19,8,5) using fast vectorized greedy with randomized restarts."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def _build_lookup(v, t):
    """Precompute all t-subsets and a mapping from frozenset to index."""
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    return all_tsubsets, tsubset_to_idx

def _fast_greedy(v, k, t, seed=42, max_time=100.0, randomize=False):
    """
    Fast greedy covering: repeatedly pick the k-subset covering the most uncovered t-subsets.
    Uses randomized tie-breaking when randomize=True.
    """
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + max_time
    
    all_tsubsets, tsubset_to_idx = _build_lookup(v, t)
    n_tsub = len(all_tsubsets)
    
    # covered[i] = True if t-subset i is covered
    covered = np.zeros(n_tsub, dtype=bool)
    num_uncovered = n_tsub
    
    blocks = []
    seen_blocks = set()
    
    while num_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Find uncovered t-subset indices (sample if too many)
        uncovered_mask = ~covered
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Sample some uncovered t-subsets to build candidate blocks around
        if randomize and len(uncovered_indices) > 20:
            sample = rng.sample(list(uncovered_indices), min(20, len(uncovered_indices)))
        else:
            sample = list(uncovered_indices[:20])
        
        best_block = None
        best_gain = -1
        best_newly_covered = None
        
        for idx in sample:
            focus = all_tsubsets[idx]
            block_set = set(focus)
            
            # Greedily add points to maximize uncovered t-subsets covered
            while len(block_set) < k:
                best_point = None
                best_local_gain = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count uncovered t-subsets that would become covered by adding x
                    # A t-subset is covered by block_set ∪ {x} if it contains x 
                    # and (t-1) of its elements are in block_set
                    gain = 0
                    for rest in itertools.combinations(block_set, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        i = tsubset_to_idx.get(ts, -1)
                        if i != -1 and not covered[i]:
                            gain += 1
                    if gain > best_local_gain:
                        best_local_gain = gain
                        best_point = x
                    elif gain == best_local_gain and best_point is not None and randomize and rng.random() < 0.3:
                        best_point = x
                
                if best_point is None:
                    break
                block_set.add(best_point)
            
            block = tuple(sorted(block_set))
            
            if block in seen_blocks:
                continue
            
            # Count how many currently uncovered t-subsets this block covers
            newly_covered = []
            for ts in itertools.combinations(block, t):
                i = tsubset_to_idx.get(ts, -1)
                if i != -1 and not covered[i]:
                    newly_covered.append(i)
            
            gain = len(newly_covered)
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_newly_covered = newly_covered
            elif gain == best_gain and best_block is not None and randomize and rng.random() < 0.2:
                best_block = block
                best_newly_covered = newly_covered
        
        if best_block is None:
            break
        
        seen_blocks.add(best_block)
        blocks.append(list(best_block))
        
        for i in best_newly_covered:
            if not covered[i]:
                covered[i] = True
                num_uncovered -= 1
    
    return blocks

def _local_refinement(blocks, v, k, t, max_time=10.0):
    """
    Try to improve the covering by removing redundant blocks.
    A block is redundant if all its t-subsets are covered by other blocks.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    all_tsubsets, tsubset_to_idx = _build_lookup(v, t)
    n_tsub = len(all_tsubsets)
    
    # For each t-subset, count how many blocks cover it
    tsub_count = np.zeros(n_tsub, dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            i = tsubset_to_idx.get(ts, -1)
            if i != -1:
                tsub_count[i] += 1
    
    # Try to remove blocks that are redundant
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = blocks[i]
            # Check if all t-subsets of this block are covered by other blocks
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx != -1 and tsub_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove this block
                for ts in itertools.combinations(block, t):
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx != -1:
                        tsub_count[idx] -= 1
                blocks.pop(i)
                improved = True
    
    return blocks

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    total_start = time.time()
    total_deadline = total_start + 140.0  # Leave 10s safety margin before 150s kill
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy strategies with different seeds
    seeds = [1, 42, 123, 777, 2024, 999, 555, 314, 271, 1618]
    
    for i, seed in enumerate(seeds):
        if time.time() > total_deadline:
            break
        
        remaining = total_deadline - time.time()
        if remaining < 3.0:
            break
        
        # Use more time for earlier runs, less for later ones
        alloc = min(remaining * 0.5, 25.0) if i < 3 else min(remaining * 0.3, 15.0)
        
        blocks = _fast_greedy(v, k, t, seed=seed, max_time=alloc, randomize=(i > 0))
        
        if blocks and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        return []
    
    # Local refinement: remove redundant blocks
    if time.time() < total_deadline - 5.0:
        refine_time = min(total_deadline - time.time() - 5.0, 15.0)
        best_blocks = _local_refinement(best_blocks, v, k, t, max_time=refine_time)
    
    return best_blocks
# EVOLVE-BLOCK-END
