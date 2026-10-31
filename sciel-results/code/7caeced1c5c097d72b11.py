"""Optimized covering design C(v,k,t) with deadline guard and multi-strategy search."""
import itertools
import time
import random
from collections import defaultdict

def _all_tsubsets(v, t):
    """Precompute all t-subsets as tuples and map to indices."""
    subs = list(itertools.combinations(range(v), t))
    return subs

def _tsubset_index(sub, v, t, comb_index_map):
    """Get index of a t-subset."""
    return comb_index_map.get(tuple(sub), -1)

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    MAX_TIME = 150.0
    
    # Precompute all t-subsets
    all_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_subs)
    
    # Map each t-subset to an index for fast lookup
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    # For each block (k-subset), precompute which t-subsets it covers
    # We'll compute this on the fly for efficiency
    
    def _covers(block):
        """Return set of t-subset indices covered by this block."""
        covered = set()
        for ts in itertools.combinations(block, t):
            idx = sub_to_idx.get(tuple(ts), -1)
            if idx >= 0:
                covered.add(idx)
        return covered
    
    def _uncovered_count(covered_set, uncovered):
        """Count how many uncovered subsets are in covered_set."""
        return sum(1 for x in covered_set if x in uncovered)
    
    def greedy_construct(uncovered, v, k, t, sub_to_idx, seed=None):
        """Greedy construction starting from given uncovered set."""
        if seed is not None:
            random.seed(seed)
        
        blocks = []
        uncovered = set(uncovered)
        
        while uncovered:
            # Pick the first uncovered t-subset (or a random one for diversity)
            if seed is not None and random.random() < 0.3:
                # Pick a random uncovered subset for diversity
                target = random.choice(list(uncovered))
            else:
                target = min(uncovered)
            
            block = list(target)
            
            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                best_gain = -1
                best_x = None
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many new t-subsets would be covered
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in sub_to_idx:
                            idx = sub_to_idx[new_sub]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            covered = _covers(block)
            uncovered -= covered
        
        return blocks, uncovered
    
    def local_search(blocks, uncovered, v, k, t, sub_to_idx, max_iters=100):
        """Try to improve by removing redundant blocks or swapping."""
        blocks = [list(b) for b in blocks]
        uncovered = set(uncovered)
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Check for redundant blocks
            for i in range(len(blocks)):
                # Check if removing block i still leaves all covered
                # Compute coverage of all other blocks
                other_covered = set()
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    other_covered |= _covers(blocks[j])
                
                # If all subsets covered by block i are also covered by others, remove it
                block_covered = _covers(blocks[i])
                if block_covered <= other_covered:
                    del blocks[i]
                    improved = True
                    break
        
        return blocks, uncovered
    
    # Strategy 1: Basic greedy
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy passes with different seeds
    for seed in [0, 1, 42, 123, 456]:
        if time.time() > DEADLINE:
            break
        blocks, unc = greedy_construct(set(range(n_subs)), v, k, t, sub_to_idx, seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: return empty or minimal
        return []
    
    # Try local search on the best solution
    if time.time() < DEADLINE - 5:
        ls_blocks, _ = local_search(best_blocks, set(), v, k, t, sub_to_idx, max_iters=50)
        if len(ls_blocks) < best_count:
            best_blocks = ls_blocks
            best_count = len(ls_blocks)
    
    # Convert to list of lists
    return [list(b) for b in best_blocks]
