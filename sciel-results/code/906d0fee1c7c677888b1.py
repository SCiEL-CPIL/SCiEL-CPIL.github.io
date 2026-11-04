"""Covering design C(v,k,t) via greedy set-cover with efficient bitset tracking."""
import itertools
import random
import sys

def construct(v, k, t):
    # For small parameters, we can use a smarter approach.
    # Strategy: Greedy set cover where each candidate block is evaluated by
    # how many currently uncovered t-subsets it covers.
    # To be faster, we use a randomized greedy with multiple restarts for small v.
    
    if v <= 20 and k <= 8:
        # Use randomized greedy with multiple passes
        return _greedy_cover(v, k, t, num_restarts=3, seed=42)
    else:
        # Fallback to basic greedy for larger instances
        return _basic_greedy(v, k, t)


def _basic_greedy(v, k, t):
    """Original greedy: pick uncovered t-subset, extend to k-subset maximizing coverage."""
    all_tsubsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        # Pick the first uncovered t-subset as seed
        seed_subset = next(iter(uncovered))
        block = list(seed_subset)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(combo + (x,)))
                    if new_tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove all t-subsets of this block from uncovered
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tsub)
    
    return blocks


def _greedy_cover(v, k, t, num_restarts=3, seed=42):
    """
    Randomized greedy set-cover:
    - Precompute all possible k-subsets? No, too many.
    - Instead, use a priority-based greedy: at each step, pick the k-subset
      that covers the most uncovered t-subsets.
    - To speed up, use local search: start from a random or heuristic block,
      then improve by swapping points.
    
    For C(19,7,4): v=19, k=7, t=4.
    Number of 4-subsets: C(19,4) = 3876.
    Number of 7-subsets: C(19,7) = 50388.
    
    We can precompute for each 7-subset, the list of 4-subsets it contains.
    Then greedy set cover: repeatedly pick the 7-subset covering most uncovered 4-subsets.
    """
    rng = random.Random(seed)
    
    # Precompute all k-subsets and their contained t-subsets
    all_ksubsets = list(itertools.combinations(range(v), k))
    # Map each t-subset to index
    tsubsets = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    
    # For each k-subset, compute the set of t-subset indices it covers
    ksub_tsubsets = []
    for ksub in all_ksubsets:
        ts_idxs = set()
        for tsub in itertools.combinations(ksub, t):
            ts_idxs.add(tsub_to_idx[tsub])
        ksub_tsubsets.append(ts_idxs)
    
    # Greedy set cover
    uncovered = set(range(len(tsubsets)))
    blocks = []
    
    while uncovered:
        best_idx = -1
        best_covered = 0
        best_new_covered = set()
        
        for i, ts_set in enumerate(ksub_tsubsets):
            covered = ts_set & uncovered
            if len(covered) > best_covered:
                best_covered = len(covered)
                best_idx = i
                best_new_covered = covered
        
        if best_idx == -1:
            break
        
        blocks.append(list(all_ksubsets[best_idx]))
        uncovered -= best_new_covered
    
    return blocks


def _greedy_cover_optimized(v, k, t, num_restarts=3, seed=42):
    """
    Optimized greedy with numpy for speed.
    """
    import numpy as np
    
    rng = random.Random(seed)
    
    all_ksubsets = list(itertools.combinations(range(v), k))
    tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubsets)
    n_ksubs = len(all_ksubsets)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    
    # Build incidence matrix: n_ksubs x n_tsubs
    # incidence[i, j] = 1 if k-subset i contains t-subset j
    incidence = np.zeros((n_ksubs, n_tsubs), dtype=np.uint8)
    for i, ksub in enumerate(all_ksubsets):
        for tsub in itertools.combinations(ksub, t):
            j = tsub_to_idx[tsub]
            incidence[i, j] = 1
    
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    blocks = []
    
    while uncovered.any():
        # For each k-subset, count how many uncovered t-subsets it covers
        covered_counts = incidence @ uncovered
        best_idx = int(np.argmax(covered_counts))
        if covered_counts[best_idx] == 0:
            break
        blocks.append(list(all_ksubsets[best_idx]))
        uncovered[incidence[best_idx] == 1] = 0
    
    return blocks
