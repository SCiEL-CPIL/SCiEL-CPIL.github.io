"""Covering design C(v,k,t) via Weighted Greedy Set-Cover."""
import itertools
import numpy as np

def construct(v, k, t):
    if v <= 25:
        return _weighted_greedy(v, k, t)
    else:
        return _basic_greedy(v, k, t)

def _basic_greedy(v, k, t):
    """Fallback simple greedy for large v."""
    all_tsubsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        seed_subset = next(iter(uncovered))
        block = list(seed_subset)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
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
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tsub)
    
    return blocks

def _weighted_greedy(v, k, t):
    """
    Weighted Greedy Set Cover.
    Uses NumPy for efficient score calculation.
    Score of a block is the sum of weights of the uncovered t-subsets it covers.
    Weights are decremented as blocks are selected, favoring 'rare' subsets.
    """
    all_ksubsets = list(itertools.combinations(range(v), k))
    tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubsets)
    n_ksubs = len(all_ksubsets)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    
    # Build incidence matrix A (n_ksubs x n_tsubs)
    # A[i, j] = 1 if k-subset i contains t-subset j
    A = np.zeros((n_ksubs, n_tsubs), dtype=np.uint8)
    for i, ksub in enumerate(all_ksubsets):
        for tsub in itertools.combinations(ksub, t):
            j = tsub_to_idx[tsub]
            A[i, j] = 1
            
    # Initial weights: number of k-subsets covering each t-subset
    weights = A.sum(axis=0).astype(np.int32)
    
    # Uncovered mask
    uncovered = np.ones(n_tsubs, dtype=bool)
    
    blocks = []
    selected = np.zeros(n_ksubs, dtype=bool)
    
    while uncovered.any():
        # Mask weights with uncovered status
        active_weights = np.where(uncovered, weights, 0)
        
        # Calculate scores for all remaining (unselected) blocks
        scores = A @ active_weights
        
        # Mask out already selected blocks
        scores[selected] = -1
        
        best_idx = int(np.argmax(scores))
        
        if scores[best_idx] <= 0:
            break
            
        blocks.append(list(all_ksubsets[best_idx]))
        selected[best_idx] = True
        
        # Update uncovered
        covered_mask = A[best_idx] == 1
        uncovered[covered_mask] = False
        
        # Update weights
        weights[covered_mask] -= 1
        
    return blocks
