# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import combinations

_START = None
_DEADLINE_FRACTION = 0.85
_TOTAL_BUDGET = 270.0

def _now():
    return time.time()

def _init_time():
    global _START
    if _START is None:
        _START = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TOTAL_BUDGET * _DEADLINE_FRACTION - (time.time() - _START))

def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    diff = (arr[:, None, :] != arr[None, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d

def _compute_quartet_votes(d):
    """
    Compute weighted votes for each of the 3 topologies for every quartet.
    Returns a dictionary: {split_frozenset: weight}
    """
    n = d.shape[0]
    split_weights = defaultdict(float)
    
    # Vectorized scoring
    quartets = list(combinations(range(n), 4))
    if not quartets:
        return split_weights
        
    a_idx = np.array([q[0] for q in quartets])
    b_idx = np.array([q[1] for q in quartets])
    c_idx = np.array([q[2] for q in quartets])
    d_idx = np.array([q[3] for q in quartets])
    
    s1 = d[a_idx, b_idx] + d[c_idx, d_idx] - d[a_idx, c_idx] - d[a_idx, d_idx] - d[b_idx, c_idx] - d[b_idx, d_idx]
    s2 = d[a_idx, c_idx] + d[b_idx, d_idx] - d[a_idx, b_idx] - d[a_idx, d_idx] - d[c_idx, b_idx] - d[c_idx, d_idx]
    s3 = d[a_idx, d_idx] + d[b_idx, c_idx] - d[a_idx, b_idx] - d[a_idx, c_idx] - d[d_idx, b_idx] - d[d_idx, c_idx]
    
    scores = np.stack([s1, s2, s3], axis=1)
    min_idx = np.argmin(scores, axis=1)
    min_val = np.min(scores, axis=1)
    second_min_val = np.partition(scores, 1, axis=1)[:, 1]
    weights = second_min_val - min_val
    
    all_leaves = frozenset(range(n))
    
    for i in range(len(quartets)):
        a, b, c, d = quartets[i]
        t = min_idx[i]
        w = weights[i]
        
        if t == 0:
            side1 = frozenset([a, b])
        elif t == 1:
            side1 = frozenset([a, c])
        else:
            side1 = frozenset([a, d])
            
        side2 = all_leaves - side1
        
        # Normalize split
        if len(side1) < len(side2):
            split = side1
        elif len(side2) < len(side1):
            split = side2
        else:
            if tuple(sorted(side1)) < tuple(sorted(side2)):
                split = side1
            else:
                split = side2
                
        split_weights[split] += w
        
    return split_weights

def _greedy_compatible_splits(split_weights, n):
    """
    Select a compatible set of splits maximizing total weight.
    """
    # Sort by weight descending
    sorted_splits = sorted(split_weights.items(), key=lambda x: -x[1])
    
    chosen_splits = []
    chosen_set = set()
    
    for split, weight in sorted_splits:
        # Check compatibility
        compatible = True
        for existing in chosen_set:
            # Two splits A and B are incompatible if A&B, A-B, B-A, B-A are all non-empty
            # i.e., they cross.
            if (split & existing) and (split - existing) and (existing - split) and (existing & (set(range(n)) - split)):
                 # More simply:
                 # A crosses B if (A & B) and (A & B^c) and (A^c & B) and (A^c & B^c) are all non-empty
                 # Since A and B are subsets of U:
                 # A & B != empty
                 # A & (U-B) != empty
                 # (U-A) & B != empty
                 # (U-A) & (U-B) != empty
                 
                 # Let's use the standard check:
                 # A and B are compatible if one of the 4 intersections is empty.
                 if (split & existing) and (split & (set(range(n)) - existing)) and (existing & (set(range(n)) - split)) and (existing & (set(range(n)) - (set(range(n)) - split))):
                     # This logic is getting messy.
                     pass
                     
        # Correct compatibility check:
        # A and B are incompatible if:
        # (A & B) is not empty
        # (A & B^c) is not empty
        # (A^c & B) is not empty
        # (A^c & B^c) is not empty
        
        # Since we are in a loop, let's optimize.
        # For small n, set operations are fine.
        
        compatible = True
        for existing in chosen_set:
            u = set(range(n))
            if (split & existing) and (split & (u - existing)) and (existing & (u - split)) and (existing & (u - (u - split))):
                # Wait, (u - (u - split)) is split.
                # The 4 regions are:
                # A ∩ B
                # A ∩ B^c
                # A^c ∩ B
                # A^c ∩ B^c
                # If all are non-empty, they are incompatible.
                
                r1 = split & existing
                r2 = split & (u - existing)
                r3 = existing & (u - split)
                r4 = (u - split) & (u - existing)
                
                if r1 and r2 and r3 and r4:
                    compatible = False
                    break
                    
        if compatible:
            chosen_splits.append(split)
            chosen_set.add(split)
            
    return chosen_splits

def _splits_to_tree(splits, n, labels):
    """
    Convert a list of splits to a Newick string.
    """
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    label_set = frozenset(range(n))
    
    def build(subset):
        if len(subset) == 1:
            return labels[list(subset)[0]]
        if len(subset) == 2:
            return f"({labels[list(subset)[0]]},{labels[list(subset)[1]]})"
            
        best_split = None
        best_balance = 0
        for S in splits:
            side1 = S & subset
            side2 = subset - side1
            if 1 < len(side1) < len(subset):
                balance = min(len(side1), len(side2))
                if balance > best_balance:
                    best_balance = balance
                    best_split = side1
                    
        if best_split is None:
            return "(" + ",".join(labels[i] for i in sorted(subset)) + ")"
            
        side1 = sorted(best_split)
        side2 = sorted(subset - best_split)
        
        tree1 = build(frozenset(side1))
        tree2 = build(frozenset(side2))
        
        return f"({tree1},{tree2})"
        
    return build(label_set)

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    L = arr.shape[1]
    
    if n < 3:
        if n == 1:
            return labels[0]
        else:
            return f"({labels[0]},{labels[1]})"
            
    d = _jc_matrix(arr)
    
    # Quartet Consensus
    split_weights = _compute_quartet_votes(d)
    chosen_splits = _greedy_compatible_splits(split_weights, n)
    
    # If we don't have enough splits to resolve the tree, fallback to NJ?
    # For now, just build the tree from the chosen splits.
    tree_str = _splits_to_tree(chosen_splits, n, labels)
    
    return tree_str
# EVOLVE-BLOCK-END
