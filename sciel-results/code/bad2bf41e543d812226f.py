import math
import time
import numpy as np
from itertools import combinations

DEADLINE_SECONDS = 255.0
SAFETY_MARGIN = 15.0

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Map ACGT to 0,1,2,3
    lut = np.zeros(256, dtype=np.int8)
    lut[65] = 0 # A
    lut[67] = 1 # C
    lut[71] = 2 # G
    lut[84] = 3 # T
    
    int_seqs = []
    for lab in labels:
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        int_seqs.append(lut[arr])
        
    int_seqs = np.array(int_seqs) # (n, L)
    
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diffs = (int_seqs[i] != int_seqs).sum(axis=1)
        p = diffs.astype(np.float64) / L
        p_clipped = np.clip(p, 0.0, 0.7499)
        d = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
        d[p >= 0.7499] = 3.0
        dist_matrix[i] = d
        dist_matrix[i, i] = 0.0
    return dist_matrix

def _get_quartet_split(i, j, k, l, dist_matrix):
    """
    Determine the best split for quartet {i,j,k,l} based on JC distances.
    Returns the split as a frozenset of the smaller side.
    """
    dij = dist_matrix[i, j]
    dik = dist_matrix[i, k]
    dil = dist_matrix[i, l]
    djk = dist_matrix[j, k]
    djl = dist_matrix[j, l]
    dkl = dist_matrix[k, l]
    
    # Sum of opposite edges
    s1 = dij + dkl # (ij, kl)
    s2 = dik + djl # (ik, jl)
    s3 = dil + djk # (il, jk)
    
    # Find minimum
    min_sum = min(s1, s2, s3)
    
    if min_sum == s1:
        # Split {i,j} | {k,l}
        side1 = frozenset([i, j])
        side2 = frozenset([k, l])
    elif min_sum == s2:
        # Split {i,k} | {j,l}
        side1 = frozenset([i, k])
        side2 = frozenset([j, l])
    else:
        # Split {i,l} | {j,k}
        side1 = frozenset([i, l])
        side2 = frozenset([j, k])
        
    # Return the smaller side (or either if equal size)
    if len(side1) <= len(side2):
        return side1
    else:
        return side2

def _build_tree_from_splits(splits, labels, n):
    """
    Build a tree from a set of splits using a greedy conflict-free approach.
    """
    if n <= 3:
        return "(" + ",".join(labels) + ");"
        
    # splits is a list of frozensets
    # We want to select a compatible subset of splits that defines a binary tree.
    # A binary tree on n leaves has n-3 non-trivial splits.
    
    # Greedy selection
    selected_splits = []
    remaining = list(splits)
    
    # Sort by size? Or just iterate?
    # To ensure we get a binary tree, we should prefer splits that are "small" (cherries) or "large"?
    # Actually, any compatible set of n-3 splits defines a binary tree.
    # We need to select n-3 compatible splits.
    
    # Heuristic: Start with cherries (size 2 splits) as they are robust.
    remaining.sort(key=lambda s: len(s))
    
    for s in remaining:
        if len(selected_splits) >= n - 3:
            break
            
        # Check compatibility with all selected splits
        compatible = True
        for t in selected_splits:
            # Conflict if:
            # 1. s is not subset of t
            # 2. t is not subset of s
            # 3. s intersection t is empty
            # 4. s union t is not all leaves
            inter = s & t
            union = s | t
            all_leaves = frozenset(range(n))
            
            if not (s <= t or t <= s or len(inter) == 0 or union == all_leaves):
                compatible = False
                break
                
        if compatible:
            selected_splits.append(s)
            
    # If we don't have enough splits (e.g. star tree), add arbitrary splits to make it binary?
    # The task requires a valid Newick. A star tree is valid but scores 0.
    # We should try to fill up to n-3 splits.
    if len(selected_splits) < n - 3:
        # Add random splits? Or just return the tree defined by selected_splits?
        # A tree with < n-3 splits is not binary.
        # We can generate a binary tree from the selected splits by resolving ambiguities.
        # For simplicity, if we have < n-3, we just build the tree from what we have.
        # The Newick parser will handle non-binary nodes (multifurcations) if we output them,
        # but the score is based on non-trivial splits.
        # A multifurcating tree has fewer splits.
        # To maximize score, we want n-3 splits.
        # If we are short, we can add "dummy" splits that are compatible.
        
        # Generate all possible splits of size 2 to n-2
        all_possible_splits = []
        for size in range(2, n-1):
            for comb in combinations(range(n), size):
                fs = frozenset(comb)
                if fs not in selected_splits:
                    # Check if compatible with current selected
                    comp = True
                    for t in selected_splits:
                        inter = fs & t
                        union = fs | t
                        all_leaves = frozenset(range(n))
                        if not (fs <= t or t <= fs or len(inter) == 0 or union == all_leaves):
                            comp = False
                            break
                    if comp:
                        all_possible_splits.append(fs)
                        
        # Add until we have n-3
        for fs in all_possible_splits:
            if len(selected_splits) >= n - 3:
                break
            selected_splits.append(fs)
            
    # Now convert selected_splits to Newick
    return _splits_to_newick(selected_splits, labels, n)

def _splits_to_newick(splits, labels, n):
    """
    Convert a set of compatible splits to a Newick string.
    """
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
        
    # Build a hierarchy
    # We can use a recursive partitioning approach.
    # Find a split that divides the leaves into two non-empty sets.
    # Recursively build subtrees.
    
    # To make it deterministic, pick the split with the smallest size (cherries first)
    # This ensures we resolve cherries first.
    
    def build_subtree(leaves):
        if len(leaves) == 1:
            return labels[leaves[0]]
        if len(leaves) == 2:
            return f"({labels[leaves[0]]},{labels[leaves[1]]})"
            
        # Find a split that partitions 'leaves'
        # The split must be such that one side is a subset of 'leaves'
        # and the other side is 'leaves' - side.
        # We look for a split s in our global splits such that:
        # s & leaves is non-empty and not all of leaves.
        
        best_split = None
        min_size = len(leaves)
        
        for s in splits:
            inter = s & frozenset(leaves)
            if len(inter) > 0 and len(inter) < len(leaves):
                # This split divides leaves
                if len(inter) < min_size:
                    min_size = len(inter)
                    best_split = inter
                    if min_size == 2: # Cherry found, good
                        break
                        
        if best_split is None:
            # No split divides this set? This shouldn't happen if we have n-3 splits.
            # Fallback: Star tree for this subtree
            return "(" + ",".join(labels[i] for i in leaves) + ")"
            
        other_leaves = frozenset(leaves) - best_split
        left_tree = build_subtree(best_split)
        right_tree = build_subtree(other_leaves)
        return f"({left_tree},{right_tree})"
        
    all_leaves = frozenset(range(n))
    return build_subtree(all_leaves) + ";"

def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return f"({','.join(labels)});"
    
    # Compute distance matrix
    dist_matrix = _jc_distance_matrix(sequences, labels)
    
    # Get quartet splits
    splits = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    split = _get_quartet_split(i, j, k, l, dist_matrix)
                    splits.append(split)
                    
    # Build tree
    tree = _build_tree_from_splits(splits, labels, n)
    
    return tree
