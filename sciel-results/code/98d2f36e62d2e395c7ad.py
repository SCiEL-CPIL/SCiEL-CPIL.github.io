# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import sys
from itertools import combinations

_START = None
_DEADLINE_FRACTION = 0.82

def _now():
    return time.time()

def _init_time():
    global _START
    if _START is None:
        _START = time.time()

def _time_left():
    _init_time()
    return max(0.0, 270.0 * _DEADLINE_FRACTION - (time.time() - _START))

def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.uint8)
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

def _nj(dist, labels):
    n = len(labels)
    if n == 1:
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()
    
    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}
    
    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]
    
    splits = set()
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        total = {a: sum(d[a][b] for b in nodes if b != a) for a in nodes}
        
        best_criterion = float('inf')
        best_a, best_b = -1, -1
        
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        new_id = -1 - len(nodes)
        trees[new_id] = f"({trees[a]},{trees[b]})"
        leaf_sets[new_id] = leaf_sets[a] | leaf_sets[b]
        
        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        if tuple(sorted(side1)) < tuple(sorted(side2)):
            split = side1
        else:
            split = side2
        if 1 < len(split) < n - 1:
            splits.add(split)
            
        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]
            
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
        
    final_tree = "(" + ",".join(trees[x] for x in nodes) + ")"
    return final_tree, splits

def _build_tree_from_splits(labels, split_votes):
    """
    Build a tree from a set of weighted splits using greedy compatible selection.
    Returns Newick string.
    """
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0]
        return f"({labels[0]},{labels[1]})"
    
    all_leaves = set(range(n))
    
    # Sort splits by weight (descending)
    sorted_splits = sorted(split_votes.items(), key=lambda x: -x[1])
    
    # Greedily select compatible splits
    selected = []
    for split, w in sorted_splits:
        if w <= 0:
            break
        side1 = split
        side2 = all_leaves - split
        compatible = True
        for existing in selected:
            ex_s1 = existing
            ex_s2 = all_leaves - existing
            if side1 < ex_s1 or side1 > ex_s1:
                compatible = False; break
            if side1 < ex_s2 or side1 > ex_s2:
                compatible = False; break
            if side2 < ex_s1 or side2 > ex_s1:
                compatible = False; break
            if side2 < ex_s2 or side2 > ex_s2:
                compatible = False; break
        if compatible:
            selected.append(split)
    
    # If we have enough splits, build tree recursively
    # Use the splits to define a hierarchical clustering
    if len(selected) >= n - 3:
        # Build tree from splits
        # Find the split that partitions the set most evenly or is most supported
        # Recursive approach: find a split that divides the current set
        def build_recursive(leaf_set, splits_available):
            if len(leaf_set) == 1:
                return labels[list(leaf_set)[0]]
            if len(leaf_set) == 2:
                a = list(leaf_set)
                return f"({labels[a[0]]},{labels[a[1]]})"
            
            # Find a split that properly partitions leaf_set
            best_split = None
            best_weight = -1
            for s, w in splits_available:
                s1 = s & leaf_set
                s2 = leaf_set - s
                if 1 < len(s1) < len(leaf_set) and 1 < len(s2) < len(leaf_set):
                    if w > best_weight:
                        best_weight = w
                        best_split = (s1, s2)
            
            if best_split is None:
                # No proper split found, use NJ on this subset
                sub_labels = [labels[i] for i in sorted(leaf_set)]
                sub_arr_indices = sorted(leaf_set)
                # We need the distance matrix for this subset
                # Fallback: just return a star-like tree
                return "(" + ",".join(labels[i] for i in sorted(leaf_set)) + ")"
            
            s1, s2 = best_split
            t1 = build_recursive(s1, splits_available)
            t2 = build_recursive(s2, splits_available)
            return f"({t1},{t2})"
        
        tree = build_recursive(all_leaves, [(s, split_votes[s]) for s in selected])
        return tree
    
    # Fallback: use NJ
    d = _jc_matrix(_arr_ref)
    tree, _ = _nj(d, labels)
    return tree

# Global reference for _build_tree_from_splits
_arr_ref = None

def reconstruct(sequences):
    global _START, _arr_ref
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    L = arr.shape[1]
    _arr_ref = arr
    
    if n <= 2:
        if n == 1:
            return labels[0]
        return f"({labels[0]},{labels[1]})"
    
    # Supermatrix approach:
    # 1. Partition sites into blocks
    # 2. Reconstruct gene tree for each block via NJ
    # 3. Combine via split voting (weighted by block size)
    # 4. Also do bootstrap consensus on full data
    # 5. Choose best candidate
    
    # Determine number of blocks
    # Aim for blocks of ~100-200 sites
    target_block_size = 150
    num_blocks = max(1, min(50, L // target_block_size))
    
    # Create random partition of sites into blocks
    site_indices = np.arange(L)
    np.random.shuffle(site_indices)
    block_sizes = np.full(num_blocks, L // num_blocks)
    block_sizes[:L % num_blocks] += 1
    blocks = []
    start = 0
    for bs in block_sizes:
        blocks.append(site_indices[start:start+bs])
        start += bs
    
    # Reconstruct gene trees for each block
    split_votes = {}
    gene_trees = []
    
    for block in blocks:
        if _time_left() < 10.0:
            break
        arr_block = arr[:, block]
        d_block = _jc_matrix(arr_block)
        tree_block, splits_block = _nj(d_block, labels)
        gene_trees.append((tree_block, splits_block))
        
        # Weight splits by block size
        weight = len(block) / L
        for s in splits_block:
            split_votes[s] = split_votes.get(s, 0.0) + weight
    
    # Also do bootstrap consensus on full data
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    
    # Bootstrap
    num_bootstraps = 0
    max_bootstraps = 500
    boot_split_counts = {}
    
    # Count splits from full NJ
    for s in splits_full:
        boot_split_counts[s] = boot_split_counts.get(s, 0) + 1
    
    while num_bootstraps < max_bootstraps and _time_left() > 10.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        
        for s in splits_boot:
            boot_split_counts[s] = boot_split_counts.get(s, 0) + 1
        
        num_bootstraps += 1
    
    total_boot = num_bootstraps + 1  # +1 for the full NJ
    
    # Combine split votes: supermatrix + bootstrap
    # Bootstrap splits get weight based on frequency
    for s, c in boot_split_counts.items():
        boot_weight = c / total_boot
        # Only add bootstrap splits that appear with reasonable frequency
        if boot_weight > 0.3:
            split_votes[s] = split_votes.get(s, 0.0) + boot_weight * 2.0
    
    # Build candidate trees
    candidates = []
    
    # Candidate 1: Full NJ
    candidates.append((tree_full, splits_full))
    
    # Candidate 2: Tree from combined split votes
    if split_votes:
        tree_splits = _build_tree_from_splits(labels, split_votes)
        # Get splits from this tree by parsing
        # For simplicity, just add it as a candidate
        candidates.append((tree_splits, set()))
    
    # Candidate 3: Bootstrap consensus (majority splits)
    if num_bootstraps > 0:
        majority_splits = {s for s, c in boot_split_counts.items() if c > 0.5 * total_boot}
        if majority_splits:
            # Build tree from majority splits
            maj_votes = {s: boot_split_counts[s] / total_boot for s in majority_splits}
            tree_maj = _build_tree_from_splits(labels, maj_votes)
            candidates.append((tree_maj, set()))
    
    # Score candidates by split agreement with the combined vote
    # Use the split_votes as reference
    best_candidate = tree_full
    best_score = -1
    
    for tree, splits in candidates:
        if not splits:
            # Need to extract splits from tree - use a simple heuristic
            # Count how many of the top split_votes appear in this tree
            # For trees without extracted splits, use NJ distance as proxy
            continue
        common = sum(1 for s in splits if s in split_votes)
        if common > best_score:
            best_score = common
            best_candidate = tree
    
    # If no splits-based candidate was better, use the one with most overlapping top splits
    if best_score < 0:
        # Fallback: use full NJ
        best_candidate = tree_full
    
    return best_candidate

# EVOLVE-BLOCK-END
