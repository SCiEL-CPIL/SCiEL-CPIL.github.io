# EVOLVE-BLOCK-START
import time
import math
import numpy as np

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

def _rf_distance(s1, s2):
    return len(s1.symmetric_difference(s2))

def _get_splits_from_tree(tree_str, all_labels):
    """Extract splits from a Newick tree string."""
    labels_set = set(all_labels)
    n = len(labels_set)
    label_to_idx = {l: i for i, l in enumerate(sorted(labels_set, key=lambda x: int(x[1:])))}
    
    def parse_tree(s, pos=0):
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse_tree(s, pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
                else:
                    break
            pos += 1  # skip ')'
            return ('node', children), pos
        else:
            end = pos
            while end < len(s) and s[end] not in ',;':
                end += 1
            label = s[pos:end].strip()
            return ('leaf', label), pos
    
    node, _ = parse_tree(tree_str)
    
    def get_leaf_set(n):
        if n[0] == 'leaf':
            return frozenset([label_to_idx[n[1]]])
        else:
            result = frozenset()
            for child in n[1]:
                result = result | get_leaf_set(child)
            return result
    
    splits = set()
    def traverse(n):
        if n[0] == 'node':
            leaf_set = get_leaf_set(n)
            if 1 < len(leaf_set) < n:
                side1 = leaf_set
                side2 = frozenset(set(range(n)) - leaf_set)
                if tuple(sorted(side1)) < tuple(sorted(side2)):
                    splits.add(side1)
                else:
                    splits.add(side2)
            for child in n[1]:
                traverse(child)
    
    traverse(node)
    return splits

def _build_tree_from_splits(splits, labels):
    """Build a Newick tree from a set of splits using recursive partitioning."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    
    label_set = frozenset(labels)
    n = len(labels)
    
    best_split = None
    best_balance = 0
    
    for s in splits:
        side1 = s & label_set
        side2 = label_set - side1
        if 1 < len(side1) < n and 1 < len(side2) < n:
            balance = min(len(side1), len(side2))
            if balance > best_balance:
                best_balance = balance
                best_split = side1
    
    if best_split is None:
        return "(" + ",".join(labels) + ")"
    
    side1 = sorted(best_split, key=lambda x: int(x[1:]))
    side2 = sorted(label_set - best_split, key=lambda x: int(x[1:]))
    
    tree1 = _build_tree_from_splits(splits, side1)
    tree2 = _build_tree_from_splits(splits, side2)
    
    return f"({tree1},{tree2})"

def _dpp_score(tree_splits, arr, n, L):
    """
    Compute DPP score: sum over splits of (n1*n2/L) * d
    where d is the JC-corrected distance across the split.
    Lower is better.
    """
    if not tree_splits:
        return 0.0
    
    total_score = 0.0
    for split in tree_splits:
        idx1 = [i for i in range(n) if i in split]
        idx2 = [i for i in range(n) if i not in split]
        
        if not idx1 or not idx2:
            continue
        
        n1 = len(idx1)
        n2 = len(idx2)
        
        total_diff = 0
        count_pairs = 0
        for i in idx1:
            for j in idx2:
                total_diff += (arr[i] != arr[j]).sum()
                count_pairs += 1
        
        if count_pairs == 0:
            continue
        
        avg_p = total_diff / (count_pairs * L)
        avg_p = min(avg_p, 0.7499)
        
        d = -0.75 * np.log(1 - (4.0/3.0) * avg_p)
        
        total_score += (n1 * n2 / L) * d
    
    return total_score

def _dpp_search(arr, labels, n, L, time_limit):
    """
    DPP-based topology search using neighbor joining as starting point,
    then local search via split exchanges.
    """
    d_full = _jc_matrix(arr)
    
    tree_full, splits_full = _nj(d_full, labels)
    best_splits = set(splits_full)
    best_score = _dpp_score(splits_full, arr, n, L)
    
    deadline = time.time() + time_limit
    iterations = 0
    
    # Precompute all possible splits for efficiency
    all_leaves_set = frozenset(range(n))
    
    while time.time() < deadline and iterations < 50:
        iterations += 1
        improved = False
        
        # Try all pairs of taxa
        for i in range(n):
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                
                separating_splits = [s for s in best_splits if (i in s) != (j in s)]
                
                if not separating_splits:
                    continue
                
                for old_split in separating_splits:
                    side_with_i = old_split if i in old_split else frozenset(all_leaves_set - old_split)
                    side_without_i = frozenset(all_leaves_set - side_with_i)
                    
                    # Try moving one taxon k from side_with_i to side_without_i
                    for k in side_with_i:
                        if k == i or k == j:
                            continue
                        
                        new_side = frozenset(side_with_i - {k})
                        if len(new_side) <= 1 or len(frozenset(all_leaves_set - new_side)) <= 1:
                            continue
                        
                        other = frozenset(all_leaves_set - new_side)
                        if tuple(sorted(new_side)) < tuple(sorted(other)):
                            norm_new = new_side
                        else:
                            norm_new = other
                        
                        if norm_new in best_splits and norm_new != old_split:
                            continue
                        
                        new_splits = set(best_splits)
                        new_splits.discard(old_split)
                        new_splits.add(norm_new)
                        
                        # Check consistency
                        consistent = True
                        split_list = list(new_splits)
                        for idx1 in range(len(split_list)):
                            for idx2 in range(idx1+1, len(split_list)):
                                s1 = split_list[idx1]
                                s2 = split_list[idx2]
                                side1 = s1
                                side2 = frozenset(all_leaves_set - s1)
                                side3 = s2
                                side4 = frozenset(all_leaves_set - s2)
                                
                                if (side1 & side3) and (side1 & side4) and (side2 & side3) and (side2 & side4):
                                    consistent = False
                                    break
                            if not consistent:
                                break
                        
                        if not consistent:
                            continue
                        
                        new_score = _dpp_score(new_splits, arr, n, L)
                        if new_score < best_score - 1e-10:
                            best_splits = new_splits
                            best_score = new_score
                            improved = True
                            break
        
        if not improved:
            break
    
    tree_from_splits = _build_tree_from_splits(best_splits, labels)
    return tree_from_splits, best_splits

def _bootstrap_nj(arr, labels, n, L, num_bootstraps, time_limit):
    """
    Run bootstrap NJ to get multiple candidate trees.
    """
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    candidates = [(tree_full, splits_full)]
    
    deadline = time.time() + time_limit
    np.random.seed(42)
    
    for _ in range(num_bootstraps):
        if time.time() > deadline:
            break
        
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        candidates.append((tree_boot, splits_boot))
    
    return candidates

def _find_central_tree(candidates, n):
    """
    Find the tree most similar to all others (minimum total RF distance).
    """
    if len(candidates) <= 1:
        return candidates[0] if candidates else (None, set())
    
    best_idx = 0
    best_score = float('inf')
    
    for i, (tree_i, splits_i) in enumerate(candidates):
        score = 0
        for j, (tree_j, splits_j) in enumerate(candidates):
            if i != j:
                score += _rf_distance(splits_i, splits_j)
        score /= (len(candidates) - 1)
        
        if score < best_score:
            best_score = score
            best_idx = i
    
    return candidates[best_idx]

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    L = arr.shape[1]
    
    # Compute distance matrix
    d_full = _jc_matrix(arr)
    
    # Initial NJ tree
    tree_full, splits_full = _nj(d_full, labels)
    
    # Bootstrap NJ to get multiple candidates
    time_for_bootstrap = min(15.0, _time_left() * 0.3)
    num_bootstraps = min(30, max(5, L // 30))
    
    if time_for_bootstrap > 3.0 and n >= 5:
        candidates = _bootstrap_nj(arr, labels, n, L, num_bootstraps, time_for_bootstrap)
        tree_central, splits_central = _find_central_tree(candidates, n)
        
        # Use the central tree as starting point for DPP refinement
        if splits_central:
            best_splits = set(splits_central)
        else:
            best_splits = set(splits_full)
    else:
        best_splits = set(splits_full)
    
    # DPP-based refinement
    time_for_dpp = min(30.0, _time_left() * 0.5)
    if time_for_dpp > 5.0:
        tree_dpp, splits_dpp = _dpp_search(arr, labels, n, L, time_for_dpp)
        score_dpp = _dpp_score(splits_dpp, arr, n, L)
        score_init = _dpp_score(best_splits, arr, n, L)
        
        if score_dpp < score_init:
            return tree_dpp
    
    # Fallback to NJ tree
    return tree_full
# EVOLVE-BLOCK-END
