# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import combinations

_START = None
_DEADLINE_FRACTION = 0.88
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

def _quartet_ml_scores(arr):
    """
    For each quartet (i,j,k,l), compute log-likelihood ratio for the three
    unrooted topologies. Returns a dict mapping frozenset({i,j,k,l}) -> (topo, score).
    Topo is a frozenset representing one side of the split (the pair that's together).
    Under JC, the three topologies are:
      1: (i,j)|(k,l)  -> split {i,j}
      2: (i,k)|(j,l)  -> split {i,k}
      3: (i,l)|(j,k)  -> split {i,l}
    """
    n, L = arr.shape
    # For each site, compute the pattern for each quartet
    # We'll accumulate log-likelihood contributions
    
    # For JC model with 4 states, given a topology and branch lengths,
    # we can compute the likelihood. But for scoring quartets without branch lengths,
    # we use the "quartet score" based on pairwise distances.
    
    # A simpler and effective approach: use the sum of pairwise distances
    # within the two groups of the split. Lower total = better fit.
    # This is equivalent to a distance-based quartet method.
    
    # Precompute pairwise difference counts
    diff_counts = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
    
    quartet_scores = {}
    
    for i, j, k, l in combinations(range(n), 4):
        # Three possible splits
        # Split 1: {i,j} vs {k,l}
        s1 = diff_counts[i,j] + diff_counts[k,l]
        # Split 2: {i,k} vs {j,l}
        s2 = diff_counts[i,k] + diff_counts[j,l]
        # Split 3: {i,l} vs {j,k}
        s3 = diff_counts[i,l] + diff_counts[j,k]
        
        best_score = min(s1, s2, s3)
        if s1 == best_score:
            best_split = frozenset([i, j])
        elif s2 == best_score:
            best_split = frozenset([i, k])
        else:
            best_split = frozenset([i, l])
        
        # Normalize: use the smaller side or canonical form
        all4 = frozenset([i, j, k, l])
        other = all4 - best_split
        if len(best_split) <= len(other):
            canonical = best_split
        else:
            canonical = other
        
        # Score = difference between best and worst (higher = more confident)
        margin = max(s1, s2, s3) - best_score
        quartet_scores[frozenset([i, j, k, l])] = (canonical, margin)
    
    return quartet_scores, diff_counts

def _quartet_weighted_tree(quartet_scores, diff_counts, labels):
    """
    Build a tree from quartet votes. For each possible split (pair of taxa vs rest,
    or more generally), accumulate support from quartets.
    """
    n = len(labels)
    all_leaves = frozenset(range(n))
    
    # For each quartet, the best split tells us which pair is "together"
    # We accumulate: for each pair (a,b), count how many quartets support (a,b) being together
    pair_support = {}
    for qset, (split, margin) in quartet_scores.items():
        # split is a 2-element frozenset (the pair together)
        a, b = tuple(sorted(split))
        key = (a, b)
        pair_support[key] = pair_support.get(key, 0.0) + margin
    
    # Now build the tree using a greedy approach:
    # Start with all taxa as separate clusters
    # Repeatedly merge the pair with highest support
    
    clusters = [frozenset([i]) for i in range(n)]
    cluster_map = {i: i for i in range(n)}  # taxon -> cluster index
    
    # Track which clusters each taxon is in
    taxon_cluster = {i: i for i in range(n)}
    
    tree_parts = {i: i for i in range(n)}  # cluster -> tree string index
    
    # We'll build the tree by repeatedly finding the best pair of clusters to merge
    # based on quartet support
    
    # For efficiency, precompute support between any two clusters
    def cluster_pair_support(c1, c2):
        """Sum of quartet margins supporting taxa from c1 and c2 being in the same group."""
        total = 0.0
        for a in c1:
            for b in c2:
                key = (min(a,b), max(a,b))
                total += pair_support.get(key, 0.0)
        return total
    
    # Greedy merge
    active = list(range(n))
    merge_tree = {i: f"t{i}" for i in range(n)}  # cluster id -> newick fragment
    
    while len(active) > 2:
        if _time_left() < 10.0:
            break
        
        best_score = -1
        best_c1, best_c2 = -1, -1
        
        for idx1, c1 in enumerate(active):
            for c2 in active[idx1+1:]:
                score = cluster_pair_support(clusters[c1], clusters[c2])
                if score > best_score:
                    best_score = score
                    best_c1, best_c2 = c1, c2
        
        if best_c1 == -1:
            break
        
        # Merge c1 and c2
        new_cluster = clusters[best_c1] | clusters[best_c2]
        new_id = len(clusters)
        clusters.append(new_cluster)
        
        # Update taxon_cluster
        for t in new_cluster:
            taxon_cluster[t] = new_id
        
        merge_tree[new_id] = f"({merge_tree[best_c1]},{merge_tree[best_c2]})"
        
        active.remove(best_c1)
        active.remove(best_c2)
        active.append(new_id)
    
    # Final tree
    if len(active) == 2:
        final_tree = f"({merge_tree[active[0]]},{merge_tree[active[1]]})"
    else:
        final_tree = merge_tree[active[0]]
    
    return final_tree, pair_support

def _get_splits_from_tree(tree_str, all_labels):
    """Extract splits from a Newick tree string."""
    labels_set = set(all_labels)
    n = len(labels_set)
    
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
            pos += 1
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
            return frozenset([n[1]])
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
                side2 = frozenset(labels_set - leaf_set)
                if tuple(sorted(side1)) < tuple(sorted(side2)):
                    splits.add(side1)
                else:
                    splits.add(side2)
            for child in n[1]:
                traverse(child)
    
    traverse(node)
    return splits

def _splits_to_tree(splits, labels):
    """Convert a set of splits to a Newick tree string."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    
    label_set = frozenset(labels)
    n = len(labels)
    
    # Filter splits to only those that partition the current label set
    valid_splits = []
    for s in splits:
        side1 = s & label_set
        side2 = label_set - side1
        if 1 < len(side1) < n and 1 < len(side2) < n:
            valid_splits.append(side1)
    
    if not valid_splits:
        return "(" + ",".join(labels) + ")"
    
    # Find the most balanced split
    best_split = None
    best_balance = 0
    for s in valid_splits:
        balance = min(len(s), n - len(s))
        if balance > best_balance:
            best_balance = balance
            best_split = s
    
    if best_split is None:
        return "(" + ",".join(labels) + ")"
    
    side1 = sorted(best_split, key=lambda x: int(x[1:]))
    side2 = sorted(label_set - best_split, key=lambda x: int(x[1:]))
    
    tree1 = _splits_to_tree(splits, side1)
    tree2 = _splits_to_tree(splits, side2)
    
    return f"({tree1},{tree2})"

def _dpp_score_vectorized(splits, arr, n, L):
    """
    Compute DPP score using vectorized operations.
    Lower is better.
    """
    if not splits or n < 3:
        return 0.0
    
    # Precompute pairwise difference matrix
    diff_matrix = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
    
    total_score = 0.0
    for split in splits:
        side1 = np.array([1 if l in split else 0 for l in range(n)])
        n1 = side1.sum()
        n2 = n - n1
        if n1 == 0 or n2 == 0:
            continue
        
        # Sum of differences between groups
        cross_diff = diff_matrix[np.ix_(side1.astype(bool), ~side1.astype(bool))].sum()
        
        count_pairs = n1 * n2
        if count_pairs == 0:
            continue
        
        avg_p = cross_diff / (count_pairs * L)
        avg_p = min(avg_p, 0.7499)
        
        d = -0.75 * np.log(1 - (4.0/3.0) * avg_p)
        total_score += (n1 * n2 / L) * d
    
    return total_score

def _dpp_refine(splits, arr, n, L, pair_support, max_iterations=40):
    """
    Refine a set of splits using DPP-guided split exchanges with quartet-informed candidates.
    """
    if n < 4:
        return splits
    
    all_leaves = frozenset(range(n))
    current_splits = set(splits)
    current_score = _dpp_score_vectorized(current_splits, arr, n, L)
    
    # Precompute diff matrix for fast scoring
    diff_matrix = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
    
    for iteration in range(max_iterations):
        if _time_left() < 8.0:
            break
        
        improved = False
        
        # Generate candidate splits from quartet-informed pairs
        # For each pair with high support, create a split that groups it with nearby taxa
        candidate_splits = set()
        
        # Try splits based on pair support: for each top-supported pair, 
        # create a split of {pair} vs rest, and also try to expand the group
        sorted_pairs = sorted(pair_support.items(), key=lambda x: -x[1])
        
        for (a, b), support in sorted_pairs[:min(20, len(sorted_pairs))]:
            if _time_left() < 8.0:
                break
            
            # Candidate: {a,b} vs rest
            new_split = frozenset([a, b])
            other = all_leaves - new_split
            if len(new_split) > len(other):
                new_split = other
            
            if 1 < len(new_split) < n - 1 and new_split not in current_splits:
                candidate_splits.add(new_split)
            
            # Try expanding: add one more taxon to the pair
            for c in range(n):
                if c == a or c == b:
                    continue
                trial = frozenset([a, b, c])
                other = all_leaves - trial
                if len(trial) > len(other):
                    trial = other
                if 1 < len(trial) < n - 1 and trial not in current_splits:
                    candidate_splits.add(trial)
        
        # Also try random balanced splits
        for _ in range(15):
            if _time_left() < 8.0:
                break
            k = np.random.randint(2, n-1)
            indices = np.random.choice(n, size=k, replace=False)
            new_split = frozenset(indices)
            other = frozenset(all_leaves - new_split)
            if len(new_split) > len(other):
                new_split = other
            if 1 < len(new_split) < n - 1 and new_split not in current_splits:
                candidate_splits.add(new_split)
        
        # Check compatibility and try adding
        for new_split in candidate_splits:
            if _time_left() < 8.0:
                break
            
            if new_split in current_splits:
                continue
            
            # Check compatibility with existing splits
            compatible = True
            for existing in current_splits:
                side1 = existing
                side2 = all_leaves - existing
                side3 = new_split
                side4 = all_leaves - new_split
                
                if (side1 & side3) and (side1 & side4) and (side2 & side3) and (side2 & side4):
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # Try adding this split
            trial_splits = current_splits | {new_split}
            trial_score = _dpp_score_vectorized(trial_splits, arr, n, L)
            
            if trial_score < current_score - 1e-10:
                current_splits = trial_splits
                current_score = trial_score
                improved = True
        
        # Try removing splits that don't help
        for existing in list(current_splits):
            if _time_left() < 8.0:
                break
            trial_splits = current_splits - {existing}
            if not trial_splits:
                continue
            trial_score = _dpp_score_vectorized(trial_splits, arr, n, L)
            if trial_score < current_score - 1e-10:
                current_splits = trial_splits
                current_score = trial_score
                improved = True
        
        if not improved:
            break
    
    return current_splits

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    L = arr.shape[1]
    
    if n <= 2:
        return "(" + ",".join(labels) + ")" if n == 2 else labels[0]
    
    # Compute quartet scores and distance matrix
    quartet_scores, diff_counts = _quartet_ml_scores(arr)
    
    # Build initial tree from quartet votes
    tree_quartet, pair_support = _quartet_weighted_tree(quartet_scores, diff_counts, labels)
    
    # Also compute NJ tree as a backup
    d_full = _jc_matrix(arr)
    
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
    
    tree_nj, splits_nj = _nj(d_full, labels)
    
    # Get splits from quartet tree
    splits_quartet = _get_splits_from_tree(tree_quartet, labels)
    
    # Score both trees
    score_quartet = _dpp_score_vectorized(splits_quartet, arr, n, L)
    score_nj = _dpp_score_vectorized(splits_nj, arr, n, L)
    
    # Choose the better starting point
    if score_quartet <= score_nj:
        best_splits = splits_quartet
        best_score = score_quartet
        best_tree = tree_quartet
    else:
        best_splits = splits_nj
        best_score = score_nj
        best_tree = tree_nj
    
    # Refine using DPP with quartet-informed candidates
    if _time_left() > 15.0:
        refined_splits = _dpp_refine(best_splits, arr, n, L, pair_support, max_iterations=40)
        refined_score = _dpp_score_vectorized(refined_splits, arr, n, L)
        
        if refined_score < best_score - 1e-10:
            best_splits = refined_splits
            best_score = refined_score
            best_tree = _splits_to_tree(refined_splits, labels)
    
    return best_tree
# EVOLVE-BLOCK-END
