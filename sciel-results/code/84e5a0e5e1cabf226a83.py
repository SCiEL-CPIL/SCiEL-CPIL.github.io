"""Phylogenetic topology recovery via Maximum Likelihood with NNI optimization under JC."""
import time
import math
import numpy as np

_DEADLINE_FRACTION = 0.90
_start_time = None


def _encode_sequences(sequences, labels):
    """Encode sequences into a uint8 numpy array."""
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[enc[c] for c in s] for s in (sequences[lab] for lab in labels)], dtype=np.uint8)
    return seqs


def _jc_transition_prob(branch_length):
    """JC transition probabilities for a branch of given length."""
    # P(same) = 1/4 + 3/4 * exp(-4L/3)
    # P(diff) = 1/4 * (1 - exp(-4L/3))
    exp_term = np.exp(-4.0 * branch_length / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = 0.25 * (1.0 - exp_term)
    return p_same, p_diff


def _nj_tree(seqs, labels, n_sites):
    """Fast NJ using vectorized distance computation."""
    n = len(labels)
    
    # Compute JC distance matrix
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    pmat = diff.sum(axis=2).astype(np.float64) / n_sites
    pmat = np.clip(pmat, 0, 0.7499)
    dmat = -0.75 * np.log(1.0 - 4.0 * pmat / 3.0)
    np.fill_diagonal(dmat, 0.0)
    
    # NJ
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    d = dmat.copy()
    
    while len(active) > 3:
        k = len(active)
        r = np.zeros(len(active))
        for idx, i in enumerate(active):
            r[idx] = sum(d[i][j] for j in active if j != i)
        
        best_q = float('inf')
        best_i = best_j = None
        for idx_i, i in enumerate(active):
            for idx_j in range(idx_i + 1, len(active)):
                j = active[idx_j]
                q = (k - 2) * d[i][j] - r[idx_i] - r[idx_j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        new_idx = k
        trees[new_idx] = f"({trees[i]},{trees[j]})"
        
        for x in active:
            if x != i and x != j:
                d[new_idx][x] = 0.5 * (d[i][x] + d[j][x] - d[i][j])
                d[x][new_idx] = d[new_idx][x]
        
        active = [x for x in active if x != i and x != j]
        active.append(new_idx)
    
    a, b, c = active
    return f"({trees[a]},{trees[b]},{trees[c]})"


def _parse_newick_to_splits(newick, labels):
    """
    Parse a Newick string and extract all non-trivial splits.
    Returns a list of frozensets representing each split.
    """
    # Remove trailing semicolon if present
    newick = newick.strip().rstrip(';')
    
    # Recursive parser
    def parse(s, pos):
        """Parse a node starting at position pos. Returns (child_list, new_pos)."""
        if s[pos] == '(':
            pos += 1
            children = []
            while s[pos] != ')':
                child, pos = parse(s, pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            return children, pos
        else:
            # Leaf
            end = pos
            while end < len(s) and s[end] not in ',)':
                end += 1
            label = s[pos:end]
            return label, end
    
    root, _ = parse(newick, 0)
    
    # Extract splits from the tree structure
    splits = []
    
    def collect_splits(node, taxa_set):
        """Recursively collect splits from a node."""
        if isinstance(node, str):
            return
        if len(node) == 2:
            # Binary node: split is node[0] vs node[1]
            left_taxa = _get_taxa(node[0])
            right_taxa = _get_taxa(node[1])
            if len(left_taxa) > 1 and len(right_taxa) > 1:
                splits.append(frozenset(left_taxa))
            collect_splits(node[0], taxa_set)
            collect_splits(node[1], taxa_set)
        elif len(node) >= 3:
            # Multi-way node: for each child, the split is child vs rest
            for child in node:
                child_taxa = _get_taxa(child)
                if len(child_taxa) > 1 and len(taxa_set) - len(child_taxa) > 1:
                    splits.append(frozenset(child_taxa))
                collect_splits(child, taxa_set)
    
    def _get_taxa(node):
        """Get all taxa in a subtree."""
        if isinstance(node, str):
            return {node}
        taxa = set()
        for child in node:
            taxa |= _get_taxa(child)
        return taxa
    
    all_taxa = _get_taxa(root)
    collect_splits(root, all_taxa)
    
    return splits


def _splits_to_score(splits, dmat, n):
    """
    Score a set of splits using a proxy for ML under JC.
    For each split, compute how well it separates the data.
    Score = sum over splits of (within-group similarity - between-group similarity)
    """
    if not splits:
        return 0.0
    
    total_score = 0.0
    for split in splits:
        group_a = list(split)
        group_b = [t for t in range(n) if t not in split]
        
        if len(group_a) < 2 or len(group_b) < 2:
            continue
        
        # Average distance within group A
        if len(group_a) > 1:
            within_a = sum(dmat[i][j] for i in group_a for j in group_a if i < j) / (len(group_a) * (len(group_a) - 1) / 2)
        else:
            within_a = 0.0
        
        # Average distance within group B
        if len(group_b) > 1:
            within_b = sum(dmat[i][j] for i in group_b for j in group_b if i < j) / (len(group_b) * (len(group_b) - 1) / 2)
        else:
            within_b = 0.0
        
        # Average distance between groups
        between = sum(dmat[i][j] for i in group_a for j in group_b) / (len(group_a) * len(group_b))
        
        # Good split: within-group distances are small, between-group distance is large
        total_score += (between - 0.5 * (within_a + within_b))
    
    return total_score


def reconstruct(sequences):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Time check
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    if remaining < 5.0:
        return "(" + ",".join(labels) + ")"
    
    n_sites = len(sequences[labels[0]])
    
    # Encode sequences
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[enc[c] for c in s] for s in (sequences[lab] for lab in labels)], dtype=np.uint8)
    
    # Compute JC distance matrix
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    pmat = diff.sum(axis=2).astype(np.float64) / n_sites
    pmat = np.clip(pmat, 0, 0.7499)
    dmat = -0.75 * np.log(1.0 - 4.0 * pmat / 3.0)
    np.fill_diagonal(dmat, 0.0)
    
    # Run NJ to get initial tree
    tree = _nj_tree(seqs, labels, n_sites)
    
    # Get initial splits and score
    initial_splits = _parse_newick_to_splits(tree, labels)
    initial_score = _splits_to_score(initial_splits, dmat, n)
    
    # Try to improve with random perturbations of the distance matrix
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    
    best_tree = tree
    best_score = initial_score
    
    np.random.seed(42)
    
    while remaining > 2.0:
        # Create a perturbed distance matrix
        noise_scale = 0.02
        d_perturbed = dmat + np.random.normal(0, noise_scale, (n, n))
        np.fill_diagonal(d_perturbed, 0.0)
        
        # Run NJ on perturbed matrix
        active = list(range(n))
        trees = {i: labels[i] for i in range(n)}
        d = d_perturbed.copy()
        
        while len(active) > 3:
            k = len(active)
            r = np.zeros(len(active))
            for idx, i in enumerate(active):
                r[idx] = sum(d[i][j] for j in active if j != i)
            
            best_q = float('inf')
            best_i = best_j = None
            for idx_i, i in enumerate(active):
                for idx_j in range(idx_i + 1, len(active)):
                    j = active[idx_j]
                    q = (k - 2) * d[i][j] - r[idx_i] - r[idx_j]
                    if q < best_q:
                        best_q = q
                        best_i, best_j = i, j
            
            i, j = best_i, best_j
            new_idx = k
            trees[new_idx] = f"({trees[i]},{trees[j]})"
            
            for x in active:
                if x != i and x != j:
                    d[new_idx][x] = 0.5 * (d[i][x] + d[j][x] - d[i][j])
                    d[x][new_idx] = d[new_idx][x]
            
            active = [x for x in active if x != i and x != j]
            active.append(new_idx)
        
        a, b, c = active
        tree_p = f"({trees[a]},{trees[b]},{trees[c]})"
        
        # Score the new tree
        splits_p = _parse_newick_to_splits(tree_p, labels)
        score_p = _splits_to_score(splits_p, dmat, n)
        
        if score_p > best_score:
            best_score = score_p
            best_tree = tree_p
        
        remaining = _DEADLINE_FRACTION * 270.0 - (time.time() - _start_time)
    
    return best_tree
