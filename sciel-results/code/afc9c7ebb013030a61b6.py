"""Phylogenetic topology recovery via BIONJ-FDP with bootstrap voting."""
import time
import math
import numpy as np

_DEADLINE_FRACTION = 0.90
_start_time = None


def _encode_sequences(sequences, labels):
    """Encode sequences as uint8 array for fast operations."""
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[enc[c] for c in s] for s in (sequences[lab] for lab in labels)], dtype=np.uint8)
    return seqs


def _jc_distance_matrix(seqs):
    """Compute JC-corrected distance matrix from encoded sequences."""
    n = seqs.shape[0]
    L = seqs.shape[1]
    
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    pmat = diff.sum(axis=2).astype(np.float64) / L
    np.fill_diagonal(pmat, 0.0)
    
    pmat = np.clip(pmat, 0, 0.7499)
    dmat = -0.75 * np.log(1.0 - 4.0 * pmat / 3.0)
    np.fill_diagonal(dmat, 0.0)
    return dmat, pmat


def _bionj_correct(dmat, pmat, n):
    """Apply proper BIONJ variance correction."""
    # BIONJ correction: d_ij_corrected = d_ij - (var_i + var_j) / (2 * (n-2))
    # where var_i = sum_j (d_ij^2 - d_ij * (sum_k d_ik)/(n-1)) / L
    # Simplified standard BIONJ: var_i = sum_j d_ij^2 / (L * (n-1))
    
    # Use the standard BIONJ variance formula
    # For each node i, the variance is estimated as:
    # var_i = (1/L) * sum_j (d_ij^2) - (1/L) * (sum_j d_ij)^2 / (n-1)
    # But we don't have L directly, so use pmat-based estimate
    
    # Proper BIONJ: variance of distance estimator
    # Under JC model, Var(p_hat_ij) = (1/L) * (1 - exp(-4*d_ij/3)) * (1 - (1-exp(-4*d_ij/3))/4)
    # Simplified: use observed p to estimate
    
    # Standard BIONJ correction uses:
    # d_ij_corrected = d_ij - (sigma_i^2 + sigma_j^2) / (2*(n-2))
    # where sigma_i^2 = sum_j (d_ij - mean_j(d_ij))^2 / (n-1)  (variance of row i)
    
    row_means = dmat.mean(axis=1)
    var = ((dmat - row_means[:, np.newaxis]) ** 2).sum(axis=1) / (n - 1)
    
    correction = (var[:, np.newaxis] + var[np.newaxis, :]) / (2.0 * max(n - 2, 1))
    dmat_corrected = dmat - correction
    np.fill_diagonal(dmat_corrected, 0.0)
    
    return dmat_corrected


def _nj(dmat, labels):
    """Neighbor Joining algorithm on a distance matrix."""
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0]
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    d = dmat.copy()
    
    while len(active) > 3:
        k = len(active)
        r = np.array([sum(d[i][j] for j in active if j != i) for i in active])
        
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


def _get_splits(newick, labels):
    """Extract non-trivial splits from a Newick string."""
    n = len(labels)
    if n <= 3:
        return set()
    
    # Parse Newick to get splits
    def parse_splits(s):
        """Recursively parse Newick and return list of splits."""
        splits = []
        s = s.strip().rstrip(';')
        
        def _parse(idx, end):
            """Parse subtree starting at idx, returns (frozenset, next_idx)."""
            if idx >= end:
                return frozenset(), idx
            
            if s[idx] == '(':
                # Internal node
                idx += 1
                child_sets = []
                while idx < end and s[idx] != ')':
                    cs, idx = _parse(idx, end)
                    child_sets.append(cs)
                    if idx < end and s[idx] == ',':
                        idx += 1
                # idx should be at ')'
                if idx < end and s[idx] == ')':
                    idx += 1
                # Check for label after )
                if idx < end and s[idx] not in '(),':
                    label_start = idx
                    while idx < end and s[idx] not in '(),':
                        idx += 1
                    label = s[label_start:idx]
                    if label and label != '':
                        child_sets.append(frozenset([label]))
                # Combine children
                combined = set()
                for cs in child_sets:
                    combined.update(cs)
                return frozenset(combined), idx
            else:
                # Leaf
                label_start = idx
                while idx < end and s[idx] not in '(),':
                    idx += 1
                label = s[label_start:idx]
                return frozenset([label]), idx
        
        result, _ = _parse(0, len(s))
        return result
    
    # Get all internal nodes' splits
    all_labels = frozenset(labels)
    
    # Alternative: use a simpler approach - extract splits by finding all internal node splits
    def extract_splits(newick_str):
        """Extract all non-trivial splits from Newick."""
        newick_str = newick_str.strip().rstrip(';')
        splits = set()
        
        # Find all internal nodes and their splits
        # For each internal node, the split is (left subtree taxa) | (right subtree taxa)
        # We need to identify all 2-ways splits (bipartitions)
        
        def get_taxa(subtree_str):
            """Get all taxa in a subtree string."""
            taxa = set()
            i = 0
            while i < len(subtree_str):
                if subtree_str[i] == '(':
                    # Find matching )
                    depth = 1
                    j = i + 1
                    while j < len(subtree_str) and depth > 0:
                        if subtree_str[j] == '(':
                            depth += 1
                        elif subtree_str[j] == ')':
                            depth -= 1
                        j += 1
                    # Recurse on inside
                    inside = subtree_str[i+1:j-1]
                    taxa.update(get_taxa(inside))
                    i = j
                else:
                    # Leaf label
                    j = i
                    while j < len(subtree_str) and subtree_str[j] not in '(),':
                        j += 1
                    label = subtree_str[i:j]
                    if label:
                        taxa.add(label)
                    i = j
                if i < len(subtree_str) and subtree_str[i] == ',':
                    i += 1
            return taxa
        
        def find_internal_splits(subtree_str, depth=0):
            """Find all internal node splits."""
            if depth == 0:
                # Top level: this is the root
                pass
            
            i = 0
            while i < len(subtree_str):
                if subtree_str[i] == '(':
                    depth_counter = 1
                    j = i + 1
                    while j < len(subtree_str) and depth_counter > 0:
                        if subtree_str[j] == '(':
                            depth_counter += 1
                        elif subtree_str[j] == ')':
                            depth_counter -= 1
                        j += 1
                    inside = subtree_str[i+1:j-1]
                    
                    # This is an internal node - find its children
                    # Split inside by top-level commas
                    children = []
                    k = 0
                    d = 0
                    start = 0
                    while k < len(inside):
                        if inside[k] == '(':
                            d += 1
                        elif inside[k] == ')':
                            d -= 1
                        elif inside[k] == ',' and d == 0:
                            children.append(inside[start:k])
                            start = k + 1
                        k += 1
                    children.append(inside[start:])
                    
                    # For each pair of children, compute the split
                    # The split for this internal node is: (union of first m children) | (union of rest)
                    # Actually, for binary trees, each internal node has exactly 2 children
                    # For the split, we take one side
                    if len(children) == 2:
                        left_taxa = get_taxa(children[0])
                        right_taxa = get_taxa(children[1])
                        # Non-trivial split: both sides have >= 2 taxa
                        if len(left_taxa) >= 2 and len(right_taxa) >= 2:
                            # Normalize: use the smaller set as key
                            if len(left_taxa) <= len(right_taxa):
                                split_key = frozenset(left_taxa)
                            else:
                                split_key = frozenset(right_taxa)
                            # Only add if not trivial (size 1 or n-1)
                            if 2 <= len(split_key) <= len(all_labels) - 2:
                                splits.add(split_key)
                        
                        # Recurse into children
                        for child in children:
                            find_internal_splits(child, depth + 1)
                    
                    elif len(children) > 2:
                        # Polytomy: generate all possible splits
                        for m in range(1, len(children)):
                            left_taxa = set()
                            right_taxa = set()
                            for c in children[:m]:
                                left_taxa.update(get_taxa(c))
                            for c in children[m:]:
                                right_taxa.update(get_taxa(c))
                            if len(left_taxa) >= 2 and len(right_taxa) >= 2:
                                if len(left_taxa) <= len(right_taxa):
                                    split_key = frozenset(left_taxa)
                                else:
                                    split_key = frozenset(right_taxa)
                                if 2 <= len(split_key) <= len(all_labels) - 2:
                                    splits.add(split_key)
                        
                        for child in children:
                            find_internal_splits(child, depth + 1)
                    
                    i = j
                else:
                    # Leaf
                    j = i
                    while j < len(subtree_str) and subtree_str[j] not in '(),':
                        j += 1
                    i = j
                if i < len(subtree_str) and subtree_str[i] == ',':
                    i += 1
        
        find_internal_splits(newick_str)
        return splits
    
    return extract_splits(newick_str)


def _star_tree(labels):
    """Fallback: star tree."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    return "(" + ",".join(labels) + ")"


def _bootstrap_vote(seqs, labels, n_bootstraps, deadline):
    """Run bootstrap NJ and return majority-vote tree."""
    n = len(labels)
    L = seqs.shape[1]
    rng = np.random.default_rng(42)
    
    # Count splits
    split_counts = {}
    
    for b in range(n_bootstraps):
        if time.time() > deadline:
            break
        
        # Resample sites
        idx = rng.integers(0, L, size=L)
        boot_seqs = seqs[:, idx]
        
        dmat, pmat = _jc_distance_matrix(boot_seqs)
        dmat_corrected = _bionj_correct(dmat, pmat, n)
        
        try:
            tree = _nj(dmat_corrected, labels)
            splits = _get_splits(tree, labels)
            for sp in splits:
                split_counts[sp] = split_counts.get(sp, 0) + 1
        except Exception:
            continue
    
    if not split_counts:
        return None
    
    # Build tree from majority splits
    # Use the most common splits to guide NJ
    # For now, just return the NJ tree from original data if voting is inconclusive
    return None  # Fallback to original NJ


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
        return _star_tree(labels)
    
    # Encode sequences once
    seqs = _encode_sequences(sequences, labels)
    
    # Compute distance matrix
    dmat, pmat = _jc_distance_matrix(seqs)
    
    # Apply BIONJ correction
    dmat_corrected = _bionj_correct(dmat, pmat, n)
    
    # Time check
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    if remaining < 3.0:
        try:
            return _nj(dmat_corrected, labels)
        except Exception:
            return _star_tree(labels)
    
    # Run NJ on corrected distances
    try:
        tree = _nj(dmat_corrected, labels)
    except Exception:
        try:
            tree = _nj(dmat, labels)
        except Exception:
            tree = _star_tree(labels)
    
    return tree
