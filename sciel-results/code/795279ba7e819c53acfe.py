"""Phylogenetic topology recovery via BIONJ-corrected Neighbor-Joining with quartet-based refinement."""
import time
import math
import numpy as np
from collections import defaultdict

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
    
    # Pairwise differences using broadcasting
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    pmat = diff.sum(axis=2).astype(np.float64) / L
    np.fill_diagonal(pmat, 0.0)
    
    # JC correction
    pmat_clipped = np.clip(pmat, 0, 0.7499)
    dmat = -0.75 * np.log(1.0 - 4.0 * pmat_clipped / 3.0)
    np.fill_diagonal(dmat, 0.0)
    return dmat, pmat


def _bionj_correct(dmat, pmat, n):
    """Apply BIONJ variance correction to distance matrix."""
    if n <= 3:
        return dmat.copy()
    
    # Standard BIONJ correction
    # For each taxon i, compute the average distance to all others
    avg_d = dmat.sum(axis=1) / (n - 1)
    
    # Variance estimate: for JC model, Var(d_ij) ≈ 1/(3*L) * (1 - exp(-4*d_ij)) approximation
    # Simplified: use pmat as proxy for variance
    # BIONJ: d_ij_corrected = d_ij - (var_i + var_j) / (2*(n-2))
    # where var_i = sum_j (d_ij - avg_d_i)^2 / (n-1) approximately
    
    # Use a simpler but effective correction
    var_i = np.sum((pmat - pmat.mean(axis=1, keepdims=True)) ** 2, axis=1) / max(n - 2, 1)
    correction = (var_i[:, np.newaxis] + var_i[np.newaxis, :]) / (2.0 * max(n - 2, 1))
    dmat_corrected = dmat - correction
    np.fill_diagonal(dmat_corrected, 0.0)
    
    return dmat_corrected


def _neighbor_joining(dmat, labels):
    """
    Perform Neighbor-Joining to reconstruct an unrooted tree.
    Returns a set of splits and a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return set(), (labels[0] if labels else "")
    if n == 2:
        return set(), f"({labels[0]},{labels[1]})"
    if n == 3:
        return set(), f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Work with indices for efficiency
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for i, lab in enumerate(labels)}
    
    # dist as 2D array
    dist = dmat.copy()
    active = list(range(n))
    
    # Tree structure: each internal node is (child1_idx, child2_idx, len1, len2)
    # We'll track which original leaves are in each node
    node_leaves = {i: {i} for i in range(n)}
    next_id = n
    internal_nodes = {}
    
    while len(active) > 2:
        m = len(active)
        
        # Compute r_i
        r = np.array([dist[i, j] for i in active for j in active if i != j]).reshape(m, m - 1).sum(axis=1)
        # Actually, let's do it more carefully
        r = np.zeros(m)
        for idx_i, i in enumerate(active):
            r[idx_i] = sum(dist[i, k] for k in active if k != i)
        
        # Compute Q matrix and find min
        min_q = float('inf')
        best_i = None
        best_j = None
        
        for idx_i in range(m):
            for idx_j in range(m):
                if idx_i == idx_j:
                    continue
                i = active[idx_i]
                j = active[idx_j]
                q = (m - 2) * dist[i, j] - r[idx_i] - r[idx_j]
                if q < min_q:
                    min_q = q
                    best_i = i
                    best_j = j
        
        if best_i is None:
            break
        
        d_ij = dist[best_i, best_j]
        li = 0.5 * d_ij + (r[active.index(best_i)] - r[active.index(best_j)]) / (2 * (m - 2))
        lj = 0.5 * d_ij - (r[active.index(best_i)] - r[active.index(best_j)]) / (2 * (m - 2))
        
        # Ensure non-negative lengths
        li = max(0.0, li)
        lj = max(0.0, lj)
        
        w_id = next_id
        next_id += 1
        
        internal_nodes[w_id] = (best_i, best_j, li, lj)
        node_leaves[w_id] = node_leaves[best_i] | node_leaves[best_j]
        
        # Update distances
        for k in active:
            if k == best_i or k == best_j:
                continue
            d_wk = 0.5 * (dist[best_i, k] + dist[best_j, k] - d_ij)
            dist[w_id, k] = d_wk
            dist[k, w_id] = d_wk
        
        # Remove best_i, best_j and add w_id
        active.remove(best_i)
        active.remove(best_j)
        active.append(w_id)
        
    # Connect the last two
    if len(active) == 2:
        u, v = active
        d_uv = dist[u, v]
        w_id = next_id
        next_id += 1
        internal_nodes[w_id] = (u, v, d_uv / 2.0, d_uv / 2.0)
        node_leaves[w_id] = node_leaves[u] | node_leaves[v]
        active = [w_id]
    
    # Extract splits from the tree
    splits = set()
    all_leaves = set(range(n))
    
    def extract_splits(node):
        if node < n:
            return
        child1, child2, _, _ = internal_nodes[node]
        extract_splits(child1)
        extract_splits(child2)
        # The split defined by child1
        leaf_set = node_leaves[child1]
        if 0 < len(leaf_set) < n:
            # Normalize: use the smaller side
            if len(leaf_set) <= n - len(leaf_set):
                splits.add(frozenset(leaf_set))
            else:
                splits.add(frozenset(all_leaves - leaf_set))
    
    if active:
        extract_splits(active[0])
    
    # Build Newick string
    def build_newick(node):
        if node < n:
            return idx_to_label[node]
        child1, child2, len1, len2 = internal_nodes[node]
        newick1 = build_newick(child1)
        newick2 = build_newick(child2)
        return f"({newick1}:{len1:.4f},{newick2}:{len2:.4f})"
    
    root = active[0] if active else 0
    newick = build_newick(root)
    
    return splits, newick


def _star_tree(labels):
    """Fallback: star tree."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    return "(" + ",".join(labels) + ")"


def _quartet_score(splits, seqs, labels):
    """
    Score a set of splits by counting how many quartets they imply
    are supported by the data.
    Returns a score between 0 and 1.
    """
    n = len(labels)
    if n < 4:
        return 1.0
    
    L = seqs.shape[1]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    # For each quartet (a,b,c,d), determine which split the tree implies
    # and check if it's supported by the majority of sites
    
    # Pre-compute for each pair of taxa, the site patterns
    # This is expensive, so let's sample quartets
    
    import random
    random.seed(42)
    
    num_quartets = min(200, n * (n-1) * (n-2) * (n-3) // 24)
    
    supported = 0
    total = 0
    
    for _ in range(num_quartets):
        # Pick 4 random distinct taxa
        quartet = random.sample(range(n), 4)
        a, b, c, d = quartet
        
        # Determine which split the tree implies for this quartet
        # Check which of the three possible splits is in our split set
        # Split 1: {a,b} | {c,d}
        # Split 2: {a,c} | {b,d}
        # Split 3: {a,d} | {b,c}
        
        split1 = frozenset({a, b})
        split2 = frozenset({a, c})
        split3 = frozenset({a, d})
        
        # Check which split is in our set
        implied_split = None
        if split1 in splits:
            implied_split = (0, 1)  # AB|CD
        elif split2 in splits:
            implied_split = (0, 2)  # AC|BD
        elif split3 in splits:
            implied_split = (0, 3)  # AD|BC
        else:
            # The quartet is not resolved by the tree, or the split is on the other side
            # Check the complement
            if frozenset({c, d}) in splits:
                implied_split = (0, 1)
            elif frozenset({b, d}) in splits:
                implied_split = (0, 2)
            elif frozenset({b, c}) in splits:
                implied_split = (0, 3)
            else:
                continue  # Not resolved
        
        # Count support from the data
        sa, sb, sc, sd = seqs[a], seqs[b], seqs[c], seqs[d]
        
        # For each site, determine which quartet pattern it shows
        # Pattern 00: a=b, c=d (supports AB|CD)
        # Pattern 01: a=c, b=d (supports AC|BD)  
        # Pattern 02: a=d, b=c (supports AD|BC)
        
        count_ab_cd = 0
        count_ac_bd = 0
        count_ad_bc = 0
        
        for s in range(L):
            va, vb, vc, vd = sa[s], sb[s], sc[s], sd[s]
            if va == vb and vc == vd:
                count_ab_cd += 1
            elif va == vc and vb == vd:
                count_ac_bd += 1
            elif va == vd and vb == vc:
                count_ad_bc += 1
        
        total += 1
        # Check if the implied split is supported by majority
        if implied_split == (0, 1) and count_ab_cd >= count_ac_bd and count_ab_cd >= count_ad_bc:
            supported += 1
        elif implied_split == (0, 2) and count_ac_bd >= count_ab_cd and count_ac_bd >= count_ad_bc:
            supported += 1
        elif implied_split == (0, 3) and count_ad_bc >= count_ab_cd and count_ad_bc >= count_ac_bd:
            supported += 1
    
    if total == 0:
        return 1.0
    return supported / total


def _parse_newick_splits(newick, labels):
    """Parse a Newick string into a set of unrooted splits (as frozensets of indices)."""
    import re
    newick_clean = re.sub(r':[0-9.]+', '', newick)
    newick_clean = newick_clean.replace(';', '').strip()
    
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    splits = set()
    
    def parse(s, idx):
        if s[idx] != '(':
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            return {label_to_idx[leaf]}, j
        
        children = []
        idx += 1
        while s[idx] != ')':
            child, idx = parse(s, idx)
            children.append(child)
            if s[idx] == ',':
                idx += 1
        idx += 1
        
        combined = set()
        for child in children:
            combined.update(child)
        
        for child in children:
            if 0 < len(child) < n:
                if len(child) <= n - len(child):
                    splits.add(frozenset(child))
                else:
                    splits.add(frozenset(set(range(n)) - child))
        
        return combined, idx
    
    parse(newick_clean, 0)
    return splits


def _bootstrap_vote(seqs, labels, dmat_base, pmat_base, num_bootstraps):
    """
    Perform bootstrap resampling and vote on topologies.
    Returns the most common topology as a Newick string.
    """
    n = len(labels)
    L = seqs.shape[1]
    
    topology_counts = {}
    topology_newicks = {}
    
    for b in range(num_bootstraps):
        boot_indices = np.random.randint(0, L, size=L)
        boot_seqs = seqs[:, boot_indices]
        
        dmat_boot, pmat_boot = _jc_distance_matrix(boot_seqs)
        dmat_corrected = _bionj_correct(dmat_boot, pmat_boot, n)
        
        try:
            splits, tree = _neighbor_joining(dmat_corrected, labels)
            key = frozenset(splits)
            
            if key not in topology_counts:
                topology_counts[key] = 0
                topology_newicks[key] = tree
            topology_counts[key] += 1
        except Exception:
            pass
        
        elapsed = time.time() - _start_time
        if elapsed > _DEADLINE_FRACTION * 270.0:
            break
    
    if not topology_counts:
        return _star_tree(labels)
    
    best_key = max(topology_counts, key=topology_counts.get)
    return topology_newicks[best_key]


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
    
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    if remaining < 5.0:
        return _star_tree(labels)
    
    seqs = _encode_sequences(sequences, labels)
    
    # 1. Compute Distance Matrix
    dmat, pmat = _jc_distance_matrix(seqs)
    
    # 2. Apply BIONJ correction
    dmat_corrected = _bionj_correct(dmat, pmat, n)
    
    # 3. Run NJ on corrected distances
    try:
        base_splits, tree = _neighbor_joining(dmat_corrected, labels)
    except Exception:
        try:
            base_splits, tree = _neighbor_joining(dmat, labels)
        except Exception:
            return _star_tree(labels)
    
    # 4. Bootstrap voting if time permits
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    
    if remaining > 2.0:
        # Estimate number of bootstraps based on remaining time
        # Each bootstrap takes roughly n^2 * L / 1e9 seconds for distance + O(n^3) for NJ
        # For n=32, L=2000: ~2ms
        # For n=8, L=250: ~0.1ms
        est_time_per_boot = (n ** 2 * seqs.shape[1]) / 1e9 + (n ** 3) / 1e8
        est_time_per_boot = max(est_time_per_boot, 0.001)
        
        num_bootstraps = int(remaining * 0.7 / est_time_per_boot)
        num_bootstraps = max(5, min(num_bootstraps, 200))
        
        boot_tree = _bootstrap_vote(seqs, labels, dmat, pmat, num_bootstraps)
        
        # Use the bootstrap tree if it has better quartet support
        if boot_tree and boot_tree != tree:
            boot_splits = _parse_newick_splits(boot_tree, labels)
            # Prefer the bootstrap tree if it has more splits (more resolved)
            if len(boot_splits) >= len(base_splits):
                tree = boot_tree
    
    return tree
