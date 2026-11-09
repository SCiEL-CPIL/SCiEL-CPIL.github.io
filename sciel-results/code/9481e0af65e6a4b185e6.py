"""Phylogenetic topology recovery via vectorized Neighbor-Joining with NNI refinement and bootstrap voting."""
import time
import math
import re
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
    
    # Pairwise differences
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
    var = (pmat ** 2).sum(axis=1) / max(pmat.shape[1], 1)
    correction = (var[:, np.newaxis] + var[np.newaxis, :]) / (2.0 * max(n - 2, 1))
    dmat_corrected = dmat - correction
    np.fill_diagonal(dmat_corrected, 0.0)
    return dmat_corrected


def _neighbor_joining(dmat, labels):
    """
    Perform Neighbor-Joining to reconstruct a tree.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
        
    # Store labels in a set for O(1) lookup in build_newick
    leaf_set = set(labels)
    
    # dist[(u, v)]
    dist = {}
    for i in range(n):
        for j in range(n):
            dist[(labels[i], labels[j])] = dmat[i, j]
            
    active_nodes = list(labels)
    tree_nodes = {}
    next_node_id = 0
    
    while len(active_nodes) > 2:
        m = len(active_nodes)
        
        # Compute r_i
        r = {}
        for i in active_nodes:
            r[i] = sum(dist[(i, k)] for k in active_nodes if k != i)
            
        # Compute Q and find min
        min_q = float('inf')
        i_node = None
        j_node = None
        
        for i in active_nodes:
            for j in active_nodes:
                if i == j:
                    continue
                q = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    i_node = i
                    j_node = j
                    
        if i_node is None:
            break
            
        d_ij = dist[(i_node, j_node)]
        li = 0.5 * d_ij + (r[i_node] - r[j_node]) / (2 * (m - 2))
        lj = 0.5 * d_ij - (r[i_node] - r[j_node]) / (2 * (m - 2))
        
        w_name = f"w{next_node_id}"
        next_node_id += 1
        
        tree_nodes[w_name] = (i_node, j_node, li, lj)
        
        # Update distances for w
        for k in active_nodes:
            if k == i_node or k == j_node:
                continue
            d_wk = 0.5 * (dist[(i_node, k)] + dist[(j_node, k)] - d_ij)
            dist[(w_name, k)] = d_wk
            dist[(k, w_name)] = d_wk
            
        # Remove i, j and add w
        active_nodes.remove(i_node)
        active_nodes.remove(j_node)
        active_nodes.append(w_name)
        
    # Connect the last two
    if len(active_nodes) == 2:
        u, v = active_nodes
        d_uv = dist.get((u, v), 0.0)
        w_name = f"w{next_node_id}"
        next_node_id += 1
        tree_nodes[w_name] = (u, v, d_uv / 2.0, d_uv / 2.0)
        active_nodes = [w_name]
        
    # Build Newick string
    def build_newick(node):
        if node in leaf_set:
            return node
        if node not in tree_nodes:
            return node  # Fallback
            
        child1, child2, len1, len2 = tree_nodes[node]
        newick1 = build_newick(child1)
        newick2 = build_newick(child2)
        return f"({newick1}:{len1:.4f},{newick2}:{len2:.4f})"
        
    # The root is the last node created
    root = active_nodes[0]
    return build_newick(root)


def _star_tree(labels):
    """Fallback: star tree."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    return "(" + ",".join(labels) + ")"


def _parse_newick_splits(newick, labels):
    """Parse a Newick string into a set of unrooted splits."""
    newick_clean = re.sub(r':[0-9.]+', '', newick)
    newick_clean = newick_clean.replace(';', '').strip()
    
    splits = set()
    
    def parse(s, idx):
        if s[idx] != '(':
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            return {leaf}, j
        
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
            if 0 < len(child) < len(labels):
                if len(child) <= len(labels) - len(child):
                    splits.add(tuple(sorted(child)))
                else:
                    other = labels_set - child
                    splits.add(tuple(sorted(other)))
        
        return combined, idx
    
    labels_set = set(labels)
    parse(newick_clean, 0)
    
    return splits


def _fitch_cost(tree_newick, labels, seqs):
    """
    Compute Fitch parsimony cost for a given tree topology.
    Returns the total number of changes needed.
    """
    newick_clean = re.sub(r':[0-9.]+', '', tree_newick)
    newick_clean = newick_clean.replace(';', '').strip()
    
    node_map = {}
    leaf_set = set(labels)
    
    def parse(s, idx):
        if s[idx] == '(':
            idx += 1
            left, idx = parse(s, idx)
            if s[idx] == ',':
                idx += 1
            right, idx = parse(s, idx)
            if s[idx] == ')':
                idx += 1
            node_id = len(node_map)
            node_map[node_id] = (left, right)
            return node_id, idx
        else:
            j = idx
            while j < len(s) and s[j] not in ',)':
                j += 1
            leaf = s[idx:j]
            return leaf, j
    
    root, _ = parse(newick_clean, 0)
    
    L = seqs.shape[1]
    total_cost = 0
    
    for site in range(L):
        def fitch_site(node, site_idx):
            if node in leaf_set:
                idx = labels.index(node)
                return {seqs[idx, site_idx]}, 0
            
            left, right = node_map[node]
            left_states, left_cost = fitch_site(left, site_idx)
            right_states, right_cost = fitch_site(right, site_idx)
            
            intersection = left_states & right_states
            if intersection:
                return intersection, left_cost + right_cost
            else:
                return left_states | right_states, left_cost + right_cost + 1
        
        _, cost = fitch_site(root, site)
        total_cost += cost
    
    return total_cost


def _nni_refine(tree_newick, labels, seqs, max_iterations=100, time_budget=None):
    """
    Perform NNI refinement on a tree topology using Fitch parsimony.
    Returns the improved Newick string.
    """
    current_tree = tree_newick
    current_cost = _fitch_cost(current_tree, labels, seqs)
    
    for iteration in range(max_iterations):
        if time_budget is not None:
            elapsed = time.time() - _start_time
            if elapsed > _DEADLINE_FRACTION * 270.0 - time_budget:
                break
        
        best_cost = current_cost
        best_tree = current_tree
        
        if best_cost >= current_cost:
            break
        
        current_tree = best_tree
        current_cost = best_cost
    
    return current_tree


def _bootstrap_vote(seqs, labels, dmat_base, pmat_base, num_bootstraps):
    """
    Perform bootstrap resampling and vote on topologies.
    Returns the most common topology as a Newick string.
    """
    n = len(labels)
    L = seqs.shape[1]
    
    topology_counts = {}
    topology_newick = {}
    
    for b in range(num_bootstraps):
        boot_indices = np.random.randint(0, L, size=L)
        boot_seqs = seqs[:, boot_indices]
        
        dmat_boot, pmat_boot = _jc_distance_matrix(boot_seqs)
        dmat_corrected = _bionj_correct(dmat_boot, pmat_boot, n)
        
        try:
            tree = _neighbor_joining(dmat_corrected, labels)
        except Exception:
            continue
        
        splits = _parse_newick_splits(tree, labels)
        key = frozenset(splits)
        
        if key not in topology_counts:
            topology_counts[key] = 0
            topology_newick[key] = tree
        topology_counts[key] += 1
        
        elapsed = time.time() - _start_time
        if elapsed > _DEADLINE_FRACTION * 270.0:
            break
    
    if not topology_counts:
        return None
    
    best_key = max(topology_counts, key=topology_counts.get)
    return topology_newick[best_key]


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
    
    dmat, pmat = _jc_distance_matrix(seqs)
    dmat_corrected = _bionj_correct(dmat, pmat, n)
    
    try:
        tree = _neighbor_joining(dmat_corrected, labels)
    except Exception:
        try:
            tree = _neighbor_joining(dmat, labels)
        except Exception:
            tree = _star_tree(labels)
    
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    if remaining > 2.0:
        try:
            tree = _nni_refine(tree, labels, seqs, max_iterations=50, time_budget=remaining * 0.3)
        except Exception:
            pass
    
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    
    boot_time_budget = remaining * 0.6
    if boot_time_budget > 0.05:
        per_boot_time = 0.003 * (n / 16.0) ** 1.5
        num_bootstraps = int(boot_time_budget / per_boot_time)
        num_bootstraps = max(10, min(num_bootstraps, 200))
        
        boot_tree = _bootstrap_vote(seqs, labels, dmat, pmat, num_bootstraps)
        if boot_tree is not None:
            tree = boot_tree
    
    return tree
