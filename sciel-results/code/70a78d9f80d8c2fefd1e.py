import math
import time
import numpy as np
from collections import defaultdict

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor distances using NumPy."""
    n = len(labels)
    if n == 0:
        return np.array([])
    
    seqs = np.array([list(sequences[label]) for label in labels])
    L = seqs.shape[1]
    
    if L == 0:
        return np.zeros((n, n))
        
    # Calculate pairwise Hamming distances
    # diff[i, j] = sum(seqs[i] != seqs[j])
    diff = np.sum(seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :], axis=2)
    
    p = diff / L
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    d[np.isnan(d) | np.isinf(d)] = 10.0
    np.fill_diagonal(d, 0.0)
    
    return d

def _nj_tree_impl(dist_matrix, labels):
    """
    Perform Neighbor Joining.
    Returns Newick string.
    """
    n_init = len(labels)
    if n_init <= 1:
        return labels[0] if n_init == 1 else ""
    if n_init == 2:
        return f"({labels[0]},{labels[1]})"
    
    max_nodes = 2 * n_init
    dist = np.zeros((max_nodes, max_nodes))
    for i in range(n_init):
        for j in range(n_init):
            dist[i, j] = dist_matrix[i, j]
            
    nodes = list(range(n_init))
    trees = {i: labels[i] for i in range(n_init)}
    next_node = n_init
    
    while len(nodes) > 3:
        m = len(nodes)
        r = np.zeros(m)
        for i_idx, a in enumerate(nodes):
            for j_idx, b in enumerate(nodes):
                if i_idx != j_idx:
                    r[i_idx] += dist[a, b]
        
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i_idx in range(m):
            for j_idx in range(i_idx+1, m):
                a = nodes[i_idx]
                b = nodes[j_idx]
                q_val = (m - 2) * dist[a, b] - r[i_idx] - r[j_idx]
                if q_val < best_q:
                    best_q = q_val
                    best_i = i_idx
                    best_j = j_idx
        
        a = nodes[best_i]
        b = nodes[best_j]
        
        new_node = next_node
        next_node += 1
        
        trees[new_node] = f"({trees[a]},{trees[b]})"
        
        for x_idx, x in enumerate(nodes):
            if x not in (a, b):
                dist[new_node, x] = 0.5 * (dist[a, x] + dist[b, x] - dist[a, b])
                dist[x, new_node] = dist[new_node, x]
        
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_node)
        
    if len(nodes) == 3:
        a, b, c = nodes
        temp = f"({trees[a]},{trees[b]})"
        return f"({temp},{trees[c]})"
    else:
        return ",".join(trees[x] for x in nodes)

def _extract_splits(newick, labels_set):
    """
    Extract all splits (clades) from a Newick string.
    Returns a list of frozensets.
    """
    splits = []
    
    def parse(s, pos):
        if pos >= len(s):
            return set(), pos
        if s[pos] == '(':
            pos += 1
            children_sets = []
            while True:
                child_set, pos = parse(s, pos)
                children_sets.append(child_set)
                if pos >= len(s):
                    break
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            while pos < len(s) and s[pos] != ',' and s[pos] != ')' and s[pos] != ';':
                pos += 1
            union_set = set().union(*children_sets) if children_sets else set()
            if 1 < len(union_set) < len(labels_set):
                splits.append(frozenset(union_set))
            return union_set, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in ',);':
                pos += 1
            label = s[start:pos]
            return {label}, pos

    if newick:
        full_set, _ = parse(newick, 0)
    return splits

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0 # Safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    labels_set = set(labels)
    
    if n_taxa <= 3:
        if n_taxa == 1:
            return f"{labels[0]};"
        if n_taxa == 2:
            return f"({labels[0]},{labels[1]})"
        if n_taxa == 3:
            return f"({labels[0]},{labels[1]},{labels[2]})"
            
    dist_matrix = _jc_distance_matrix(sequences, labels)
    initial_newick = _nj_tree_impl(dist_matrix, labels)
    
    if time.time() - start_time > 10:
        return initial_newick

    num_sites = len(sequences[labels[0]])
    split_counts = defaultdict(int)
    n_bootstraps = 0
    max_bootstraps = 500
    
    while n_bootstraps < max_bootstraps:
        if time.time() - start_time > time_limit - 30:
            break
            
        indices = np.random.choice(num_sites, num_sites, replace=True)
        boot_seqs = {label: "".join(sequences[label][i] for i in indices) for label in labels}
        
        boot_dist = _jc_distance_matrix(boot_seqs, labels)
        boot_newick = _nj_tree_impl(boot_dist, labels)
        boot_splits = _extract_splits(boot_newick, labels_set)
        
        for s in boot_splits:
            split_counts[s] += 1
            
        n_bootstraps += 1
        
    if n_bootstraps == 0:
        return initial_newick
        
    threshold = 0.5
    supported_splits = [s for s, c in split_counts.items() if c / n_bootstraps >= threshold]
    
    if not supported_splits:
        return initial_newick
        
    def build_tree(taxa_set, available_splits):
        if len(taxa_set) <= 1:
            return list(taxa_set)[0] if taxa_set else ""
        if len(taxa_set) == 2:
            return f"({list(taxa_set)[0]},{list(taxa_set)[1]})"
        if len(taxa_set) == 3:
            l = list(taxa_set)
            return f"({l[0]},{l[1]},{l[2]})"
            
        best_split = None
        best_support = -1
        
        for s in available_splits:
            if s.issubset(taxa_set) and not s.issuperset(taxa_set):
                support = split_counts[s]
                if support > best_support:
                    best_support = support
                    best_split = s
        
        if best_split is None:
            for s in available_splits:
                if s.issubset(taxa_set) and not s.issuperset(taxa_set):
                    best_split = s
                    break
            
            if best_split is None:
                return "(" + ",".join(list(taxa_set)) + ")"
        
        part1 = best_split
        part2 = taxa_set - best_split
        
        splits_for_part1 = [s for s in available_splits if s.issubset(part1)]
        splits_for_part2 = [s for s in available_splits if s.issubset(part2)]
        
        tree1 = build_tree(part1, splits_for_part1)
        tree2 = build_tree(part2, splits_for_part2)
        
        return f"({tree1},{tree2})"
        
    try:
        consensus_newick = build_tree(labels_set, supported_splits)
        return consensus_newick
    except Exception:
        return initial_newick
