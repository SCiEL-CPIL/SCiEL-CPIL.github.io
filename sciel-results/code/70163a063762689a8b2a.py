import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _jc_distance_matrix(A):
    """
    Compute Jukes-Cantor distance matrix.
    A: numpy array of shape (n, L) with values 0-3.
    Returns: n x n numpy array.
    """
    n, L = A.shape
    if n < 2:
        return np.zeros((n, n))
    
    # Hamming distances using broadcasting
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming / float(L)
    
    # JC correction
    arg = 1.0 - 4.0 * p / 3.0
    arg = np.clip(arg, 1e-12, 1.0)
    d = -0.75 * np.log(arg)
    
    # Symmetrize and zero diagonal
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d

def _run_nj(dist, labels):
    """
    Neighbor Joining.
    dist: n x n numpy array.
    labels: list of n labels.
    Returns: Newick string.
    """
    n = len(labels)
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    nodes = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    
    # Mutable distance dict
    D = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                D[(i, j)] = dist[i, j]
    
    new_id = n
    while len(nodes) > 2:
        m = len(nodes)
        # Totals
        tot = {x: sum(D[(x, y)] for y in nodes if y != x) for x in nodes}
        
        # Find best pair
        best = float('inf')
        a, b = None, None
        for i_idx, x in enumerate(nodes):
            for y in nodes[i_idx + 1:]:
                crit = (m - 2) * D[(x, y)] - tot[x] - tot[y]
                if crit < best:
                    best = crit
                    a, b = x, y
        
        # Create new node
        node = new_id
        new_id += 1
        tree_str[node] = f"({tree_str[a]},{tree_str[b]})"
        
        # Update distances to new node
        rest = [x for x in nodes if x != a and x != b]
        d_ab = D[(a, b)]
        for x in rest:
            d_ax = D[(a, x)]
            d_bx = D[(b, x)]
            d_new = 0.5 * (d_ax + d_bx - d_ab)
            D[(node, x)] = d_new
            D[(x, node)] = d_new
        
        # Remove a, b, add node
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(node)
    
    a, b = nodes
    tree_str[new_id] = f"({tree_str[a]},{tree_str[b]})"
    return tree_str[new_id] + ";"

def _fitch_cost_tree(tree, A, label_to_idx):
    """
    Compute Fitch parsimony cost of a tree.
    tree: nested dict with 'label' or 'children' keys.
    A: n x L array of ints (0-3).
    label_to_idx: dict mapping label to index.
    Returns: total cost.
    """
    def compute(node):
        if 'label' in node:
            idx = label_to_idx[node['label']]
            return 0, A[idx, :]
        
        child_results = [compute(child) for child in node['children']]
        total_cost = 0
        # For each site, compute Fitch
        sets = [cr[1] for cr in child_results]
        L = A.shape[1]
        result = np.zeros(L, dtype=np.int32)
        for site in range(L):
            site_sets = [s[site] for s in sets]
            inter = site_sets[0]
            for s in site_sets[1:]:
                inter &= s
            if inter != 0:
                result[site] = inter
            else:
                union = site_sets[0]
                for s in site_sets[1:]:
                    union |= s
                result[site] = union
                total_cost += 1
        for cr in child_results:
            total_cost += cr[0]
        return total_cost, result
    
    cost, _ = compute(tree)
    return cost

def _parse_newick(s):
    """Parse Newick into nested dict."""
    pos = 0
    def parse():
        nonlocal pos
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child = parse()
                children.append(child)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            while pos < len(s) and s[pos] not in (',', ';', ')'):
                pos += 1
            return {'children': children}
        else:
            start = pos
            while pos < len(s) and s[pos] not in (',', ';', ')'):
                pos += 1
            return {'label': s[start:pos]}
    tree = parse()
    return tree

def _to_newick(node):
    """Convert tree structure to Newick string."""
    if 'label' in node:
        return node['label']
    return "(" + ",".join(_to_newick(child) for child in node['children']) + ")"

def _get_leaves(node):
    """Get set of leaf labels in subtree."""
    if 'label' in node:
        return {node['label']}
    leaves = set()
    for child in node['children']:
        leaves |= _get_leaves(child)
    return leaves

def _nni_search(tree_newick, A, labels, deadline):
    """
    Perform NNI search to improve tree topology using Fitch parsimony.
    """
    n = len(labels)
    if n < 4:
        return tree_newick
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    tree = _parse_newick(tree_newick)
    current_cost = _fitch_cost_tree(tree, A, label_to_idx)
    
    def find_nni_edges(node, parent=None, parent_child_idx=None):
        edges = []
        if 'children' in node and len(node['children']) == 2:
            if parent is not None and 'children' in parent and len(parent['children']) == 2:
                edges.append((node, parent, parent_child_idx))
            for i, child in enumerate(node['children']):
                edges.extend(find_nni_edges(child, node, i))
        elif 'children' in node:
            for i, child in enumerate(node['children']):
                edges.extend(find_nni_edges(child, node, i))
        return edges
    
    max_rounds = 30
    for _ in range(max_rounds):
        if time.time() > deadline - 15:
            break
        
        edges = find_nni_edges(tree)
        improved = False
        
        for node, parent, pidx in edges:
            if time.time() > deadline - 15:
                break
            
            a = node['children'][0]
            b = node['children'][1]
            c_idx = 1 - pidx
            c = parent['children'][c_idx]
            
            # Try move 0: ((a,b),c) -> ((a,c),b)
            # Save original
            orig_a = _deep_copy_tree(a)
            orig_b = _deep_copy_tree(b)
            orig_c = _deep_copy_tree(c)
            
            new_inner = {'children': [a, c]}
            parent['children'][pidx] = new_inner
            parent['children'][c_idx] = b
            
            new_cost = _fitch_cost_tree(tree, A, label_to_idx)
            
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                # Revert
                parent['children'][pidx] = node
                parent['children'][c_idx] = orig_c
                node['children'] = [orig_a, orig_b]
            
            if improved:
                break
            
            # Try move 1: ((a,b),c) -> ((b,c),a)
            new_inner2 = {'children': [b, c]}
            parent['children'][pidx] = new_inner2
            parent['children'][c_idx] = a
            
            new_cost2 = _fitch_cost_tree(tree, A, label_to_idx)
            
            if new_cost2 < current_cost:
                current_cost = new_cost2
                improved = True
            else:
                # Revert
                parent['children'][pidx] = node
                parent['children'][c_idx] = orig_c
                node['children'] = [orig_a, orig_b]
            
            if improved:
                break
        
        if not improved:
            break
    
    return _to_newick(tree) + ";"

def _deep_copy_tree(tree):
    if 'label' in tree:
        return {'label': tree['label']}
    return {'children': [_deep_copy_tree(child) for child in tree['children']]}

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    L = len(sequences[labels[0]])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    A = np.zeros((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            A[i, j] = char_map[c]
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    start_time = time.time()
    deadline = 255.0
    
    # 1. NJ tree
    D = _jc_distance_matrix(A)
    tree_nj = _run_nj(D, labels)
    
    # 2. NNI refinement
    tree = tree_nj
    if time.time() < deadline - 30:
        tree = _nni_search(tree, A, labels, deadline)
    
    return tree

# EVOLVE-BLOCK-END
