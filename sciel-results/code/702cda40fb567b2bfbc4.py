# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START = None
_DEADLINE_FRACTION = 0.80

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

def _parse_newick_to_tree(newick, labels):
    """Parse Newick into a tree structure with node indices.
    Returns: nodes (list of dicts), n_nodes, leaf_to_node map
    Each node: {'children': [..], 'is_leaf': bool, 'label': str or None}
    Node 0 is the root.
    """
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    nodes = []
    label_to_node = {}
    
    def parse(s, pos):
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child_id, pos = parse(s, pos)
                children.append(child_id)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            node_id = len(nodes)
            nodes.append({'children': children, 'is_leaf': False, 'label': None})
            return node_id, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in ':,):':
                pos += 1
            label = s[start:pos]
            # Skip optional branch length
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',)':
                    pos += 1
            node_id = len(nodes)
            nodes.append({'children': [], 'is_leaf': True, 'label': label})
            label_to_node[label] = node_id
            return node_id, pos
    
    root_id, _ = parse(newick, 0)
    return nodes, root_id, label_to_node

def _get_postorder(nodes, root_id):
    """Get postorder traversal of tree."""
    order = []
    visited = set()
    
    def dfs(node_id):
        if node_id in visited:
            return
        visited.add(node_id)
        for c in nodes[node_id]['children']:
            dfs(c)
        order.append(node_id)
    
    dfs(root_id)
    return order

def _compute_loglik(nodes, root_id, arr, label_to_node, branch_lengths):
    """Compute log-likelihood under JC model using Felsenstein's pruning algorithm.
    branch_lengths: dict mapping node_id -> length of branch from this node to its parent.
    For root, branch_lengths[root] is ignored (or 0).
    """
    n_taxa = len(label_to_node)
    L = arr.shape[1]
    
    # Precompute: for each node, the label if leaf
    node_label = {}
    for label, node_id in label_to_node.items():
        node_label[node_id] = label
    
    # Build parent map
    parent = {}
    for i, node in enumerate(nodes):
        for c in node['children']:
            parent[c] = i
    
    # Postorder
    postorder = _get_postorder(nodes, root_id)
    
    # For JC model:
    # P(same) = 0.25 + 0.75 * exp(-4/3 * d)
    # P(diff) = 0.25 * (1 - exp(-4/3 * d))
    # Due to symmetry, we only need to track:
    #   a = P(same as parent) contribution
    #   b = P(different from parent) contribution (all 3 diff states are equal)
    # So we can use 2-dimensional vectors instead of 4.
    
    # For each node, store (same, diff) likelihoods relative to parent state.
    # Actually, let's use the standard 4-state approach but exploit JC symmetry.
    # For JC, given a child vector v = [v0, v1, v2, v3], the parent contribution is:
    #   p[s] = 0.25*(1-exp(-4d/3)) * sum_{c != s} v[c] + (0.25+0.75*exp(-4d/3)) * v[s]
    #         = 0.25*(1-e) * (sum(v) - v[s]) + (0.25+0.75*e) * v[s]
    #         = 0.25*(1-e) * sum(v) + (0.25+0.75*e - 0.25*(1-e)) * v[s]
    #         = 0.25*(1-e) * sum(v) + (0.25+0.75*e - 0.25 + 0.25*e) * v[s]
    #         = 0.25*(1-e) * sum(v) + (0.25*e + 0.75*e) * v[s]  -- wait let me redo
    
    # Let e = exp(-4d/3)
    # P(same) = 0.25 + 0.75e
    # P(diff) = 0.25(1-e)
    
    # p[s] = P(same)*v[s] + P(diff)*sum_{c!=s} v[c]
    #       = (0.25+0.75e)*v[s] + 0.25(1-e)*(sum(v)-v[s])
    #       = (0.25+0.75e)*v[s] + 0.25(1-e)*sum(v) - 0.25(1-e)*v[s]
    #       = [(0.25+0.75e) - 0.25(1-e)]*v[s] + 0.25(1-e)*sum(v)
    #       = [0.25+0.75e-0.25+0.25e]*v[s] + 0.25(1-e)*sum(v)
    #       = e*v[s] + 0.25(1-e)*sum(v)
    
    # So p[s] = e*v[s] + 0.25*(1-e)*sum(v)
    # This is very efficient! We only need v[s] and sum(v).
    
    # For a leaf with observed state obs:
    # v[obs] = 1, v[other] = 0
    # sum(v) = 1
    # So parent contribution: p[s] = e*(1 if s==obs else 0) + 0.25*(1-e)*1
    
    log_likelihood = 0.0
    
    for site in range(L):
        # For each node in postorder, compute (v_vector, sum_v)
        # But we only need to propagate up, so we can compute on the fly.
        # Store for each node: the 4-vector or just (v_obs, sum_v) for leaves,
        # and for internal nodes, we need the full 4-vector to compute parent.
        
        # Actually, for internal nodes we need the full 4-vector.
        # But due to JC symmetry at internal nodes (if all subtrees are symmetric),
        # we might not. Let's just use full 4-vectors for correctness.
        
        node_vec = {}  # node_id -> np.array(4)
        
        for node_id in postorder:
            node = nodes[node_id]
            if node['is_leaf']:
                obs = arr[label_to_node[node['label']], site]
                vec = np.zeros(4)
                vec[obs] = 1.0
                node_vec[node_id] = vec
            else:
                # Combine children
                # For each child, propagate up the branch
                # The branch length is the length from child to this node (parent)
                # In our convention, branch_lengths[child] = length from child to parent
                child_vecs = []
                for c in node['children']:
                    bl = branch_lengths.get(c, 0.0)
                    e = math.exp(-4.0 * bl / 3.0)
                    cv = node_vec[c]
                    sum_cv = cv.sum()
                    # p[s] = e*cv[s] + 0.25*(1-e)*sum_cv
                    pv = e * cv + 0.25 * (1.0 - e) * sum_cv
                    child_vecs.append(pv)
                # Multiply all child contributions element-wise
                result = np.ones(4)
                for pv in child_vecs:
                    result *= pv
                node_vec[node_id] = result
        
        # Root likelihood: sum of root vector (since root has no parent)
        root_vec = node_vec[root_id]
        site_lk = root_vec.sum()
        if site_lk > 0:
            log_likelihood += math.log(site_lk)
        else:
            log_likelihood += -1e10  # very unlikely
    
    return log_likelihood

def _nni_refine(tree_newick, arr, labels, max_iterations=50):
    """Refine tree using NNI moves, maximizing JC likelihood.
    Returns refined Newick string and log-likelihood.
    """
    nodes, root_id, label_to_node = _parse_newick_to_tree(tree_newick, labels)
    n_nodes = len(nodes)
    
    # Build parent map
    parent = {}
    for i, node in enumerate(nodes):
        for c in node['children']:
            parent[c] = i
    
    # Set branch lengths (simple heuristic: 0.1 for all)
    branch_lengths = {i: 0.1 for i in range(n_nodes)}
    branch_lengths[root_id] = 0.0
    
    # Compute initial likelihood
    current_ll = _compute_loglik(nodes, root_id, arr, label_to_node, branch_lengths)
    best_ll = current_ll
    best_nodes = [dict(n) for n in nodes]  # deep copy
    
    for iteration in range(max_iterations):
        if _time_left() < 10.0:
            break
        
        improved = False
        # Find all internal edges (u,v) where both are internal nodes
        for u in range(n_nodes):
            if _time_left() < 10.0:
                break
            if nodes[u]['is_leaf']:
                continue
            for v in nodes[u]['children']:
                if nodes[v]['is_leaf']:
                    continue
                # u is parent of v, both internal
                # u must have exactly 2 children, v must have exactly 2 children
                if len(nodes[u]['children']) != 2:
                    continue
                if len(nodes[v]['children']) != 2:
                    continue
                
                # Find the other child of u (w)
                if nodes[u]['children'][0] == v:
                    w = nodes[u]['children'][1]
                else:
                    w = nodes[u]['children'][0]
                
                c, d = nodes[v]['children']
                
                # Try NNI move 1: ((c,w),d) -> restructure
                # Original: u has children [v, w], v has children [c, d]
                # NNI 1: u has children [c, new_v], new_v has children [d, w]
                # NNI 2: u has children [d, new_v], new_v has children [c, w]
                
                # For NNI 1:
                # u.children = [c, v] (keep v but change v's children)
                # v.children = [d, w]
                # Save old state
                old_u_children = nodes[u]['children'][:]
                old_v_children = nodes[v]['children'][:]
                
                # Apply NNI 1
                nodes[u]['children'] = [c, v]
                nodes[v]['children'] = [d, w]
                
                ll1 = _compute_loglik(nodes, root_id, arr, label_to_node, branch_lengths)
                
                if ll1 > best_ll + 1e-10:
                    best_ll = ll1
                    improved = True
                    # Keep this change
                else:
                    # Revert
                    nodes[u]['children'] = old_u_children
                    nodes[v]['children'] = old_v_children
                
                # Try NNI 2:
                old_u_children = nodes[u]['children'][:]
                old_v_children = nodes[v]['children'][:]
                
                nodes[u]['children'] = [d, v]
                nodes[v]['children'] = [c, w]
                
                ll2 = _compute_loglik(nodes, root_id, arr, label_to_node, branch_lengths)
                
                if ll2 > best_ll + 1e-10:
                    best_ll = ll2
                    improved = True
                    # Keep this change
                else:
                    # Revert
                    nodes[u]['children'] = old_u_children
                    nodes[v]['children'] = old_v_children
        
        if not improved:
            break
    
    # Convert back to Newick
    def to_newick(node_id):
        node = nodes[node_id]
        if node['is_leaf']:
            return node['label']
        else:
            return "(" + ",".join(to_newick(c) for c in node['children']) + ")"
    
    return to_newick(root_id), best_ll

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

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    
    # Try NNI refinement if we have time
    if _time_left() > 30.0 and n >= 5:
        try:
            refined_tree, ll = _nni_refine(tree_full, arr, labels, max_iterations=30)
            return refined_tree
        except Exception:
            pass
    
    return tree_full
# EVOLVE-BLOCK-END
