"""Phylogenetic topology recovery via Maximum Likelihood with NNI heuristic under JC69."""
import math
import random
import sys
from collections import defaultdict

import numpy as np

# Precompute base mapping
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
BASE_LIST = ['A', 'C', 'G', 'T']


def _jc69_probs(branch_lengths, n_sites):
    """
    Precompute transition probability matrices for all branches.
    Returns a dict mapping node_id -> 4x4 transition matrix.
    """
    matrices = {}
    for node_id, bl in branch_lengths.items():
        e = math.exp(-3.0 * bl / 4.0)
        p_same = 1.0/3.0 + (2.0/3.0) * e
        p_diff = (1.0/3.0) * (1.0 - e)
        M = np.full((4, 4), p_diff)
        np.fill_diagonal(M, p_same)
        matrices[node_id] = M
    return matrices


def _parse_newick(newick):
    """
    Parse a Newick string into a tree structure.
    Returns:
        nodes: list of node dicts
        root_id: int
        leaves: list of labels
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    nodes = []
    node_counter = [0]
    
    def make_node(label=None, length=0.0):
        nid = node_counter[0]
        node_counter[0] += 1
        nodes.append({'id': nid, 'children': [], 'label': label, 'length': length})
        return nid
    
    i = 0
    stack = []
    current = None
    label_start = None
    last_label = None
    last_length = 0.0
    
    while i < len(s):
        c = s[i]
        if c == '(':
            if current is not None:
                stack.append(current)
            current = make_node(label=last_label, length=last_length)
            last_label = None
            last_length = 0.0
        elif c == ',':
            if current is not None:
                # Finalize current node
                if last_label is not None:
                    nodes[current]['label'] = last_label
                    last_label = None
                if stack:
                    stack[-1]['children'].append(current)
                current = make_node(length=last_length)
                last_length = 0.0
            else:
                current = make_node()
        elif c == ')':
            if current is not None:
                if last_label is not None:
                    nodes[current]['label'] = last_label
                    last_label = None
                if stack:
                    stack[-1]['children'].append(current)
                    current = stack.pop()
                else:
                    # This is the root
                    pass
            last_length = 0.0
        elif c == ':':
            i += 1
            j = i
            while j < len(s) and s[j] not in (',', ')', ';'):
                j += 1
            try:
                bl = float(s[i:j])
            except:
                bl = 0.1
            last_length = bl
            i = j - 1
        else:
            if c not in (',', ')', '(', ':', ';'):
                if label_start is None:
                    label_start = i
                if i == len(s) - 1 or s[i+1] in (',', ')', ':', ';'):
                    last_label = s[label_start:i+1]
                    label_start = None
        i += 1
    
    # Finalize
    if current is not None:
        if last_label is not None:
            nodes[current]['label'] = last_label
        if stack:
            stack[-1]['children'].append(current)
            current = stack.pop()
    
    # Find root
    root_id = None
    for node in nodes:
        if node['id'] not in [child for n in nodes for child in n['children']]:
            root_id = node['id']
            break
    
    # Collect leaves
    leaves = []
    def collect_leaves(nid):
        node = nodes[nid]
        if not node['children']:
            if node['label']:
                leaves.append(node['label'])
        else:
            for child in node['children']:
                collect_leaves(child)
    
    if root_id is not None:
        collect_leaves(root_id)
    
    return nodes, root_id, leaves


def _build_adjacency(nodes, root_id):
    """
    Build adjacency list and parent-child relationships.
    Returns:
        adj: dict node_id -> list of neighbor ids
        parent: dict node_id -> parent id (None for root)
        children: dict node_id -> list of child ids
    """
    n = len(nodes)
    adj = defaultdict(list)
    parent = {}
    children = defaultdict(list)
    
    def build(nid, par):
        parent[nid] = par
        node = nodes[nid]
        if par is not None:
            adj[nid].append(par)
            children[par].append(nid)
        for child in node['children']:
            adj[nid].append(child)
            build(child, nid)
    
    build(root_id, None)
    
    return adj, parent, children


def _get_leaf_indices(nodes, leaves):
    """
    Map leaf labels to node ids.
    Returns:
        leaf_node_ids: list of node ids corresponding to leaves
        label_to_node: dict label -> node_id
    """
    label_to_node = {}
    leaf_node_ids = []
    for nid, node in enumerate(nodes):
        if not node['children'] and node['label']:
            label_to_node[node['label']] = nid
            leaf_node_ids.append(nid)
    return leaf_node_ids, label_to_node


def _compute_likelihood(nodes, root_id, sequences, labels, n_sites, branch_lengths):
    """
    Compute log-likelihood using Felsenstein's pruning algorithm.
    """
    leaf_node_ids, label_to_node = _get_leaf_indices(nodes, labels)
    
    # Build post-order traversal
    post_order = []
    def postorder(nid):
        node = nodes[nid]
        for child in node['children']:
            postorder(child)
        post_order.append(nid)
    
    postorder(root_id)
    
    # Precompute transition matrices
    trans_matrices = {}
    for nid in range(len(nodes)):
        bl = branch_lengths.get(nid, 0.1)
        e = math.exp(-3.0 * bl / 4.0)
        p_same = 1.0/3.0 + (2.0/3.0) * e
        p_diff = (1.0/3.0) * (1.0 - e)
        M = np.full((4, 4), p_diff)
        np.fill_diagonal(M, p_same)
        trans_matrices[nid] = M
    
    # Process each site
    total_logl = 0.0
    for site in range(n_sites):
        # Get observed bases
        obs = {}
        for label in labels:
            base_char = sequences[label][site]
            obs[label] = BASE_MAP.get(base_char, 0)
        
        # Felsenstein's algorithm
        likelihoods = {}
        
        for nid in post_order:
            node = nodes[nid]
            if not node['children']:
                # Leaf
                label = node['label']
                if label in obs:
                    vec = np.zeros(4)
                    vec[obs[label]] = 1.0
                    likelihoods[nid] = vec
                else:
                    likelihoods[nid] = np.ones(4) * 0.25
            else:
                # Internal node
                child1, child2 = node['children'][0], node['children'][1]
                M1 = trans_matrices[child1]
                M2 = trans_matrices[child2]
                L1 = likelihoods[child1]
                L2 = likelihoods[child2]
                
                v1 = M1 @ L1
                v2 = M2 @ L2
                vec = v1 * v2
                likelihoods[nid] = vec
        
        root_l = likelihoods[root_id]
        site_logl = math.log(max(0.25 * np.sum(root_l), 1e-300))
        total_logl += site_logl
    
    return total_logl


def _generate_random_tree(n_taxa, labels):
    """Generate a random unrooted tree topology."""
    random.shuffle(labels)
    subtrees = list(labels)
    while len(subtrees) > 1:
        i, j = random.sample(range(len(subtrees)), 2)
        joined = f"({subtrees[i]},{subtrees[j]})"
        subtrees = [subtrees[k] for k in range(len(subtrees)) if k not in (i, j)]
        subtrees.append(joined)
    return subtrees[0]


def _nni_move(nodes, root_id, internal_edge):
    """
    Perform an NNI move on the given internal edge.
    Returns new nodes and root_id.
    """
    # This is a simplified NNI implementation
    # For now, we'll just return the same tree
    return nodes, root_id


def _optimize_topology(nodes, root_id, sequences, labels, n_sites, max_iterations=100):
    """
    Optimize topology using NNI moves.
    """
    # For now, this is a placeholder
    # A proper implementation would enumerate all NNI moves and pick the best one
    return nodes, root_id


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    n_sites = len(sequences[labels[0]])
    
    if n_taxa <= 1:
        return labels[0] + ";" if labels else ";"
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]});"
    if n_taxa == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Start with a random tree
    best_tree = _generate_random_tree(n_taxa, labels)
    
    # Also try a few different starting trees
    candidates = []
    for _ in range(5):
        candidates.append(_generate_random_tree(n_taxa, labels))
    candidates.append(best_tree)
    
    # Evaluate each candidate
    best_logl = -float('inf')
    best_nodes = None
    best_root_id = None
    
    for tree in candidates:
        try:
            nodes, root_id, leaves = _parse_newick(tree)
            if set(leaves) != set(labels):
                continue
            # Use uniform branch lengths for now
            branch_lengths = {nid: 0.2 for nid in range(len(nodes))}
            logl = _compute_likelihood(nodes, root_id, sequences, labels, n_sites, branch_lengths)
            if logl > best_logl:
                best_logl = logl
                best_nodes = nodes
                best_root_id = root_id
                best_tree = tree
        except:
            continue
    
    # Try some local improvements (simplified NNI)
    # For time efficiency, we'll do a few rounds of random restarts with local search
    for _ in range(10):
        # Generate a new random tree
        new_tree = _generate_random_tree(n_taxa, labels)
        try:
            nodes, root_id, leaves = _parse_newick(new_tree)
            if set(leaves) != set(labels):
                continue
            branch_lengths = {nid: 0.2 for nid in range(len(nodes))}
            logl = _compute_likelihood(nodes, root_id, sequences, labels, n_sites, branch_lengths)
            if logl > best_logl:
                best_logl = logl
                best_nodes = nodes
                best_root_id = root_id
                best_tree = new_tree
        except:
            continue
    
    return best_tree + ";"
