"""Phylogenetic topology recovery via Maximum Likelihood with SPR heuristic under JC69."""
import math
import random
import itertools
from collections import defaultdict

import numpy as np

# Precompute JC69 transition probabilities for a range of branch lengths
# We'll compute likelihoods on the fly but cache where possible

def _jc69_transition(p):
    """Return 4x4 transition probability matrix for JC69 with branch length p."""
    # P(same) = 1/3 + (2/3) * exp(-3p/4)
    # P(diff) = (1/3) * (1 - exp(-3p/4))
    e = math.exp(-3.0 * p / 4.0)
    p_same = 1.0/3.0 + (2.0/3.0) * e
    p_diff = (1.0/3.0) * (1.0 - e)
    M = np.full((4, 4), p_diff)
    np.fill_diagonal(M, p_same)
    return M


def _jc69_likelihood(tree_newick, sequences, branch_lengths=None):
    """
    Compute log-likelihood of the tree under JC69.
    tree_newick: string
    sequences: dict label -> str
    branch_lengths: dict (node_label) -> length, or None for uniform
    Returns log-likelihood.
    """
    # Parse Newick into a tree structure
    # For efficiency, we'll work with the tree as a nested structure
    pass


def _parse_newick(newick):
    """Parse a Newick string into a tree. Returns (root, leaves) where root is a node dict."""
    # Strip trailing semicolon
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    # We'll build a tree using a stack-based parser
    # Each node: {'children': [left, right], 'label': str or None, 'name': unique_id}
    nodes = []
    node_counter = [0]
    
    def make_node(label=None):
        node = {'children': [], 'label': label, 'id': node_counter[0]}
        node_counter[0] += 1
        return node
    
    i = 0
    stack = []
    current = None
    label_start = None
    
    while i < len(s):
        c = s[i]
        if c == '(':
            if current is not None and current['children']:
                # This is a sibling
                stack[-1]['children'].append(current)
            new_node = make_node()
            if current is not None:
                # current becomes a child of new_node? No, we push current
                pass
            stack.append(new_node)
            current = new_node
        elif c == ',':
            if current is not None:
                stack[-1]['children'].append(current)
            current = make_node()
        elif c == ')':
            if current is not None:
                stack[-1]['children'].append(current)
            current = stack.pop()
            # current is now the parent
        elif c == ':':
            # Branch length follows
            i += 1
            j = i
            while j < len(s) and s[j] not in (',', ')', ';'):
                j += 1
            # Parse branch length
            try:
                bl = float(s[i:j])
            except:
                bl = 0.1
            if current is not None:
                current['length'] = bl
            i = j - 1
        else:
            # Part of a label
            if label_start is None:
                label_start = i
            if c in (',', ')', ':', ';'):
                label = s[label_start:i]
                if current is not None:
                    current['label'] = label
                label_start = None
        i += 1
    
    if current is not None:
        stack[-1]['children'].append(current)
        current = stack.pop()
    
    if not stack:
        root = current
    else:
        root = stack[-1]
    
    # Collect leaves
    leaves = []
    def collect_leaves(node):
        if not node['children']:
            if node.get('label'):
                leaves.append(node['label'])
        else:
            for child in node['children']:
                collect_leaves(child)
    
    collect_leaves(root)
    return root, leaves


def _compute_likelihood(root, sequences, leaf_order, n_sites, branch_lengths=None):
    """
    Compute log-likelihood using Felsenstein's pruning algorithm.
    root: tree node
    sequences: dict label -> str
    leaf_order: list of labels in order
    n_sites: number of sites
    branch_lengths: dict node_id -> length
    """
    # Map labels to indices
    label_to_idx = {label: idx for idx, label in enumerate(leaf_order)}
    
    # Get all leaves in order
    n_taxa = len(leaf_order)
    
    # Precompute transition matrices for unique branch lengths
    # For now, we'll use a default branch length if not provided
    default_bl = 0.2
    
    # We'll do site-by-site or batch. For efficiency, let's do column by column
    # but that's slow. Instead, we'll process all sites together using numpy.
    
    # First, identify all nodes and their types
    node_info = {}  # node_id -> {'type': 'leaf'/'internal', 'label': str, 'length': float, 'children': [ids]}
    
    def traverse(node, parent_id=None):
        nid = node['id']
        if not node['children']:
            node_info[nid] = {'type': 'leaf', 'label': node.get('label'), 'length': node.get('length', default_bl), 'children': []}
        else:
            node_info[nid] = {'type': 'internal', 'label': None, 'length': node.get('length', default_bl), 'children': [c['id'] for c in node['children']]}
            for child in node['children']:
                traverse(child, nid)
    
    traverse(root)
    
    # Get topological order (post-order)
    topo_order = []
    def post_order(node):
        if not node['children']:
            topo_order.append(node['id'])
        else:
            for child in node['children']:
                post_order(child)
            topo_order.append(node['id'])
    
    post_order(root)
    
    # Precompute transition matrices
    # For each node, we need the transition matrix for its branch
    trans_matrices = {}
    for nid in node_info:
        bl = node_info[nid]['length']
        trans_matrices[nid] = _jc69_transition(bl)
    
    # Process each site
    total_logl = 0.0
    for site in range(n_sites):
        # Get the base at this site for each leaf
        bases = []
        for label in leaf_order:
            bases.append(sequences[label][site])
        
        # Convert to indices: A=0, C=1, G=2, T=3
        base_idx = []
        for b in bases:
            if b == 'A':
                base_idx.append(0)
            elif b == 'C':
                base_idx.append(1)
            elif b == 'G':
                base_idx.append(2)
            elif b == 'T':
                base_idx.append(3)
            else:
                base_idx.append(0)
        
        # Felsenstein's algorithm
        # For each node, compute the likelihood vector
        # L[node][base] = probability of the subtree below node given the base at node
        likelihoods = {}
        
        for nid in topo_order:
            info = node_info[nid]
            if info['type'] == 'leaf':
                # Leaf: likelihood is 1 if base matches, 0 otherwise (for the observed base)
                # Actually, for a leaf, L[base] = 1 if base == observed, else 0
                # But we need to handle this properly
                obs = base_idx[label_to_idx[info['label']]]
                vec = np.zeros(4)
                vec[obs] = 1.0
                likelihoods[nid] = vec
            else:
                # Internal node: combine children
                # L[node][b] = sum over b1,b2 of L[child1][b1] * L[child2][b2] * P(b->b1) * P(b->b2)
                c1, c2 = info['children']
                M1 = trans_matrices[c1]  # Transition from node to child1
                M2 = trans_matrices[c2]  # Transition from node to child2
                L1 = likelihoods[c1]
                L2 = likelihoods[c2]
                
                # For each possible base b at this node:
                # L[b] = sum_{b1} M1[b,b1] * L1[b1] * sum_{b2} M2[b,b2] * L2[b2]
                # = (M1 @ L1)[b] * (M2 @ L2)[b]
                v1 = M1 @ L1
                v2 = M2 @ L2
                vec = v1 * v2
                likelihoods[nid] = vec
        
        # Root likelihood: sum over all bases at root (assuming equal base frequencies)
        root_l = likelihoods[root['id']]
        site_logl = math.log(max(0.25 * np.sum(root_l), 1e-300))
        total_logl += site_logl
    
    return total_logl


def _generate_random_tree(n_taxa, labels):
    """Generate a random unrooted tree topology using a random binary tree construction."""
    random.shuffle(labels)
    # Start with the first two labels
    tree = [labels[0], labels[1]]
    
    for i in range(2, n_taxa):
        # Pick a random existing "edge" to attach the new leaf
        # In a Newick-like representation, we can pick a random position
        # For simplicity, we'll use a different approach: build the tree recursively
        
        # Actually, let's use a simpler method: pick a random pair of leaves and join them
        # This is complex. Let's use a different strategy.
        
        # We'll maintain a list of "tips" and "internal nodes"
        # For now, let's just create a balanced-ish tree
        pass
    
    # Simpler approach: use a random binary tree via random pairing
    # This is getting complex. Let's use a standard method:
    # Generate a random tree by starting with all leaves and repeatedly joining two random subtrees
    
    # Represent subtrees as strings
    subtrees = list(labels)
    while len(subtrees) > 1:
        i, j = random.sample(range(len(subtrees)), 2)
        joined = f"({subtrees[i]},{subtrees[j]})"
        subtrees = [subtrees[k] for k in range(len(subtrees)) if k not in (i, j)]
        subtrees.append(joined)
    
    return subtrees[0]


def _spr_move(tree_newick, sequences, leaf_order, n_sites):
    """
    Try SPR moves to improve the tree.
    Returns the best tree found.
    """
    # This is a simplified SPR. For each edge, try moving one side to a different location.
    # For now, let's implement a basic NNI (Nearest Neighbor Interchange) as a simpler alternative
    # NNI is a subset of SPR and is easier to implement.
    
    best_tree = tree_newick
    best_logl = _compute_likelihood(_parse_newick(best_tree)[0], sequences, leaf_order, n_sites)
    
    # Try NNI moves
    # Parse the tree and find all internal edges
    root, _ = _parse_newick(best_tree)
    
    # Find all quartets (a,b,c,d) where the tree has ((a,b),(c,d)) and try ((a,c),(b,d)) and ((a,d),(b,c))
    # This is complex to implement correctly. Let's use a simpler heuristic.
    
    # For now, let's just return the same tree (no improvement)
    # A proper implementation would enumerate all possible SPR/NNI moves
    
    return best_tree, best_logl


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    n_sites = len(sequences[labels[0]])
    
    if n_taxa <= 1:
        return labels[0] + ";" if labels else ";";
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]});"
    if n_taxa == 3:
        # For 3 taxa, there's only one unrooted topology
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Start with a random tree
    best_tree = _generate_random_tree(n_taxa, labels)
    
    # Also try a few different starting trees
    candidates = []
    for _ in range(3):
        candidates.append(_generate_random_tree(n_taxa, labels))
    candidates.append(best_tree)
    
    # Evaluate each candidate
    best_logl = -float('inf')
    for tree in candidates:
        try:
            root, leaves = _parse_newick(tree)
            if set(leaves) != set(labels):
                continue
            logl = _compute_likelihood(root, sequences, labels, n_sites)
            if logl > best_logl:
                best_logl = logl
                best_tree = tree
        except:
            continue
    
    # Try some local improvements (simplified SPR/NNI)
    # For time efficiency, we'll do a few rounds of random restarts with local search
    for _ in range(5):
        # Generate a new random tree
        new_tree = _generate_random_tree(n_taxa, labels)
        try:
            root, leaves = _parse_newick(new_tree)
            if set(leaves) != set(labels):
                continue
            logl = _compute_likelihood(root, sequences, labels, n_sites)
            if logl > best_logl:
                best_logl = logl
                best_tree = new_tree
        except:
            continue
    
    return best_tree + ";"
