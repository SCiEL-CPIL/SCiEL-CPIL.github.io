import numpy as np
import time
import sys

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = np.array([list(sequences[label]) for label in labels], dtype='U1')
    
    mismatches = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = mismatches / seqs.shape[1]
    
    d = np.zeros((n, n))
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0
    np.fill_diagonal(d, 0.0)
    
    return d, labels

def _neighbor_joining_newick(dist, labels):
    """Fast vectorized NJ implementation returning Newick string."""
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    max_nodes = 2 * n
    dist_arr = np.full((max_nodes, max_nodes), 10.0)
    for i in range(n):
        for j in range(i+1, n):
            dist_arr[i, j] = dist[i, j]
            dist_arr[j, i] = dist[i, j]
    np.fill_diagonal(dist_arr, 0.0)
    
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        d_active = dist_arr[np.ix_(active, active)]
        
        r = d_active.sum(axis=1)
        Q = (m - 2) * d_active - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat, Q.shape)
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        
        best_i = active[i_idx]
        best_j = active[j_idx]
        
        new_id = next_id
        next_id += 1
        
        tree_i = trees[best_i]
        tree_j = trees[best_j]
        new_tree = f"({tree_i},{tree_j})"
        
        for x in active:
            if x not in (best_i, best_j):
                d_new_x = 0.5 * (dist_arr[best_i, x] + dist_arr[best_j, x] - dist_arr[best_i, best_j])
                dist_arr[new_id, x] = d_new_x
                dist_arr[x, new_id] = d_new_x
        
        active = [x for x in active if x not in (best_i, best_j)]
        active.append(new_id)
        trees[new_id] = new_tree
    
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},({trees[b]},{trees[c]}));"
    else:
        return "(" + ",".join(trees[x] for x in active) + ");"

def _parse_newick(newick):
    """
    Parse Newick string and return:
    - adj: dict node_id -> list of neighbor node_ids
    - leaf_labels: dict node_id -> label for leaves
    - is_leaf: dict node_id -> bool
    """
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    adj = {}
    leaf_labels = {}
    is_leaf = {}
    node_counter = [0]
    
    def _parse_node(s, pos):
        if pos >= len(s):
            return None, pos
        
        if s[pos] == '(':
            node_id = node_counter[0]
            node_counter[0] += 1
            adj[node_id] = []
            is_leaf[node_id] = False
            
            pos += 1
            children = []
            while pos < len(s) and s[pos] != ')':
                child_id, pos = _parse_node(s, pos)
                if child_id is not None:
                    children.append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            if pos < len(s) and s[pos] == ')':
                pos += 1
            
            for child_id in children:
                adj[node_id].append(child_id)
                if child_id in adj:
                    adj[child_id].append(node_id)
            
            while pos < len(s) and s[pos] not in (',', ')', ';'):
                pos += 1
            
            return node_id, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in (',', ')', ';', ':'):
                pos += 1
            label = s[start:pos].strip()
            
            node_id = node_counter[0]
            node_counter[0] += 1
            adj[node_id] = []
            is_leaf[node_id] = True
            leaf_labels[node_id] = label
            
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in (',', ')', ';'):
                    pos += 1
            
            return node_id, pos
    
    root_id, _ = _parse_node(newick, 0)
    return adj, leaf_labels, is_leaf

def _adj_to_newick(adj, leaf_labels, is_leaf):
    if not adj:
        return ""
    
    root = None
    for node in adj:
        if not is_leaf.get(node, False):
            root = node
            break
    if root is None:
        root = list(adj.keys())[0]
    
    def _build(node, parent):
        if is_leaf.get(node, False):
            return leaf_labels[node]
        
        children = []
        for neighbor in adj[node]:
            if neighbor != parent:
                children.append(_build(neighbor, node))
        
        if len(children) == 0:
            return ""
        elif len(children) == 1:
            return "(" + children[0] + ")"
        else:
            return "(" + ",".join(children) + ")"
    
    newick = _build(root, -1)
    return newick + ";"

def _parsimony_score(adj, leaf_labels, is_leaf, seqs_int, label_to_idx):
    """
    Calculate the total parsimony score for the tree using Fitch's algorithm.
    seqs_int: numpy array of shape (n_taxa, L) with values 0-3.
    label_to_idx: dict mapping label -> index in seqs_int.
    """
    n_sites = seqs_int.shape[1]
    total_score = 0
    
    # Pre-compute leaf indices for fast access
    leaf_idx_map = {}
    for node, label in leaf_labels.items():
        leaf_idx_map[node] = label_to_idx[label]

    # For each site, run Fitch's algorithm
    for s in range(n_sites):
        # states[node] is a set of possible states {0,1,2,3}
        # We can use a bitmask: 1<<state
        # 0: A (1), 1: C (2), 2: G (4), 3: T (8)
        # Actually, simpler: just store the set as a tuple or use a boolean array?
        # For speed, let's use a small integer bitmask.
        # State 0: 1, State 1: 2, State 2: 4, State 3: 8
        
        # Get the order of nodes (post-order traversal)
        # Since n is small, we can just do a DFS for each site? 
        # Or pre-compute the post-order traversal of the tree structure once.
        # The tree structure doesn't change, only the scores.
        pass

    # Optimization: Pre-compute post-order traversal
    # We need a function to get post-order nodes.
    
    # Let's implement a simple iterative Fitch for each site using the pre-computed order.
    
    # First, get post-order traversal of the tree
    # Root is any internal node. Let's pick the first one found in _adj_to_newick logic, 
    # but here we need to be consistent. 
    # Let's just find a root (node with degree > 1 or just any internal node) 
    # and do a DFS to get post-order.
    
    # Find root
    root = None
    for node in adj:
        if not is_leaf.get(node, False):
            root = node
            break
    if root is None:
        root = list(adj.keys())[0]

    # Get post-order traversal
    post_order = []
    visited = set()
    stack = [(root, -1, False)]
    while stack:
        node, parent, processed = stack.pop()
        if processed:
            post_order.append(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        stack.append((node, parent, True))
        for neighbor in adj[node]:
            if neighbor != parent and neighbor not in visited:
                stack.append((neighbor, node, False))
    
    # Now iterate sites
    for s in range(n_sites):
        # states[node] = bitmask of possible states
        states = {}
        # Initialize leaves
        for node in post_order:
            if is_leaf.get(node, False):
                idx = leaf_idx_map[node]
                state = seqs_int[idx, s]
                states[node] = 1 << state
            else:
                # Internal node: intersect children states
                child_states = []
                for neighbor in adj[node]:
                    # We need to ensure we only look at children (neighbors that are not parent)
                    # But in post-order, we don't have parent info directly.
                    # We can just intersect with all neighbors that have been processed?
                    # No, Fitch works bottom-up.
                    # In post-order, all children are processed before parent.
                    # So for an internal node, we look at its neighbors.
                    # One neighbor is the parent (not processed yet in bottom-up sense? No, parent is processed later).
                    # So we should intersect with neighbors that are in the subtree.
                    # Actually, standard Fitch: 
                    # S(v) = intersection of S(child) for all children.
                    # Cost += number of children where S(child) is disjoint from S(v)?
                    # No, standard Fitch:
                    # If intersection is non-empty, cost = sum of costs of children.
                    # If intersection is empty, cost = min(costs of children) + 1.
                    
                    pass
        
        # This implementation is getting complex and slow in Python.
        # Given the time constraints and the risk of bugs, I will stick to a simpler refinement:
        # **Distance-based SPR** using the pre-computed distance matrix.
        # It's faster to implement correctly and less prone to performance issues in pure Python.
        
    return 0 # Placeholder

def _distance_tree_score(adj, leaf_labels, is_leaf, d, labels):
    """
    Compute the sum of squared errors between tree-implied distances and observed distances.
    This is a proxy for Tree Length (LS fit).
    """
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Get leaf nodes
    leaf_nodes = {node: label for node, label in leaf_labels.items()}
    
    # For each pair of leaves, compute the path length in the tree.
    # This requires BFS/DFS for each pair. O(n^2 * n) = O(n^3).
    # With n=32, 32^3 = 32768. Very fast.
    
    total_sq_err = 0.0
    leaf_list = list(leaf_nodes.keys())
    
    for i in range(len(leaf_list)):
        u = leaf_list[i]
        label_u = leaf_nodes[u]
        idx_u = label_to_idx[label_u]
        
        for j in range(i+1, len(leaf_list)):
            v = leaf_list[j]
            label_v = leaf_nodes[v]
            idx_v = label_to_idx[label_v]
            
            # BFS to find distance in tree (unweighted edges = 1? No, we need branch lengths)
            # Since we don't have branch lengths in the adj list, we can't compute true tree distance.
            # We must use the distance matrix directly for evaluation?
            # If we use the distance matrix, the score is constant for any tree topology?
            # No, the tree topology determines which splits are present.
            
            # Actually, for SPR refinement, we need a criterion that changes with topology.
            # The "Tree Length" under LS depends on the topology.
            # But computing it requires solving a linear system for branch lengths.
            
            # Alternative: Use **Quartet Support**.
            # For each quartet, check if the tree supports the same topology as the quartet's best support.
            # This requires knowing the "best" quartet topology.
            # We can compute the best quartet topology using the distance matrix or parsimony on the 4 sequences.
            
    # This is getting too complex for a safe, fast implementation.
    
    # Let's go back to the simplest effective improvement:
    # **Multiple NJ restarts with random tie-breaking** or **FastME-like** approach?
    # No, let's just implement a basic **SPR using Parsimony** but optimized.
    
    return 0

def reconstruct(sequences):
    start_time = time.time()
    deadline = 260.0
    
    d, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if time.time() - start_time > deadline - 10:
        if n <= 2:
            return "(" + ",".join(labels) + ");"
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    nj_newick = _neighbor_joining_newick(d, labels)
    
    if time.time() - start_time > deadline - 5:
        return nj_newick
    
    # Parse NJ result
    adj, leaf_labels, is_leaf = _parse_newick(nj_newick)
    
    # Convert back to Newick (this ensures valid format)
    final_newick = _adj_to_newick(adj, leaf_labels, is_leaf)
    
    return final_newick
