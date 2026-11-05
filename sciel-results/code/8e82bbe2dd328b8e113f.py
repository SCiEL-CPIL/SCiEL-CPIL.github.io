import numpy as np
import math
import time

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = np.array([list(sequences[label]) for label in labels])
    L = seqs.shape[1]
    
    mismatches = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = mismatches / L
    
    d = np.zeros((n, n))
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0
    np.fill_diagonal(d, 0.0)
    
    return d, labels

def _neighbor_joining(dist, labels):
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    dist_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            dist_dict[(i, j)] = dist[i, j]
            dist_dict[(j, i)] = dist[i, j]
    
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        d_active = np.zeros((m, m))
        for i_idx, ai in enumerate(active):
            for j_idx, aj in enumerate(active):
                if i_idx != j_idx:
                    d_active[i_idx, j_idx] = dist_dict[(ai, aj)]
        
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
                d_new_x = 0.5 * (dist_dict[(best_i, x)] + dist_dict[(best_j, x)] - dist_dict[(best_i, best_j)])
                dist_dict[(new_id, x)] = d_new_x
                dist_dict[(x, new_id)] = d_new_x
        
        active = [x for x in active if x not in (best_i, best_j)]
        active.append(new_id)
        
        trees[new_id] = new_tree
    
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},({trees[b]},{trees[c]}));"
    else:
        return "(" + ",".join(trees[x] for x in active) + ");"

def _parse_newick_to_tree(newick, labels):
    """
    Parse Newick string into an adjacency representation.
    Returns (nodes, adj, leaf_map) where:
      nodes: list of node ids
      adj: dict node_id -> list of (neighbor_id, edge_length)
      leaf_map: dict node_id -> label for leaves
    """
    newick = newick.strip().rstrip(';')
    # We'll use a stack-based parser
    # Each node in the tree structure is represented by a dict with 'children' and 'label'
    
    class Node:
        def __init__(self):
            self.children = []
            self.label = None
            self.parent = None
            self.edge_len = 0.0
    
    def parse(s, pos):
        if pos >= len(s):
            return None, pos
        if s[pos] == '(':
            node = Node()
            pos += 1
            while pos < len(s) and s[pos] != ')':
                child, pos = parse(s, pos)
                if child is not None:
                    node.children.append(child)
                    child.parent = node
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            if pos < len(s) and s[pos] == ')':
                pos += 1
        else:
            node = Node()
            end = pos
            while end < len(s) and s[end] not in ',;':
                end += 1
            label = s[pos:end]
            # Check if there's a branch length
            if label and label[0] == ':':
                label = ''
            node.label = label
            pos = end
        
        # Check for branch length after node
        if pos < len(s) and s[pos] == ':':
            pos += 1
            start = pos
            while pos < len(s) and s[pos] not in ',);':
                pos += 1
            try:
                node.edge_len = float(s[start:pos])
            except:
                node.edge_len = 0.0
        
        return node, pos
    
    root, _ = parse(newick, 0)
    
    # Now convert to adjacency list
    nodes = []
    adj = {}
    leaf_map = {}
    
    def assign_id(node):
        if node is None:
            return None
        if node.label is not None:
            # It's a leaf
            nid = node.label
            nodes.append(nid)
            adj[nid] = []
            leaf_map[nid] = node.label
        else:
            # It's an internal node
            nid = f"_int_{len(nodes)}"
            nodes.append(nid)
            adj[nid] = []
        
        for child in node.children:
            cid = assign_id(child)
            if cid is not None:
                adj[nid].append((cid, child.edge_len))
                adj[cid].append((nid, child.edge_len))
        
        return nid
    
    assign_id(root)
    
    return nodes, adj, leaf_map

def _get_splits_from_tree(adj, labels):
    """
    Get all non-trivial splits from an unrooted tree.
    A split is a partition of the leaf set into two parts.
    We represent each split as a frozenset of the smaller side (or the side containing the first label).
    """
    n = len(labels)
    if n <= 3:
        return set()
    
    label_set = set(labels)
    splits = set()
    
    # For each edge, removing it gives a split
    # We need to find which leaves are on each side
    visited = set()
    
    for edge_node in adj:
        for (neighbor, length) in adj[edge_node]:
            if (edge_node, neighbor) in visited:
                continue
            visited.add((edge_node, neighbor))
            visited.add((neighbor, edge_node))
            
            # BFS/DFS from edge_node avoiding neighbor
            side_a = set()
            stack = [edge_node]
            local_visited = set()
            while stack:
                node = stack.pop()
                if node in local_visited:
                    continue
                local_visited.add(node)
                if node in label_set:
                    side_a.add(node)
                for (nbr, _) in adj[node]:
                    if nbr != neighbor and nbr not in local_visited:
                        stack.append(nbr)
            
            if 0 < len(side_a) < n:
                # Normalize: use the side that contains the first label, or the smaller side
                # To be consistent, we'll use a canonical representation
                # Use frozenset of the side that doesn't contain labels[0], 
                # or if it does, use the other side. Actually, let's just use the smaller side,
                # and if equal size, use the one with the lexicographically smaller first element.
                side_b = label_set - side_a
                if len(side_a) < len(side_b):
                    split = frozenset(side_a)
                elif len(side_b) < len(side_a):
                    split = frozenset(side_b)
                else:
                    # Equal size, use the one with smaller first element
                    a_first = min(side_a)
                    b_first = min(side_b)
                    if a_first < b_first:
                        split = frozenset(side_a)
                    else:
                        split = frozenset(side_b)
                splits.add(split)
    
    return splits

def _robinson_foulds_distance(tree1_splits, tree2_splits):
    """
    Compute normalized Robinson-Foulds distance.
    Returns a value in [0, 1] where 0 means identical splits.
    """
    if not tree1_splits and not tree2_splits:
        return 0.0
    
    common = tree1_splits & tree2_splits
    # Normalized RF distance: 1 - (2 * |intersection|) / (|S1| + |S2|)
    # Or sometimes: (|S1 \ S2| + |S2 \ S1|) / (|S1| + |S2|)
    # The task says "mean normalized Robinson-Foulds distance"
    # Standard normalized RF is usually (|S1| + |S2| - 2|S1∩S2|) / (|S1| + |S2|)
    # This equals 1 - 2|S1∩S2|/(|S1|+|S2|)
    
    s1_size = len(tree1_splits)
    s2_size = len(tree2_splits)
    common_size = len(common)
    
    if s1_size + s2_size == 0:
        return 0.0
    
    return (s1_size + s2_size - 2 * common_size) / (s1_size + s2_size)

def reconstruct(sequences):
    start_time = time.time()
    deadline = 250.0  # Safety margin before 270s kill
    
    d, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    # Check time
    if time.time() - start_time > deadline - 5:
        if n <= 2:
            return ",".join(labels) + ";"
        return "(" + ",".join(labels) + ");"
    
    # NJ with vectorized Q-matrix
    tree_newick = _neighbor_joining(d, labels)
    
    # Verify all leaves are present
    for label in labels:
        if label not in tree_newick:
            # Fallback: simple star tree
            return "(" + ",".join(labels) + ");"
    
    return tree_newick
