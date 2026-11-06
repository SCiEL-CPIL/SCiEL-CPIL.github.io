<<<<<<< SEARCH
def _build_newick(tree, labels_map):
    """
    Recursively build Newick string from the internal tree structure.
    tree: dict or string node name
    labels_map: maps internal node ids to their structure
    """
    if isinstance(tree, str):
        return tree
    if len(tree) == 2:
        left = _build_newick(tree[0], labels_map)
        right = _build_newick(tree[1], labels_map)
        return f"({left},{right})"
    return str(tree)

def _compute_splits(tree_struct, final_node, labels):
    """
    Compute the set of non-trivial splits (frozensets) from a tree structure.
    Returns a set of frozensets.
    """
    n = len(labels)
    if n < 4:
        return set()
    
    # Build adjacency list
    adj = {}
    
    def _add_edge(u, v):
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    
    def _build_adj(node):
        if isinstance(node, str):
            return node
        left, right = node
        left_id = _build_adj(left)
        right_id = _build_adj(right)
        _add_edge(node, left_id)
        _add_edge(node, right_id)
        return node
    
    # We need to traverse the tree and record internal edges
    # A split is defined by each internal edge: the set of leaves on one side
    
    # Build a map from node to its children
    def _get_leaves(node):
        if isinstance(node, str):
            return frozenset([node])
        left, right = node
        return _get_leaves(left) | _get_leaves(right)
    
    def _collect_splits(node, parent=None):
        if isinstance(node, str):
            return []
        splits = []
        left, right = node
        splits.extend(_collect_splits(left, node))
        splits.extend(_collect_splits(right, node))
        return splits
    
    # Better approach: for each internal node, the split is the leaf set under that node
    # vs the rest. Only consider splits where both sides have >= 2 leaves.
    
    all_splits = set()
    
    def _recurse(node):
        if isinstance(node, str):
            return frozenset([node])
        left, right = node
        left_leaves = _recurse(left)
        right_leaves = _recurse(right)
        # The split defined by the edge from this node to its parent
        # We record the leaf set of this subtree
        if 2 <= len(left_leaves) <= n - 2:
            all_splits.add(left_leaves)
        if 2 <= len(right_leaves) <= n - 2:
            all_splits.add(right_leaves)
        return left_leaves | right_leaves
    
    _recurse(tree_struct[final_node])
    
    # Deduplicate: a split {A, B} and {complement} are the same
    # We already only record subtrees, so each split appears once (as the smaller side or the subtree side)
    # Actually we may have both S and complement if both subtrees at a node have size in [2, n-2]
    # Let's normalize: for each split, keep the canonical form (smaller set, or tie-break)
    canonical = set()
    for s in all_splits:
        comp = frozenset(labels) - s
        if len(s) < len(comp) or (len(s) == len(comp) and min(s) < min(comp)):
            canonical.add(s)
        else:
            canonical.add(comp)
    
    return canonical

def _run_nj(dist_mat, labels, time_start, time_limit):
    """
    Run Neighbor-Joining and return (tree_struct, final_node).
    """
    nodes = list(range(len(labels)))
    D = dist_mat.copy()
    tree_struct = {i: labels[i] for i in range(len(labels))}
    
    while len(nodes) > 3:
        if time.time() - time_start > time_limit:
            return None, None
        
        n = len(nodes)
        r = D.sum(axis=1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        flat_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = flat_idx
        if i == j:
            continue
        
        node_i = nodes[i]
        node_j = nodes[j]
        
        new_node = f"n_{i}_{j}"
        tree_struct[new_node] = (tree_struct[node_i], tree_struct[node_j])
        
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[i] = False
        keep_mask[j] = False
        kept_indices = np.where(keep_mask)[0]
        
        new_nodes_list = [nodes[k] for k in kept_indices]
        new_nodes_list.append(new_node)
        
        new_n = n - 1
        D_new = np.zeros((new_n, new_n), dtype=np.float64)
        
        D_keep_keep = D[np.ix_(kept_indices, kept_indices)]
        D_new[:-1, :-1] = D_keep_keep
        
        dist_i_kept = D[kept_indices, i]
        dist_j_kept = D[kept_indices, j]
        dist_ij = D[i, j]
        
        new_branches = 0.5 * (dist_i_kept + dist_j_kept - dist_ij)
        
        D_new[:-1, -1] = new_branches
        D_new[-1, :-1] = new_branches
        
        nodes = new_nodes_list
        D = D_new
    
    if len(nodes) == 3:
        a, b, c = nodes
        tree_struct['final_root'] = (tree_struct[a], (tree_struct[b], tree_struct[c]))
        final_node = 'final_root'
    elif len(nodes) == 2:
        a, b = nodes
        tree_struct['final_root'] = (tree_struct[a], tree_struct[b])
        final_node = 'final_root'
    else:
        final_node = nodes[0]
    
    return tree_struct, final_node

def _quartet_score(tree_struct, final_node, labels, dist_mat):
    """
    Compute a quartet-based score: for each quartet of leaves, count how many
    of the 3 possible topologies are consistent with the tree.
    Lower is better. Uses the distance matrix as a proxy for correctness.
    """
    n = len(labels)
    if n < 4:
        return 0.0
    
    label_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Get all splits from the tree
    all_splits = set()
    
    def _recurse(node):
        if isinstance(node, str):
            return frozenset([node])
        left, right = node
        left_leaves = _recurse(left)
        right_leaves = _recurse(right)
        if 2 <= len(left_leaves) <= n - 2:
            all_splits.add(frozenset(left_leaves))
        if 2 <= len(right_leaves) <= n - 2:
            all_splits.add(frozenset(right_leaves))
        return left_leaves | right_leaves
    
    _recurse(tree_struct[final_node])
    
    # For scoring, we use the "quartet consistency" approach:
    # For each quartet {a,b,c,d}, the tree induces one of 3 topologies.
    # The "correct" topology under NJ is determined by the distance matrix:
    # the topology that minimizes total branch length.
    
    score = 0.0
    n_quartets = 0
    
    for a, b, c, d in itertools.combinations(range(n), 4):
        la, lb, lc, ld = labels[a], labels[b], labels[c], labels[d]
        
        # Determine which topology the tree induces
        # Check if {a,b} is split from {c,d} by any edge in the tree
        # A split S separates {a,b} from {c,d} if exactly two of {a,b,c,d} are in S
        # and the two in S are {a,b} or {c,d}
        
        # For each split, check what partition it induces on the 4 leaves
        def split_induces(s):
            """Returns which pair is on which side, or None if split doesn't separate"""
            members = {la, lb, lc, ld}
            in_s = members & s
            if len(in_s) == 1 or len(in_s) == 3:
                return None  # doesn't separate cleanly
            if len(in_s) == 2:
                # The two in S vs the two not in S
                return frozenset(in_s)
            return None
        
        # The tree's topology on this quartet is determined by the two internal splits
        # that each separate the 4 leaves into 2+2.
        
        separating_splits = []
        for s in all_splits:
            part = split_induces(s)
            if part is not None:
                separating_splits.append(part)
        
        # There should be exactly 2 such splits (the two internal edges of the quartet subtree)
        # These define the topology.
        
        if len(separating_splits) >= 2:
            # The two splits together define the topology
            # Topology 1: (ab)(cd) -> splits {a,b} and {a,c} are NOT the separating ones
            # Let's just count: the quartet topology is (ab|cd) if {a,b} is one of the separating splits
            # Actually, the separating splits are the ones that divide the 4 into 2+2.
            # For topology (ab|cd): the internal edges separate {a,b} from {c,d} and {a,c} from {b,d}? No.
            # For (ab|cd): the two cherries are {a,b} and {c,d}. The internal edge connects them.
            # The splits that separate are: {a,b} (one side) and {c,d} (complement) - but that's the same split.
            # Hmm, I need to think about this more carefully.
            
            pass
        
        n_quartets += 1
    
    return score

# The quartet scoring above is incomplete. Let's use a simpler but effective approach:
# Score = number of splits in the tree that are "consistent" with the distance matrix.
# A split S is consistent if the average distance between leaves in S and leaves not in S
# is greater than the average distance within S or within the complement.

def _score_tree(tree_struct, final_node, labels, dist_mat):
    """
    Score a tree by checking if each split is supported by the distance matrix.
    A split {A, B} is supported if d(A, A) + d(B, B) < d(A, B) * 2 / (|A|*|B|) * ... 
    Simpler: a split is good if the average within-cluster distance is less than the between-cluster distance.
    """
    n = len(labels)
    if n < 4:
        return 0.0
    
    label_idx = {lab: i for i, lab in enumerate(labels)}
    
    all_splits = set()
    
    def _recurse(node):
        if isinstance(node, str):
            return frozenset([node])
        left, right = node
        left_leaves = _recurse(left)
        right_leaves = _recurse(right)
        if 2 <= len(left_leaves) <= n - 2:
            all_splits.add(frozenset(left_leaves))
        if 2 <= len(right_leaves) <= n - 2:
            all_splits.add(frozenset(right_leaves))
        return left_leaves | right_leaves
    
    _recurse(tree_struct[final_node])
    
    score = 0.0
    for s in all_splits:
        if len(s) >= 2 and len(s) <= n - 2:
            idx_s = np.array([label_idx[lab] for lab in s])
            idx_comp = np.array([label_idx[lab] for lab in labels if lab not in s])
            
            # Average within-group distance
            if len(idx_s) >= 2:
                d_ss = dist_mat[np.ix_(idx_s, idx_s)]
                np.fill_diagonal(d_ss, 0)
                within_s = d_ss.sum() / (len(idx_s) * (len(idx_s) - 1))
            else:
                within_s = 0
            
            if len(idx_comp) >= 2:
                d_cc = dist_mat[np.ix_(idx_comp, idx_comp)]
                np.fill_diagonal(d_cc, 0)
                within_c = d_cc.sum() / (len(idx_comp) * (len(idx_comp) - 1))
            else:
                within_c = 0
            
            # Between-group distance
            d_sc = dist_mat[np.ix_(idx_s, idx_comp)]
            between = d_sc.mean()
            
            # A good split has small within and large between
            # Score contribution: between - (within_s + within_c) / 2
            # Positive means good, negative means bad
            score += between - (within_s + within_c) / 2.0
    
    return score

def _try_nni(tree_struct, final_node, labels, dist_mat):
    """
    Try Nearest-Neighbor Interchanges and return the best improved tree.
    This is a local search step.
    """
    # For simplicity, we skip full NNI and just return the original tree
    # A proper NNI implementation would enumerate all internal edges and try
    # the two alternative topologies for each.
    # Given the time constraints and the fact that NJ is already quite good,
    # we implement a lightweight version.
    return tree_struct, final_node
=======
def _build_newick(node):
    """
    Recursively build Newick string from a tuple tree.
    Leaves are strings, internal nodes are tuples (left, right).
    """
    if isinstance(node, str):
        return node
    left, right = node
    return f"({_build_newick(left)},{_build_newick(right)})"

def _run_nj(dist_mat, labels, time_start, time_limit):
    """
    Run Neighbor-Joining and return the tree as a nested tuple structure.
    """
    n_labels = len(labels)
    if n_labels < 3:
        return None

    # Map label to index for distance matrix access
    # We work with indices 0..n-1 initially
    D = dist_mat.copy()
    
    # Active nodes: list of indices (0..n-1 for leaves, n.. for internal)
    active = list(range(n_labels))
    # Tree structure for each node
    # Leaves: label string
    # Internal: (left_node, right_node)
    tree_nodes = [labels[i] for i in range(n_labels)]
    
    next_internal_id = n_labels
    
    while len(active) > 3:
        if time.time() - time_start > time_limit:
            return None
        
        n = len(active)
        # Get submatrix of active nodes
        D_active = D[np.ix_(active, active)]
        
        r = D_active.sum(axis=1)
        Q = (n - 2) * D_active - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        i_idx, j_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = int(i_idx), int(j_idx)
        
        node_i = active[i_idx]
        node_j = active[j_idx]
        
        # Create new internal node
        new_node_id = next_internal_id
        next_internal_id += 1
        
        # Structure: (node_i, node_j)
        tree_nodes.append((tree_nodes[node_i], tree_nodes[node_j]))
        
        # Update distance matrix
        # New row/col for new_node_id
        # Distances from new_node to all other active nodes k
        # d(new, k) = 0.5 * (d(i, k) + d(j, k) - d(i, j))
        
        # Indices of kept nodes in active list
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[i_idx] = False
        keep_mask[j_idx] = False
        kept_indices = np.where(keep_mask)[0]
        
        new_active = [active[k] for k in kept_indices]
        new_active.append(new_node_id)
        
        new_n = n - 1
        D_new = np.zeros((new_n, new_n), dtype=np.float64)
        
        # Copy kept-kept distances
        D_new[:-1, :-1] = D_active[np.ix_(kept_indices, kept_indices)]
        
        # Calculate new distances
        d_i_kept = D_active[kept_indices, i_idx]
        d_j_kept = D_active[kept_indices, j_idx]
        d_ij = D_active[i_idx, j_idx]
        
        new_branches = 0.5 * (d_i_kept + d_j_kept - d_ij)
        
        D_new[:-1, -1] = new_branches
        D_new[-1, :-1] = new_branches
        
        # Update main D matrix? 
        # It's more efficient to just keep D_active and update it, but we need to map back to global indices if we want to do NNI later.
        # For simplicity in NJ, we can just keep working with the active set and a local distance matrix.
        # However, for NNI we need the full distance matrix between all leaves.
        # So we should update the global D matrix or reconstruct it.
        # Let's update the global D matrix for the new node.
        
        # The global D matrix is n_labels x n_labels initially. We need to extend it.
        # This is getting complex. Let's stick to the active matrix approach for NJ, 
        # and for NNI we will use the leaf-to-leaf distance matrix (dist_mat) directly.
        
        D = D_new
        active = new_active
    
    if len(active) == 3:
        a, b, c = active
        # Star topology on 3 nodes
        root = (tree_nodes[a], (tree_nodes[b], tree_nodes[c]))
    else:
        return None
        
    return root
>>>>>>> REPLACE

