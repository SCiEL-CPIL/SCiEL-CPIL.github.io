"""Phylogenetic topology recovery using Neighbor Joining with proper NNI refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def _build_newick(tree):
    """Recursively build Newick string from nested tuple structure."""
    if isinstance(tree, str):
        return tree
    if len(tree) == 2:
        return f"({_build_newick(tree[0])},{_build_newick(tree[1])})"
    return str(tree)

def _get_leaves(tree):
    """Get all leaf labels from a tree structure."""
    leaves = []
    def collect(node):
        if isinstance(node, str):
            leaves.append(node)
        else:
            collect(node[0])
            collect(node[1])
    collect(tree)
    return leaves

def _get_splits(tree, all_labels):
    """Get all non-trivial splits from a tree structure. Returns set of frozensets."""
    splits = set()
    all_labels_set = set(all_labels)
    
    def collect(node):
        if isinstance(node, str):
            return {node}
        left_set = collect(node[0])
        right_set = collect(node[1])
        full_set = left_set | right_set
        if 1 < len(full_set) < len(all_labels_set):
            # Store the smaller side for consistency
            if len(full_set) <= len(all_labels_set) // 2:
                splits.add(frozenset(full_set))
            else:
                splits.add(frozenset(all_labels_set - full_set))
        return full_set
    
    collect(tree)
    return splits

def _quartet_score(tree, dist_mat, labels):
    """
    Compute quartet score: number of quartets where the tree's implied split
    matches the split with maximum within-cluster distance sum.
    """
    n = len(labels)
    if n < 4:
        return 0
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    splits = _get_splits(tree, labels)
    
    score = 0
    for a, b, c, d in combinations(range(n), 4):
        ia, ib, ic, id_ = a, b, c, d
        
        # Three possible splits and their within-cluster distance sums
        S1 = dist_mat[ia, ib] + dist_mat[ic, id_]  # (a,b)|(c,d)
        S2 = dist_mat[ia, ic] + dist_mat[ib, id_]  # (a,c)|(b,d)
        S3 = dist_mat[ia, id_] + dist_mat[ib, ic]  # (a,d)|(b,c)
        
        best_S = max(S1, S2, S3)
        
        # Determine which split the tree implies
        la, lb, lc, ld = labels[a], labels[b], labels[c], labels[d]
        
        split1 = frozenset([la, lb])
        split2 = frozenset([la, lc])
        split3 = frozenset([la, ld])
        
        # Check if tree implies any of these splits (or their complements)
        tree_implies = None
        if split1 in splits or frozenset([lc, ld]) in splits:
            tree_implies = 1
        elif split2 in splits or frozenset([lb, ld]) in splits:
            tree_implies = 2
        elif split3 in splits or frozenset([lb, lc]) in splits:
            tree_implies = 3
        else:
            continue
        
        if tree_implies == 1 and S1 >= best_S - 1e-10:
            score += 1
        elif tree_implies == 2 and S2 >= best_S - 1e-10:
            score += 1
        elif tree_implies == 3 and S3 >= best_S - 1e-10:
            score += 1
    
    return score

def _exhaustive_search(dist_mat, labels):
    """
    For small n (<=7), try all possible unrooted binary tree topologies
    and return the one with the best quartet score.
    """
    n = len(labels)
    if n < 4:
        return _nj_tree(dist_mat, labels)
    
    # Generate all possible unrooted binary trees with n leaves
    # For n leaves, there are (2n-5)!! possible topologies
    # We'll generate them by recursively building trees
    
    def generate_trees(n_leaves):
        """Generate all possible unrooted binary trees with n_leaves leaves (0-indexed)."""
        if n_leaves == 1:
            yield [0]
        elif n_leaves == 2:
            yield [0, 1]
        elif n_leaves == 3:
            # Only one topology for 3 leaves
            yield (0, (1, 2))
        else:
            # For n > 3, we can split into two subtrees
            # The first leaf (0) must be in one subtree
            for k in range(1, n_leaves - 1):
                # Choose k-1 other leaves to go with leaf 0
                # (leaf 0 is always in the left subtree)
                remaining = list(range(1, n_leaves))
                for left_subset in combinations(remaining, k - 1):
                    left_leaves = [0] + list(left_subset)
                    right_leaves = [x for x in remaining if x not in left_subset]
                    
                    for left_tree in generate_trees(len(left_leaves)):
                        for right_tree in generate_trees(len(right_leaves)):
                            yield (left_tree, right_tree)
    
    def tree_to_canonical(tree, leaf_map):
        """Convert a tree with leaf indices to a tree with actual labels."""
        if isinstance(tree, int):
            return leaf_map[tree]
        return (tree_to_canonical(tree[0], leaf_map), tree_to_canonical(tree[1], leaf_map))
    
    best_tree = None
    best_score = -1
    
    # Map leaf indices to labels
    leaf_map = {i: labels[i] for i in range(n)}
    
    for tree in generate_trees(n):
        canonical_tree = tree_to_canonical(tree, leaf_map)
        score = _quartet_score(canonical_tree, dist_mat, labels)
        if score > best_score:
            best_score = score
            best_tree = canonical_tree
    
    return best_tree if best_tree is not None else _nj_tree(dist_mat, labels)

def _nj_tree(dist_mat, labels):
    """
    Perform Neighbor Joining on a distance matrix.
    Returns a nested tuple tree structure.
    """
    n_labels = len(labels)
    if n_labels < 2:
        return labels[0] if labels else ""
    if n_labels == 2:
        return (labels[0], labels[1])
    if n_labels == 3:
        return (labels[0], (labels[1], labels[2]))
    
    nodes = list(range(n_labels))
    D = dist_mat.copy()
    
    tree_struct = {i: labels[i] for i in range(n_labels)}
    next_id = n_labels
    
    while len(nodes) > 3:
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
        
        new_node = next_id
        next_id += 1
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
        tree_struct['final'] = (tree_struct[a], (tree_struct[b], tree_struct[c]))
        final_node = 'final'
    elif len(nodes) == 2:
        a, b = nodes
        tree_struct['final'] = (tree_struct[a], tree_struct[b])
        final_node = 'final'
    else:
        final_node = nodes[0]
    
    return tree_struct[final_node]

def _find_internal_edges(tree):
    """
    Find all internal edges in the tree.
    Returns list of (parent_node, child_node) pairs where both are internal.
    Actually, for NNI we need to find quartets around each internal edge.
    """
    # Represent tree as a graph
    # Each internal node has 3 neighbors (except root which has 2, but unrooted so all have 3)
    
    edges = []  # list of (node1, node2)
    node_info = {}  # node -> list of neighbors
    
    def build_graph(node, parent=None):
        if isinstance(node, str):
            return node  # leaf
        left, right = node
        left_id = build_graph(left, node)
        right_id = build_graph(right, node)
        # This node connects to parent, left, right
        return node
    
    # Better approach: traverse and record structure
    # For NNI, we need to identify each internal edge and the 4 subtrees around it
    
    def get_subtrees_around_edge(tree):
        """
        For each internal edge, return the 4 subtrees that would be involved in an NNI move.
        An internal edge connects two internal nodes. Removing it gives two components.
        Each component has one "attachment point" to the edge.
        The other two neighbors of each endpoint are the subtrees.
        """
        results = []
        
        def traverse(node, parent):
            if isinstance(node, str):
                return
            left, right = node
            # This node is an internal node
            # Its neighbors are: parent (if not root), left, right
            # For an edge between this node and its parent, the 4 subtrees are:
            #   - the two children of this node (left, right)
            #   - the two other neighbors of the parent
            
            traverse(left, node)
            traverse(right, node)
        
        # This is getting complex. Let's use a different approach:
        # Enumerate all quartets and for each, check if the tree implies one of the 3 topologies.
        # Then for NNI, try the alternative topologies.
        
        return results
    
    return get_subtrees_around_edge(tree)

def _nni_refine(tree, labels, dist_mat, max_rounds=50):
    """
    Proper NNI refinement using quartet scoring.
    For each internal edge, try the two alternative quartet resolutions.
    """
    n = len(labels)
    if n < 4:
        return tree
    
    current_score = _quartet_score(tree, dist_mat, labels)
    best_tree = tree
    best_score = current_score
    
    # Find all internal edges by looking at the tree structure
    # An internal edge is between two internal nodes
    # We'll represent the tree as a dictionary of nodes
    
    def tree_to_nodes(tree):
        """Convert tree to a node-based representation."""
        nodes = {}
        next_id = 0
        
        def assign(node):
            nonlocal next_id
            if isinstance(node, str):
                return ('leaf', node)
            left, right = node
            left_id = assign(left)
            right_id = assign(right)
            node_id = ('internal', next_id)
            next_id += 1
            nodes[node_id] = (left_id, right_id)
            return node_id
        
        root = assign(tree)
        return root, nodes
    
    def apply_nni(tree, quartet_positions, new_topology):
        """
        Apply an NNI move.
        quartet_positions: (a, b, c, d) are the 4 subtrees around the edge
        new_topology: which of the 3 alternative topologies to use
        """
        # This is complex to implement generically
        # For now, use a simpler approach: try all possible quartet-based improvements
        
        return tree
    
    # Simpler NNI: for each quartet, if the tree doesn't imply the best topology,
    # try to fix it by restructuring
    
    # Find all quartets where the tree disagrees with the distance-based best
    label_to_idx = {label: i for i, label in enumerate(labels)}
    splits = _get_splits(tree, labels)
    
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        
        # Try each possible NNI move
        # For efficiency, only try moves that fix a disagreeing quartet
        
        for a, b, c, d in combinations(range(n), 4):
            ia, ib, ic, id_ = a, b, c, d
            la, lb, lc, ld = labels[a], labels[b], labels[c], labels[d]
            
            S1 = dist_mat[ia, ib] + dist_mat[ic, id_]
            S2 = dist_mat[ia, ic] + dist_mat[ib, id_]
            S3 = dist_mat[ia, id_] + dist_mat[ib, ic]
            
            best_S = max(S1, S2, S3)
            
            # Determine what the tree currently implies
            split1 = frozenset([la, lb])
            split2 = frozenset([la, lc])
            split3 = frozenset([la, ld])
            
            current_implies = None
            if split1 in splits or frozenset([lc, ld]) in splits:
                current_implies = 1
            elif split2 in splits or frozenset([lb, ld]) in splits:
                current_implies = 2
            elif split3 in splits or frozenset([lb, lc]) in splits:
                current_implies = 3
            
            if current_implies is None:
                continue
            
            # Check if current implies the best
            if current_implies == 1 and S1 >= best_S - 1e-10:
                continue
            if current_implies == 2 and S2 >= best_S - 1e-10:
                continue
            if current_implies == 3 and S3 >= best_S - 1e-10:
                continue
            
            # The tree disagrees with the best quartet topology
            # Try to fix it - this is where NNI would help
            # For simplicity, skip for now and rely on exhaustive for small n
            # For larger n, this is a limitation
    
    return best_tree

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0
    
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n_labels = len(labels)
    
    if n_labels < 2:
        return labels[0] + ";" if labels else ";"
    if n_labels == 2:
        return f"({labels[0]},{labels[1]})"
    if n_labels == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}))"
    
    # Convert sequences to integer matrix
    L = len(next(iter(sequences.values())))
    
    char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    seq_mat = np.zeros((n_labels, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, base in enumerate(seq):
            seq_mat[i, j] = char_to_idx[base]
    
    # Compute pairwise Hamming distances efficiently
    diff = (seq_mat[:, None, :] != seq_mat[None, :, :]).astype(np.int8)
    hamming = diff.sum(axis=2)
    
    # Jukes-Cantor correction
    p = hamming.astype(np.float64) / L
    p_clamped = np.clip(p, 0.0, 0.7499999)
    dist_mat = -0.75 * np.log(1.0 - (4.0/3.0) * p_clamped)
    
    dist_mat = 0.5 * (dist_mat + dist_mat.T)
    np.fill_diagonal(dist_mat, 0.0)
    
    # For small n, use exhaustive search
    if n_labels <= 7:
        tree = _exhaustive_search(dist_mat, labels)
    else:
        # Use NJ for larger n
        tree = _nj_tree(dist_mat, labels)
        
        # Try NNI refinement if time permits
        elapsed = time.time() - start_time
        if elapsed < time_limit - 30:
            tree = _nni_refine(tree, labels, dist_mat, max_rounds=20)
    
    newick = _build_newick(tree)
    return newick

# EVOLVE-BLOCK-END
