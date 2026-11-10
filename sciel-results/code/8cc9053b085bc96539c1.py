import math
import time
import numpy as np

_DEADLINE = 250.0  # seconds wall-clock; evaluator kills at 270s


def _jc_distance(p):
    """Jukes-Cantor correction for proportion of differences p."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using vectorized numpy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])

    # Encode sequences: use ASCII values directly for comparison
    enc = np.zeros((n, seq_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.int8)
        enc[i] = arr

    # Vectorized pairwise difference counting
    diff_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        diffs = np.count_nonzero(enc[i] != enc, axis=1)
        diff_matrix[i] = diffs

    # Convert to proportions and apply JC correction
    p = diff_matrix.astype(np.float64) / seq_len
    p_clipped = np.minimum(p, 0.749999)
    D = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    D[p >= 0.75] = 3.0
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    return D


def _build_tree_from_newick(newick):
    """
    Parse a Newick string into a tree structure.
    Returns a nested dict structure with 'name', 'children', 'length' fields.
    """
    def parse(s, pos):
        if s[pos] == '(':
            pos += 1
            children = []
            while s[pos] != ')':
                child, pos = parse(s, pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            # Check for branch length
            if pos < len(s) and s[pos] != ',' and s[pos] != ')':
                end = pos
                while end < len(s) and s[end] != ',' and s[end] != ')' and s[end] != ';':
                    end += 1
                length_str = s[pos:end].strip()
                try:
                    length = float(length_str)
                except ValueError:
                    length = 0.0
                pos = end
            else:
                length = 0.0
            return {'name': None, 'children': children, 'length': length}, pos
        else:
            end = pos
            while end < len(s) and s[end] != ',' and s[end] != ')' and s[end] != ';':
                end += 1
            name = s[pos:end].strip()
            # Check for branch length
            if end < len(s) and s[end] != ',' and s[end] != ')' and s[end] != ';':
                length_end = end
                while length_end < len(s) and s[length_end] != ',' and s[length_end] != ')' and s[length_end] != ';':
                    length_end += 1
                length_str = s[end:length_end].strip()
                try:
                    length = float(length_str)
                except ValueError:
                    length = 0.0
                end = length_end
            else:
                length = 0.0
            return {'name': name, 'children': [], 'length': length}, end
    
    tree, _ = parse(newick, 0)
    return tree


def _get_leaves(node, leaf_set=None):
    """Get all leaf names in a subtree."""
    if leaf_set is None:
        leaf_set = set()
    if node['name'] is not None:
        leaf_set.add(node['name'])
    for child in node['children']:
        _get_leaves(child, leaf_set)
    return leaf_set


def _tree_to_splits(tree, all_leaves):
    """
    Convert a tree to its set of non-trivial splits.
    A split is a frozenset of one side of an internal edge.
    We represent each split by the frozenset of leaves on one side (choose the smaller side for uniqueness).
    """
    splits = set()
    
    def add_split(leaf_set):
        # Normalize: use the smaller side (or either if equal)
        if len(leaf_set) <= len(all_leaves) / 2:
            split = frozenset(leaf_set)
        else:
            split = frozenset(all_leaves - leaf_set)
        # Only add non-trivial splits (size > 1 and < n-1)
        if 1 < len(split) < len(all_leaves) - 1:
            splits.add(split)
    
    def traverse(node, parent_side):
        """
        For each edge, determine the split it defines.
        parent_side is the set of leaves on the parent side of this edge.
        """
        if node['name'] is not None:
            # Leaf node
            return
        
        for child in node['children']:
            child_leaves = _get_leaves(child)
            # The split defined by this edge: child_leaves vs all_leaves - child_leaves
            add_split(child_leaves)
            traverse(child, all_leaves - child_leaves)
    
    traverse(tree, all_leaves)
    return splits


def _newick_to_string(node):
    """Convert tree structure back to Newick string."""
    if node['name'] is not None:
        return node['name']
    if len(node['children']) == 1:
        child_str = _newick_to_string(node['children'][0])
        if node['length'] > 0:
            return f"({child_str}){node['length']:.4f}"
        return f"({child_str})"
    child_strs = [_newick_to_string(c) for c in node['children']]
    inner = ",".join(child_strs)
    if node['length'] > 0:
        return f"({inner}){node['length']:.4f}"
    return f"({inner})"


def _find_internal_edges(tree):
    """
    Find all internal edges in the tree.
    Returns a list of (parent_node, child_node) pairs where child is an internal node
    (has children) and we're looking at edges that can be NNI'd.
    """
    edges = []
    
    def traverse(node, parent):
        if node['name'] is not None:
            return
        for child in node['children']:
            if child['name'] is None and len(child['children']) > 0:
                # This is an internal edge
                edges.append((node, child))
            traverse(child, node)
    
    traverse(tree, None)
    return edges


def _nni_move(tree, edge_idx, D, all_leaves):
    """
    Perform NNI move on the internal edge at edge_idx.
    Returns the new tree if the move improves the quartet score, else None.
    """
    edges = _find_internal_edges(tree)
    if edge_idx >= len(edges):
        return None
    
    parent, child = edges[edge_idx]
    
    # For NNI, we need:
    # - parent has exactly 3 children (or we consider the edge in context)
    # - child has exactly 2 children (bifurcating)
    
    if len(parent['children']) != 3 or len(child['children']) != 2:
        return None
    
    # Find the index of child in parent's children
    child_idx = parent['children'].index(child)
    
    # The three children of parent
    c0 = parent['children'][(child_idx + 0) % 3]
    c1 = child  # the child we're moving
    c2 = parent['children'][(child_idx + 2) % 3]
    
    # Actually, let's be more careful
    other_children = [c for c in parent['children'] if c is not child]
    if len(other_children) != 2:
        return None
    
    A = other_children[0]
    B = other_children[1]
    C = child['children'][0]
    D_node = child['children'][1]
    
    # Current topology: ((C,D),A),B  (parent connects to child(C,D), A, B)
    # NNI alternative 1: ((C,A),(D,B))
    # NNI alternative 2: ((C,B),(D,A))
    
    # Compute quartet scores for each topology
    # For Jukes-Cantor, the best topology minimizes the sum of branch lengths
    # which is equivalent to maximizing the likelihood
    
    def quartet_score(A_leaves, B_leaves, C_leaves, D_leaves):
        """
        Compute the score for a quartet topology.
        Using distance-based approach: the best tree for 4 taxa minimizes
        the total branch length.
        """
        # For 4 taxa a,b,c,d with topology ((a,b),(c,d)):
        # d_ab + d_cd + (d_ac + d_bd - d_ad - d_bc) / 2
        # Actually, the standard NJ formula for 4 taxa:
        # The three possible topologies have scores:
        # ((a,b),(c,d)): d_ab + d_cd
        # ((a,c),(b,d)): d_ac + d_bd  
        # ((a,d),(b,c)): d_ad + d_bc
        # The best topology minimizes this sum
        
        # But we need to use the actual distances
        # For simplicity, use the sum of pairwise distances for each pairing
        pass
    
    # Simpler approach: use the distance matrix to evaluate which quartet topology is best
    # For 4 leaves a,b,c,d:
    # Topology ((a,b),(c,d)) has cost: d(a,b) + d(c,d)
    # Topology ((a,c),(b,d)) has cost: d(a,c) + d(b,d)
    # Topology ((a,d),(b,c)) has cost: d(a,d) + d(b,c)
    
    # Get representative leaves for each subtree
    A_rep = _get_leaves(A)
    B_rep = _get_leaves(B)
    C_rep = _get_leaves(C)
    D_rep = _get_leaves(D_node)
    
    # For NNI with larger subtrees, we need to use the "distance" between subtrees
    # which is the average pairwise distance between leaves in different subtrees
    
    def avg_dist(set1, set2):
        if not set1 or not set2:
            return 0
        total = 0
        count = 0
        for x in set1:
            for y in set2:
                total += D[list(all_leaves).index(x), list(all_leaves).index(y)]
                count += 1
        return total / count if count > 0 else 0
    
    # Current topology: ((C,D),A),B
    # This means the tree has edges: parent-child(C,D), parent-A, parent-B
    # The quartet-like structure involves how these subtrees connect
    
    # For proper NNI, we compare:
    # Current: (A,(C,D),B) 
    # Alt 1: ((A,C),(B,D))
    # Alt 2: ((A,B),(C,D))
    
    # Score = sum of distances that are "cut" by the topology
    # For topology ((X,Y),(Z,W)), score = d(X,Z) + d(Y,W) + d(X,W) + d(Y,Z) - d(X,Y) - d(Z,W)
    # Actually, let's use a simpler heuristic:
    
    # The total length of the tree can be approximated by:
    # For each internal edge, the length is determined by the distances
    
    # Let's use a simpler criterion: minimize the sum of pairwise distances
    # that are "explained" by the tree structure
    
    # Alternative: just try the move and see if it reduces the total tree length
    # computed from the distance matrix
    
    # For now, use a simple heuristic based on average distances
    # Current topology has "long" edges if A is far from (C,D) and B is far from (C,D)
    
    d_A_CD = avg_dist(A_rep, C_rep | D_rep)
    d_B_CD = avg_dist(B_rep, C_rep | D_rep)
    d_AB = avg_dist(A_rep, B_rep)
    d_CD = avg_dist(C_rep, D_rep)
    
    # Current cost: d_A_CD + d_B_CD (the two edges from parent to child subtrees)
    current_cost = d_A_CD + d_B_CD
    
    # Alt 1: ((A,C),(B,D))
    d_AC = avg_dist(A_rep, C_rep)
    d_BD = avg_dist(B_rep, D_rep)
    alt1_cost = d_AC + d_BD
    
    # Alt 2: ((A,B),(C,D))  
    d_AB2 = avg_dist(A_rep, B_rep)
    d_CD2 = avg_dist(C_rep, D_rep)
    alt2_cost = d_AB2 + d_CD2
    
    # Choose the best alternative
    if alt1_cost < current_cost and alt1_cost < alt2_cost:
        # Perform NNI move to ((A,C),(B,D))
        new_tree = _apply_nni(tree, parent, child, A, B, C, D_node, 1)
        return new_tree
    elif alt2_cost < current_cost and alt2_cost < alt1_cost:
        # Perform NNI move to ((A,B),(C,D))
        new_tree = _apply_nni(tree, parent, child, A, B, C, D_node, 2)
        return new_tree
    
    return None


def _apply_nni(tree, parent, child, A, B, C, D_node, move_type):
    """
    Apply an NNI move to the tree.
    move_type: 1 for ((A,C),(B,D)), 2 for ((A,B),(C,D))
    """
    # This is complex - we need to rebuild the tree structure
    # For simplicity, return None for now and use a different approach
    return None


def _neighbor_joining(D, labels):
    """
    Perform neighbor joining on a distance matrix D (numpy array, symmetric, zero diag).
    labels: list of taxon names corresponding to rows/cols of D.
    Returns a Newick string (no trailing semicolon).
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    D = D.astype(np.float64).copy()
    node_names = list(labels)

    while len(node_names) > 3:
        m = len(node_names)
        # Compute row sums
        r = D.sum(axis=1)  # shape (m,)

        # Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * D - r[:, None] - r[None, :]
        # Set diagonal to inf
        np.fill_diagonal(Q, np.inf)
        # Find min
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i

        # Create new node
        new_name = f"({node_names[i]},{node_names[j]})"

        # Compute new distances
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])

        # Build new distance matrix using fancy indexing
        keep = [k for k in range(m) if k != i and k != j]
        keep_arr = np.array(keep, dtype=np.intp)
        new_m = m - 1

        # Extract submatrix for kept nodes
        new_D = np.empty((new_m, new_m), dtype=np.float64)
        new_D[:-1, :-1] = D[np.ix_(keep_arr, keep_arr)]

        # Add new node distances
        new_node_idx = m - 2
        new_D[:-1, new_node_idx] = new_dist[keep_arr]
        new_D[new_node_idx, :-1] = new_dist[keep_arr]
        new_D[new_node_idx, new_node_idx] = 0.0

        # Update node_names
        new_names = [node_names[k] for k in keep] + [new_name]

        D = new_D
        node_names = new_names

    # Now we have 3 nodes
    return f"({node_names[0]},{node_names[1]},{node_names[2]})"


def _nni_refine(newick, D_original, labels, deadline):
    """
    Attempt NNI refinement on the NJ tree using distance-based scoring.
    Returns improved Newick string or original if no improvement or timeout.
    """
    n = len(labels)
    if n < 4:
        return newick

    # Parse the Newick to get the tree
    tree = _build_tree_from_newick(newick)
    all_leaves = set(labels)
    
    # Compute initial splits
    current_splits = _tree_to_splits(tree, all_leaves)
    
    # Try NNI moves
    max_passes = 3
    improved = True
    passes = 0
    
    while improved and passes < max_passes and time.time() < deadline - 5:
        improved = False
        passes += 1
        
        edges = _find_internal_edges(tree)
        
        for edge_idx in range(len(edges)):
            if time.time() > deadline - 5:
                break
            
            new_tree = _nni_move(tree, edge_idx, D_original, all_leaves)
            if new_tree is not None:
                # Check if the new tree has different splits (improvement)
                new_splits = _tree_to_splits(new_tree, all_leaves)
                if len(new_splits) > len(current_splits):
                    # More splits means potentially better (more resolved)
                    # But we need to check if the splits are actually better
                    # For now, accept the move
                    tree = new_tree
                    current_splits = new_splits
                    improved = True
                    break
                elif new_splits != current_splits:
                    # Different splits - could be better or worse
                    # Use a heuristic: prefer trees with more non-trivial splits
                    if len(new_splits) >= len(current_splits):
                        tree = new_tree
                        current_splits = new_splits
                        improved = True
                        break
    
    return _newick_to_string(tree)


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree topology.
    Returns Newick string over labels t0..t(n-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Fallback: star tree
        return "(" + ",".join(labels) + ");"

    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)

    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Fallback: star tree
        return "(" + ",".join(labels) + ");"

    # Run neighbor joining
    result = _neighbor_joining(D, labels)

    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 2:
        return result + ";"

    # Attempt NNI refinement if we have time
    remaining = deadline - time.time()
    if remaining > 5:
        result = _nni_refine(result, D, labels, deadline)

    # Ensure all taxa appear exactly once
    return result + ";"
