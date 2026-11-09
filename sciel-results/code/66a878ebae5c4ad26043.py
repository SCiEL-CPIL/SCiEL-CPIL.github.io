import numpy as np
import math
import time


def _jc_dist_from_counts(counts, L):
    """Compute JC-corrected distance from a 4x4 mismatch count matrix."""
    mismatches = counts.sum() - (counts[0, 0] + counts[1, 1] + counts[2, 2] + counts[3, 3])
    p = mismatches / L
    if p <= 0:
        return 0.0
    if p >= 0.75:
        return 10.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _build_jc_distance_matrix(sequences, labels, L):
    """Build JC-corrected distance matrix using vectorized operations."""
    n = len(labels)
    dist = np.zeros((n, n))
    # Encode sequences as integer arrays
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    enc = np.zeros((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        enc[i] = np.array([base_map[c] for c in seq], dtype=np.uint8)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Count mismatches per position
            diff = enc[i] != enc[j]
            mismatches = diff.sum()
            p = mismatches / L
            if p <= 0:
                d = 0.0
            elif p >= 0.75:
                d = 10.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_build(dist, labels):
    """Build NJ tree, return nested tuple structure."""
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return (labels[0], labels[1])
    
    # Work with node indices
    current_dist = dist.copy()
    current_nodes = [i for i in range(n)]
    node_map = {i: i for i in range(n)}  # current index -> original label index
    
    while len(current_nodes) > 2:
        m = len(current_nodes)
        # Compute r values
        r = current_dist.sum(axis=1)
        
        # Find pair with minimum Q
        best_q = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * current_dist[i, j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        # Create new node
        new_label_idx = len(current_nodes)  # new unique id
        new_node_tuple = (node_map[current_nodes[best_i]], node_map[current_nodes[best_j]])
        
        # Compute new distances
        new_dists = np.zeros(m)
        for k in range(m):
            if k == best_i or k == best_j:
                new_dists[k] = 0
            else:
                new_dists[k] = 0.5 * (current_dist[best_i, k] + current_dist[best_j, k] - current_dist[best_i, best_j])
        
        # Build new node list and distance matrix
        new_nodes = []
        new_map = {}
        count = 0
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            new_nodes.append(current_nodes[k])
            new_map[k] = count
            count += 1
        
        new_m = m - 1
        new_dist = np.zeros((new_m, new_m))
        for old_i in range(m):
            if old_i == best_i or old_i == best_j:
                continue
            new_i = new_map[old_i]
            for old_j in range(m):
                if old_j == best_i or old_j == best_j:
                    continue
                new_j = new_map[old_j]
                new_dist[new_i, new_j] = current_dist[old_i, old_j]
        
        # Add new node
        new_nodes.append(new_label_idx)
        new_map[new_label_idx] = len(new_nodes) - 1
        new_node_idx = len(new_nodes) - 1
        node_map[new_label_idx] = new_node_tuple  # store tuple as "node identity"
        
        for old_k in range(m):
            if old_k == best_i or old_k == best_j:
                continue
            new_k = new_map[old_k]
            new_dist[new_k, new_node_idx] = new_dists[old_k]
            new_dist[new_node_idx, new_k] = new_dists[old_k]
        
        current_nodes = new_nodes
        current_dist = new_dist
    
    return (node_map[current_nodes[0]], node_map[current_nodes[1]])


def _tuple_to_tree(t):
    """Convert nested tuple to tree structure (list-based)."""
    if isinstance(t, int):
        return [t]  # leaf
    else:
        left = _tuple_to_tree(t[0])
        right = _tuple_to_tree(t[1])
        return [None, left, right]


def _get_leaves(tree):
    """Get list of leaf labels in order."""
    if tree[0] is not None:
        return [tree[0]]
    return _get_leaves(tree[1]) + _get_leaves(tree[2])


def _fitch_parsimony(tree, encoded, leaf_order):
    """Compute Fitch parsimony score. tree is nested list, encoded is (L, n) uint8 array.
    Returns (score, node_masks) where node_masks maps internal node id to (L,) mask.
    leaf_order: list of leaf indices in tree order."""
    # Build leaf index mapping: position in leaf_order -> encoded column
    # tree leaves are in the same order as leaf_order
    # We do a post-order traversal
    node_masks = {}
    leaf_pos = 0
    score = 0
    
    def _fitch_postorder(node, node_id):
        nonlocal leaf_pos, score
        if node[0] is not None:
            # Leaf
            idx = leaf_order[leaf_pos]
            node_masks[node_id] = encoded[:, idx]
            leaf_pos += 1
            return
        # Internal
        left_id = node_id * 2
        right_id = node_id * 2 + 1
        _fitch_postorder(node[1], left_id)
        _fitch_postorder(node[2], right_id)
        
        lm = node_masks[left_id]
        rm = node_masks[right_id]
        inter = lm & rm
        union = lm | rm
        # Count positions where intersection is empty
        empty = (inter == 0)
        steps = empty.sum()
        score += steps
        # New mask: where intersection is empty, use union; else intersection
        node_masks[node_id] = np.where(empty, union, inter)
    
    _fitch_postorder(tree, 1)
    return score, node_masks


def _nni_moves(tree):
    """Generate all NNI moves. Returns list of (u, v, w, x, y) tuples.
    For each internal edge (u-v) where v is internal:
    u has children v and w, v has children x and y.
    NNI moves: (swap w and x), (swap w and y), (swap x and y is trivial/identity).
    Actually the three NNI moves are:
    1. (w, x) and (v, y)  ->  (w, y) and (v, x)  [swap x and w across]
    2. (w, x) and (v, y)  ->  (w, x) and (v, y)  [identity]
    3. (w, x) and (v, y)  ->  (w, y) and (v, x)  [same as 1? No...]
    
    Let me reconsider. The three possible quartet topologies for leaves (w, x, y, z) where
    the edge connects subtree A=(x,y) to subtree B=(w,z) via parent u:
    Current: ((x,y),(w,z))
    NNI 1:  ((x,w),(y,z))
    NNI 2:  ((x,z),(y,w))
    
    So for edge u-v where v=(x,y) and u=(v,w):
    Move 1: u=(x, new_v), new_v=(w, y)
    Move 2: u=(y, new_v), new_v=(w, x)
    """
    moves = []
    
    def _collect(node, parent):
        nonlocal moves
        if node[0] is not None:
            return
        # node is internal
        left = node[1]
        right = node[2]
        # Check if left is internal -> edge (node, left)
        if left[0] is None:
            # node is u, left is v
            w = right
            x = left[1]
            y = left[2]
            # Move 1: u=(x, new_v), new_v=(w, y)
            moves.append((node, left, w, x, y, 1))
            # Move 2: u=(y, new_v), new_v=(w, x)
            moves.append((node, left, w, x, y, 2))
        # Check if right is internal -> edge (node, right)
        if right[0] is None:
            # node is u, right is v
            w = left
            x = right[1]
            y = right[2]
            moves.append((node, right, w, x, y, 1))
            moves.append((node, right, w, x, y, 2))
        # Recurse
        _collect(left, node)
        _collect(right, node)
    
    _collect(tree, None)
    return moves


def _apply_nni(tree, u, v, w, x, y, move_type):
    """Apply NNI move in-place on the tree. Returns the modified tree (same object)."""
    # u is parent, v is child, w is u's other child, x and y are v's children
    if move_type == 1:
        # New: u=(x, new_v), new_v=(w, y)
        # v becomes new_v: v=(w, y)
        v[1] = w
        v[2] = y
        # u: replace v with x, and make u=(x, v)
        if u[1] == v:
            u[1] = x
            u[2] = v
        else:
            u[2] = x
            u[1] = v
    else:  # move_type == 2
        # New: u=(y, new_v), new_v=(w, x)
        v[1] = w
        v[2] = x
        if u[1] == v:
            u[1] = y
            u[2] = v
        else:
            u[2] = y
            u[1] = v
    return tree


def _revert_nni(tree, u, v, w, x, y, move_type, orig_u_left, orig_u_right, orig_v_left, orig_v_right):
    """Revert NNI move."""
    u[1] = orig_u_left
    u[2] = orig_u_right
    v[1] = orig_v_left
    v[2] = orig_v_right


def _tree_to_newick(tree, labels):
    """Convert tree to Newick string."""
    if tree[0] is not None:
        return labels[tree[0]]
    left = _tree_to_newick(tree[1], labels)
    right = _tree_to_newick(tree[2], labels)
    return f"({left},{right})"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 260  # Safety margin before 270s kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        return f"({labels[0]},{labels[1]});"
    
    L = len(sequences[labels[0]])
    
    # Build encoded array
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            encoded[i, j] = base_map[c]
    
    # Build JC distance matrix
    dist = _build_jc_distance_matrix(sequences, labels, L)
    
    # Build NJ tree
    tree_tuple = _nj_build(dist, labels)
    tree = _tuple_to_tree(tree_tuple)
    
    # Get leaf order
    leaf_order = _get_leaves(tree)
    
    # Compute initial Fitch score
    current_score, _ = _fitch_parsimony(tree, encoded, leaf_order)
    
    # NNI refinement
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        best_gain = 0
        best_move = None
        
        moves = _nni_moves(tree)
        
        for (u, v, w, x, y, move_type) in moves:
            if time.time() > deadline:
                break
            
            # Save state
            orig_u_left = u[1]
            orig_u_right = u[2]
            orig_v_left = v[1]
            orig_v_right = v[2]
            
            # Apply move
            _apply_nni(tree, u, v, w, x, y, move_type)
            
            # Compute new score
            new_score, _ = _fitch_parsimony(tree, encoded, leaf_order)
            
            # Revert
            _revert_nni(tree, u, v, w, x, y, move_type, orig_u_left, orig_u_right, orig_v_left, orig_v_right)
            
            gain = current_score - new_score
            if gain > best_gain:
                best_gain = gain
                best_move = (u, v, w, x, y, move_type)
        
        if best_move is not None and best_gain > 0:
            u, v, w, x, y, move_type = best_move
            # Apply best move
            _apply_nni(tree, u, v, w, x, y, move_type)
            current_score -= best_gain
            improved = True
        
        if not improved:
            break
    
    newick = _tree_to_newick(tree, labels) + ";"
    return newick
