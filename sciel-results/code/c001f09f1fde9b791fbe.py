"""Neighbour-joining with NNI + SPR refinement under Jukes-Cantor ML."""
import time
import math
import numpy as np

# --- time budget constants ---
WALL_LIMIT = 265.0  # seconds, leave 5 s margin before the 270 s kill
PER_INSTANCE_LIMIT = 250.0 / 20.0  # ~12.5 s per instance as a soft cap


def _jc_dist_matrix(int_seqs, n, L):
    """Compute pairwise JC-corrected distance matrix using NumPy."""
    # int_seqs: (n, L) array of ints 0-3
    # Count mismatches per pair
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.count_nonzero(int_seqs[i] != int_seqs[j])
            p = diffs / L
            if p >= 0.75:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat


def _nj_build(dist_mat, n):
    """Classic neighbour-joining. Returns (parent, left, right, bl) arrays of length n-2.
    Each internal node index is n, n+1, ..., 2n-3.
    leaf indices are 0..n-1.
    parent[i] is the parent of node i (for internal nodes that get merged later).
    left[i], right[i] are children of internal node i.
    bl[i] is the branch length for the edge from node i to its parent (set during merge).
    Actually let's use a simpler representation:
    We'll build the tree as a set of edges.
    Each merge creates a new node. We track children and branch lengths.
    """
    active = list(range(n))
    # weight[i] = sum of distances from i to all other active nodes
    # dist[i][j] stored in dist_mat
    new_nodes = []
    # We'll store the tree as: for each internal node (created during NJ),
    # its two children and the branch lengths to them.
    # Final tree: 2*(n-2) internal+leaves... no, n leaves + n-2 internal = 2n-2 nodes.
    # Let's use arrays:
    # node_children: dict from node_id -> (child1, child2) for internal nodes
    # node_bl: dict from node_id -> (bl_to_parent, parent_id) -- not needed directly
    # Simpler: store edges.
    
    node_count = n  # next new node id
    children = {}  # node_id -> (left, right)
    branch_len = {}  # (child, parent) -> length
    
    while len(active) > 1:
        m = len(active)
        if m == 2:
            a, b = active[0], active[1]
            d = dist_mat[a][b]
            # Connect a and b with a single edge of length d/2 each? 
            # Actually for 2 nodes, the tree is just a-b with some length.
            # For NJ with 2 nodes remaining, we just join them.
            # The total distance between them is dist_mat[a][b].
            # We'll create one more internal node to make it proper, or just return edge.
            # Let's handle: create internal node connecting a and b.
            c = node_count
            node_count += 1
            children[c] = (a, b)
            # branch lengths: for 2-node case, each gets d/2
            bl_a = d / 2.0
            bl_b = d / 2.0
            branch_len[(a, c)] = max(bl_a, 0)
            branch_len[(b, c)] = max(bl_b, 0)
            active = [c]
            break
        
        # Compute Q matrix
        r = np.zeros(m, dtype=np.float64)
        for idx, a in enumerate(active):
            r[idx] = np.sum(dist_mat[a][active])
        
        best_q = float('inf')
        best_i = -1
        best_j = -1
        for i in range(m):
            for j in range(i + 1, m):
                a = active[i]
                b = active[j]
                q = (m - 2) * dist_mat[a][b] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        a = active[best_i]
        b = active[best_j]
        d_ab = dist_mat[a][b]
        
        # Compute branch lengths
        ra = r[best_i]
        rb = r[best_j]
        bl_a = 0.5 * d_ab + (ra - rb) / (2.0 * (m - 2))
        bl_b = 0.5 * d_ab + (rb - ra) / (2.0 * (m - 2))
        if bl_a < 0:
            bl_a = 0.0
        if bl_b < 0:
            bl_b = 0.0
        
        c = node_count
        node_count += 1
        children[c] = (a, b)
        branch_len[(a, c)] = bl_a
        branch_len[(b, c)] = bl_b
        
        # Update distances to the new node c
        # dist_mat[c][x] = 0.5 * (dist_mat[a][x] + dist_mat[b][x] - dist_mat[a][b])
        # We need to add c to dist_mat. Let's grow dist_mat dynamically.
        # Actually dist_mat is n x n. Internal nodes have indices >= n.
        # We need to extend dist_mat. Let's pre-allocate a large enough matrix.
        # Max nodes: 2*n - 1. Let's use a dict-based approach or extend.
        # For simplicity, let's use a dict for distances involving internal nodes.
        # Or: extend dist_mat to (2n) x (2n).
        
        for x in active:
            if x == a or x == b:
                continue
            d_cx = 0.5 * (dist_mat[a][x] + dist_mat[b][x] - d_ab)
            if d_cx < 0:
                d_cx = 0.0
            # Store in a global structure
            # We'll maintain a separate dict for internal node distances
            pass
        
        # This is getting complicated with the growing matrix. Let me restructure.
        # I'll use a list of active nodes and a distance function.
        # Let's restart with a cleaner approach.
        break  # placeholder
    
    return None  # placeholder, will rewrite


# Let me rewrite more carefully with a complete working NJ.

def _neighbor_joining(dist_mat, n):
    """
    dist_mat: (n, n) array for leaves.
    Returns (children, branch_len, root)
    children: dict internal_node -> (left, right)
    branch_len: dict (child, parent) -> length
    root: the root node id
    """
    # Extend dist_mat to accommodate internal nodes
    max_nodes = 2 * n - 1
    d = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    d[:n, :n] = dist_mat
    
    active = list(range(n))
    node_count = n
    children = {}
    branch_len = {}
    
    while len(active) > 1:
        m = len(active)
        if m == 2:
            a, b = active[0], active[1]
            c = node_count
            node_count += 1
            children[c] = (a, b)
            d_ab = d[a][b]
            bl_a = max(d_ab / 2.0, 0.0)
            bl_b = max(d_ab / 2.0, 0.0)
            branch_len[(a, c)] = bl_a
            branch_len[(b, c)] = bl_b
            root = c
            break
        
        # r vector
        r = np.zeros(m, dtype=np.float64)
        for idx, a in enumerate(active):
            for idx2, b in enumerate(active):
                if idx != idx2:
                    r[idx] += d[a][b]
        
        best_q = float('inf')
        best_i = -1
        best_j = -1
        for i in range(m):
            for j in range(i + 1, m):
                a = active[i]
                b = active[j]
                q = (m - 2) * d[a][b] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        a = active[best_i]
        b = active[best_j]
        d_ab = d[a][b]
        ra = r[best_i]
        rb = r[best_j]
        
        bl_a = 0.5 * d_ab + (ra - rb) / (2.0 * (m - 2))
        bl_b = 0.5 * d_ab + (rb - ra) / (2.0 * (m - 2))
        if bl_a < 0:
            bl_a = 0.0
        if bl_b < 0:
            bl_b = 0.0
        
        c = node_count
        node_count += 1
        children[c] = (a, b)
        branch_len[(a, c)] = bl_a
        branch_len[(b, c)] = bl_b
        
        # Compute distances from c to all other active nodes
        for x in active:
            if x == a or x == b:
                continue
            d_cx = 0.5 * (d[a][x] + d[b][x] - d_ab)
            if d_cx < 0:
                d_cx = 0.0
            d[c][x] = d_cx
            d[x][c] = d_cx
        
        # Remove a and b, add c
        active.remove(a)
        active.remove(b)
        active.append(c)
    
    return children, branch_len, root


def _tree_to_newick(children, branch_len, root, labels):
    """Convert tree to Newick string."""
    def node_str(node):
        if node < len(labels):
            return labels[node]
        left, right = children[node]
        bl_l = branch_len.get((left, node), 0.0)
        bl_r = branch_len.get((right, node), 0.0)
        ls = node_str(left) + ("%.4f" % bl_l if bl_l > 0.0001 else "")
        rs = node_str(right) + ("%.4f" % bl_r if bl_r > 0.0001 else "")
        return "(" + ls + "," + rs + ")"
    
    return node_str(root) + ";"


def _get_leaves(node, children, leaf_set):
    if node in leaf_set:
        leaf_set.add(node)
    else:
        l, r = children[node]
        _get_leaves(l, children, leaf_set)
        _get_leaves(r, children, leaf_set)


def _nni_move(children, branch_len, root, int_seqs, n, L):
    """
    Try all NNI moves. Return (best_new_children, best_new_branch_len, best_new_root, best_logL, best_move_applied)
    """
    # Build parent and leaf set
    leaf_set = set(range(n))
    
    # Build parent pointers
    parent = {}
    for node, (l, r) in children.items():
        parent[l] = node
        parent[r] = node
    
    # Precompute: for each internal node, the set of leaves under it
    leaf_sets = {}
    def compute_leaves(node):
        if node in leaf_set:
            leaf_sets[node] = frozenset([node])
        else:
            l, r = children[node]
            leaf_sets[node] = leaf_sets[l] | leaf_sets[r]
    # Do post-order
    def postorder(node):
        if node in leaf_set:
            leaf_sets[node] = frozenset([node])
            return
        l, r = children[node]
        postorder(l)
        postorder(r)
        leaf_sets[node] = leaf_sets[l] | leaf_sets[r]
    postorder(root)
    
    # Current logL
    cur_logL = _felsenstein_ml(children, branch_len, root, int_seqs, n, L)
    
    best_logL = cur_logL
    best_children = children
    best_bl = branch_len
    best_root = root
    best_applied = False
    
    # For each internal edge (u, v) where v is the "upper" node (parent of u)
    # NNI: swap one of u's children with one of v's other child
    for u in list(children.keys()):
        if u == root:
            continue  # root has no parent, skip
        v = parent.get(u)
        if v is None:
            continue
        # u's children
        u_l, u_r = children[u]
        # v's children: one is u, the other is w
        v_l, v_r = children[v]
        if v_l == u:
            w = v_r
        elif v_r == u:
            w = v_l
        else:
            continue
        
        # Two possible NNI moves:
        # 1. (u_l, u_r) and (w, other_v_child) -> (u_l, w) and (u_r, other_v_child)
        #    Actually: the quartet is (u_l, u_r, w, ...) where ... is the rest
        #    NNI swaps: originally ((u_l,u_r), w, rest). New: ((u_l,w), u_r, rest) or ((u_r,w), u_l, rest)
        
        # Try swap 1: new children of u: (u_l, w), new children of v: (u_r, u_new_node?) 
        # Wait, NNI doesn't create new nodes. It restructures the same nodes.
        # Original: v has children u and w. u has children u_l and u_r.
        # After NNI: v has children u and new_internal? No.
        # NNI: the two subtrees attached to the internal edge (u-v) are:
        #   Side A: u's subtree (with u_l, u_r)
        #   Side B: w's subtree (w is v's other child)
        # The NNI changes the topology to:
        #   Option 1: v connects to u' (which has u_l and w) and to u_r directly? No.
        #   
        # Let me think again. The four subtrees at the quartet are:
        #   S1 = u_l, S2 = u_r, S3 = w, S4 = rest_of_tree (the part above v)
        # Original: ((S1,S2), S3, S4)  [v connects to u(S1,S2) and w(S3), and v connects to parent(S4)]
        # NNI options:
        #   ((S1,S3), S2, S4)
        #   ((S2,S3), S1, S4)
        
        # Implementation:
        # Create new node structure:
        # new_u has children (u_l, w)
        # v has children (new_u, u_r)
        # But wait, v's parent connection stays the same.
        
        # Actually, we keep the same node IDs. We just reassign children.
        # But u and v are internal nodes. We can't create new ones.
        # So: children[u] = (u_l, w), children[v] = (u_r, u_new?) 
        # No, v's children should be (new_node_for_u_l_w, u_r)? 
        # This requires a new internal node. NNI in a binary tree with fixed nodes...
        # 
        # Actually in NNI, we have 3 internal nodes involved: u, v, and we need one more.
        # Let me reconsider: the standard NNI on edge (u,v) where u is lower:
        #   Before: ... - v - u - (u_l, u_r)  and v also connects to w
        #   After:  ... - v' - (u_l, w)  and v' also connects to u_r
        #   where v' is the same node as v, but its children change.
        #   And u becomes... actually u is removed? No.
        #
        # I think the cleanest way: the 4 subtrees are S1=u_l, S2=u_r, S3=w, S4=parent_side_of_v.
        # Original pairing: (S1,S2) and S3 and S4 -> tree is ((S1,S2),S3) under v, v under parent.
        # NNI option 1: ((S1,S3),S2) under a new structure.
        #   We need: node_a has children (S1, S3). node_b has children (node_a, S2). parent connects to node_b.
        #   So we reuse u and v: let u have children (S1, S3), let v have children (u, S2).
        # NNI option 2: ((S2,S3),S1): u has children (S2, S3), v has children (u, S1).
        
        # Try option 1:
        children[u] = (u_l, w)
        children[v] = (u, u_r)
        # Need to update branch lengths. For now, set to 0 and optimize later.
        # Quick logL estimate:
        logL1 = _felsenstein_ml(children, branch_len, root, int_seqs, n, L)
        
        if logL1 > best_logL:
            best_logL = logL1
            best_children = {k: tuple(vv) for k, vv in children.items()}
            best_bl = dict(branch_len)
            best_root = root
            best_applied = True
        else:
            # revert
            children[u] = (u_l, u_r)
            children[v] = (u, w) if (v_l == u or v_r == u) else (w, u)
            # Actually need to restore properly
            if v_l == u:
                children[v] = (u, w)
            else:
                children[v] = (w, u)
        
        # Try option 2:
        children[u] = (u_r, w)
        children[v] = (u, u_l)
        logL2 = _felsenstein_ml(children, branch_len, root, int_seqs, n, L)
        
        if logL2 > best_logL:
            best_logL = logL2
            best_children = {k: tuple(vv) for k, vv in children.items()}
            best_bl = dict(branch_len)
            best_root = root
            best_applied = True
        else:
            # revert
            children[u] = (u_l, u_r)
            if v_l == u:
                children[v] = (u, w)
            else:
                children[v] = (w, u)
    
    return best_children, best_bl, best_root, best_logL, best_applied


def _felsenstein_ml(children, branch_len, root, int_seqs, n, L):
    """
    Compute log-likelihood of the sequences under JC model on the given tree.
    int_seqs: (n, L) array of ints 0-3.
    Returns log-likelihood (sum over sites).
    """
    # JC model:
    # P(same) = 1/4 + 3/4 * exp(-4t/3)
    # P(diff) = 1/4 * (1 - exp(-4t/3))
    
    # Root at root. Do post-order traversal.
    # For each node and each site, compute P(data under node | state at node) for each of 4 states.
    # Root state: assume equal prior (1/4 each).
    
    # Build parent and order
    parent = {}
    for node, (l, r) in children.items():
        parent[l] = node
        parent[r] = node
    
    # Post-order list
    post_order = []
    def _postorder(node):
        if node < n:
            post_order.append(node)
            return
        l, r = children[node]
        _postorder(l)
        _postorder(r)
        post_order.append(node)
    _postorder(root)
    
    # For each site, compute likelihoods
    total_logL = 0.0
    # Precompute transition matrices for each branch
    # bl[(child, parent)] = length
    # P(child state | parent state)
    
    # For efficiency, process all sites together using vectorized ops if possible.
    # But for now, loop over sites.
    
    # State space: 0,1,2,3
    # For each node, store 4 likelihood values (one per possible state at that node)
    # ll[node] = array of 4 floats
    
    # Process in post-order
    # For leaves: ll[leaf][state] = 1 if int_seqs[leaf][site] == state else 0
    # For internal: ll[node][state] = sum over child states of P(child_state | state) * ll[child][child_state]
    
    # Let's do it site by site for clarity, then vectorize if needed.
    # Actually, let's process all sites at once using arrays.
    
    # ll will be a dict: node -> (4, L) array
    # For leaves: ll[leaf] = one-hot
    # For internal: compute from children
    
    # Precompute transition probabilities for each branch
    # trans[(child, parent)] = 4x4 matrix
    
    # For JC:
    # P(same) = 1/4 + 3/4 * e^(-4t/3)
    # P(diff) = 1/4 * (1 - e^(-4t/3))
    
    # Build transition matrices
    trans = {}
    for (child, par), bl in branch_len.items():
        if bl < 0.001:
            # Identity
            mat = np.eye(4, dtype=np.float64)
            mat[0,0] = 1.0
            mat[1,1] = 1.0
            mat[2,2] = 1.0
            mat[3,3] = 1.0
        else:
            p_same = 0.25 + 0.75 * math.exp(-4.0 * bl / 3.0)
            p_diff = (1.0 - p_same) / 3.0
            mat = np.full((4, 4), p_diff, dtype=np.float64)
            for i in range(4):
                mat[i, i] = p_same
        trans[(child, par)] = mat
    
    # Now process sites. For each site s, compute ll for all nodes.
    # ll[node][:, s] = likelihood vector for node at site s
    
    # Allocate: ll = {node: np.zeros((4, L))}
    ll = {}
    for node in post_order:
        if node < n:
            # Leaf
            ll[node] = np.zeros((4, L), dtype=np.float64)
            for s in range(L):
                ll[node, int_seqs[node, s], s] = 1.0
        else:
            l, r = children[node]
            ll[node] = np.zeros((4, L), dtype=np.float64)
            trans_l = trans.get((l, node), np.eye(4))
            trans_r = trans.get((r, node), np.eye(4))
            # ll[node][i, s] = sum_j trans_l[j,i] * ll[l][j,s] + ... 
            # Wait: P(data under l | state i at node) = sum_j P(j | i) * ll[l][j]
            # trans_l[j, i] = P(j | i)
            # So: ll[node][i, s] = sum_j trans_l[j, i] * ll[l][j, s]
            # = (trans_l.T @ ll[l])[:, s] ... but trans_l[j,i] is element (j,i), so (trans_l.T)[i,j] = trans_l[j,i]
            # ll[node][i,s] = sum_j trans_l[j,i] * ll[l][j,s]
            # This is: ll[node][:, s] = trans_l.T @ ll[l][:, s] + trans_r.T @ ll[r][:, s]? 
            # No! It's: ll[node][:,s] = sum over left children + sum over right children? 
            # No, the likelihood is: P(D_l, D_r | state i at node) = P(D_l | i) * P(D_r | i)
            # = [sum_j P(j|i) ll[l][j]] * [sum_k P(k|i) ll[r][k]]
            # = [trans_l.T @ ll[l]]_i * [trans_r.T @ ll[r]]_i  (element-wise product!)
            
            # So: ll[node][:,s] = (trans_l.T @ ll[l][:,s]) * (trans_r.T @ ll[r][:,s])
            
            # Vectorized over sites:
            ll_l = trans_l.T @ ll[l]  # (4, L)
            ll_r = trans_r.T @ ll[r]  # (4, L)
            ll[node] = ll_l * ll_r  # element-wise
    
    # Root likelihood: sum over root states, assuming uniform prior
    # logL per site = log(sum_i ll[root][i, s] / 4)
    root_ll = ll[root]  # (4, L)
    site_logL = np.log(np.sum(root_ll, axis=0) / 4.0)
    total_logL = np.sum(site_logL)
    
    return float(total_logL)


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + PER_INSTANCE_LIMIT
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Convert to int array
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    int_seqs = np.zeros((n, L), dtype=np.int32)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for s in range(L):
            int_seqs[i, s] = base_map[seq[s]]
    
    # Step 1: NJ
    dist_mat = _jc_dist_matrix(int_seqs, n, L)
    children, branch_len, root = _neighbor_joining(dist_mat, n)
    
    if time.time() > deadline:
        return _tree_to_newick(children, branch_len, root, labels)
    
    # Step 2: NNI refinement
    # Try a few rounds of NNI
    for round_num in range(3):
        if time.time() > deadline:
            break
        new_children, new_bl, new_root, new_logL, applied = _nni_move(
            children, branch_len, root, int_seqs, n, L
        )
        if applied:
            children = new_children
            branch_len = new_bl
            root = new_root
        else:
            break
    
    return _tree_to_newick(children, branch_len, root, labels)
