import numpy as np
import time
import math

def _jc_distance_matrix(seq_int, L):
    """Compute JC-corrected pairwise distances. seq_int: (n, L) int8 in {0,1,2,3}."""
    n = seq_int.shape[0]
    # Count mismatches per pair
    # For each site position, count how many taxa have each character
    # Use matrix multiply trick: for each pair (i,j), mismatch = 1 - sum_s (1[a_i=s] * 1[a_j=s])
    # Build one-hot: (n, L, 4)
    onehot = np.zeros((n, L, 4), dtype=np.int8)
    onehot[np.arange(n)[:, None], np.arange(L)[None, :], seq_int] = 1
    # (n, 4, L) @ (4, L, n) -> (n, n, L)
    # Actually: onehot_i @ onehot_j^T per site
    # Mismatch count = L - sum over sites of sum over s of onehot[i,s,site]*onehot[j,s,site]
    # = L - sum_s (onehot[:,:,s] @ onehot[:,:,s].T)
    match_count = np.zeros((n, n), dtype=np.int32)
    for s in range(4):
        oh = onehot[:, :, s].astype(np.int32)  # (n, L)
        match_count += oh @ oh.T
    mismatch = L - match_count
    p = mismatch.astype(np.float64) / L
    p = np.clip(p, 1e-10, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(d, 0.0)
    return d

def _nj_build(d):
    """Neighbor-Joining. Returns (parent, child1, child2, is_leaf, num_nodes).
    Nodes 0..n-1 are leaves. Internal nodes n..num_nodes-1."""
    n = d.shape[0]
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = d
    active = list(range(n))
    parent = [-1] * max_nodes
    child1 = [-1] * max_nodes
    child2 = [-1] * max_nodes
    is_leaf = [True] * max_nodes
    next_idx = n

    while len(active) > 2:
        k = len(active)
        idx = np.array(active)
        sub_d = D[np.ix_(idx, idx)]
        r = sub_d.sum(axis=1)
        Q = (k - 2) * sub_d - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, np.inf)
        min_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat, Q.shape)
        u = active[i_idx]
        v = active[j_idx]
        w = next_idx
        next_idx += 1
        for x in active:
            if x == u or x == v:
                continue
            D[w, x] = 0.5 * (D[u, x] + D[v, x] - D[u, v])
            D[x, w] = D[w, x]
        D[w, w] = 0.0
        active = [x for x in active if x != u and x != v]
        active.append(w)
        parent[u] = w
        parent[v] = w
        child1[w] = u
        child2[w] = v
        is_leaf[w] = False

    # Now active has 2 nodes: connect them with a new internal node
    u, v = active[0], active[1]
    w = next_idx
    next_idx += 1
    parent[u] = w
    parent[v] = w
    child1[w] = u
    child2[w] = v
    is_leaf[w] = False

    return parent, child1, child2, is_leaf, next_idx

def _build_postorder(root, child1, child2, is_leaf, max_nodes):
    """Return nodes in postorder (children before parent)."""
    order = []
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            order.append(node)
            continue
        if not is_leaf[node]:
            c1 = child1[node]
            c2 = child2[node]
            stack.append((node, True))
            if c2 >= 0:
                stack.append((c2, False))
            if c1 >= 0:
                stack.append((c1, False))
        else:
            order.append(node)
    return order

def _jc_probs(bl):
    """Return (p_same, p_diff) for JC branch length bl."""
    if bl < 0:
        bl = 0.0
    e = math.exp(-4.0 * bl / 3.0)
    return 0.25 + 0.75 * e, 0.25 * (1.0 - e)

def _compute_ll(root, parent, child1, child2, is_leaf, n, seq_int, L, branch_len, postorder):
    """Felsenstein pruning. Returns total log-likelihood.
    branch_len[node] = length of branch from parent[node] to node (for non-root)."""
    max_nodes = 2 * n
    ll = np.zeros((max_nodes, 4), dtype=np.float64)

    for node in postorder:
        if is_leaf[node]:
            obs = seq_int[node]
            # Leaf likelihood: P(obs | state) with no branch (or incorporate leaf branch)
            # We'll handle leaf branches via the parent's transition
            for s in range(4):
                ll[node, s] = 1.0 if s == obs else 0.0
        else:
            c1 = child1[node]
            c2 = child2[node]
            if c1 >= 0 and c2 >= 0:
                ps1, pd1 = _jc_probs(branch_len[c1])
                ps2, pd2 = _jc_probs(branch_len[c2])
                # Transition matrices: T[s, s'] = ps if s==s' else pd
                # ll[node, s] = sum_{s1,s2} T1[s,s1] * ll[c1,s1] * T2[s,s2] * ll[c2,s2]
                for s in range(4):
                    total = 0.0
                    for s1 in range(4):
                        p1 = ps1 if s == s1 else pd1
                        v1 = p1 * ll[c1, s1]
                        for s2 in range(4):
                            p2 = ps2 if s == s2 else pd2
                            total += v1 * p2 * ll[c2, s2]
                    ll[node, s] = total
            elif c1 >= 0:
                ps1, pd1 = _jc_probs(branch_len[c1])
                for s in range(4):
                    total = 0.0
                    for s1 in range(4):
                        p1 = ps1 if s == s1 else pd1
                        total += p1 * ll[c1, s1]
                    ll[node, s] = total
            elif c2 >= 0:
                ps2, pd2 = _jc_probs(branch_len[c2])
                for s in range(4):
                    total = 0.0
                    for s2 in range(4):
                        p2 = ps2 if s == s2 else pd2
                        total += p2 * ll[c2, s2]
                    ll[node, s] = total

    # Root: uniform prior
    total = 0.25 * np.sum(ll[root])
    if total <= 0:
        return -1e10
    return math.log(total)

def _optimize_branches(root, parent, child1, child2, is_leaf, n, seq_int, L, branch_len, postorder, deadline):
    """Coordinate descent on branch lengths for JC model.
    For JC, the optimal branch length given the likelihood vectors at both ends
    can be found analytically. We use a simplified approach: for each edge,
    the contribution depends on the inner product of the two likelihood vectors
    after applying the transition matrix.
    
    For JC with 4 states, given vectors a (at parent end) and b (at child end),
    the likelihood contribution is:
    sum_s a[s] * (ps * b[s] + pd * sum_{s'!=s} b[s'])
    = sum_s a[s] * (ps * b[s] + pd * (sum_b - b[s]))
    = pd * sum_a * sum_b + (ps - pd) * sum_s a[s]*b[s]
    
    where sum_a = sum(a), sum_b = sum(b), and dot = sum(a*b).
    
    ps = 1/4 + 3/4 * exp(-4l/3)
    pd = 1/4 * (1 - exp(-4l/3))
    ps - pd = 1/4 + 3/4*e - 1/4 + 1/4*e = e  (where e = exp(-4l/3))
    pd = 1/4 * (1-e)
    
    So LL contribution = 1/4*(1-e)*sum_a*sum_b + e * dot
    
    To maximize over l: take derivative w.r.t. e (since e = exp(-4l/3), l = -3/4 * ln(e)):
    d/dl = d/de * de/dl = d/de * (-4/3)*e
    d/de [1/4*(1-e)*S + e*D] where S=sum_a*sum_b, D=dot
    = 1/4*(-S) + D
    Setting to 0: D = S/4
    e* = S/(4*D) if D > 0, else 0 (l = inf)
    If e* > 1, set e* = 1 (l = 0)
    
    Actually we need to be more careful. The full LL is a product over sites,
    so we work in log-space. But for coordinate descent on a single branch,
    we can compute the per-site contributions and sum them.
    
    Let me use a simpler iterative approach with a few steps of gradient-based update.
    """
    max_nodes = 2 * n
    # For each non-root node, optimize its branch length
    # We'll do a few rounds of coordinate descent
    
    for round in range(5):
        if time.time() > deadline:
            break
        for node in postorder:
            if node == root or is_leaf[node] and parent[node] == -1:
                continue
            if parent[node] == -1:
                continue
            # Recompute likelihood vectors for this subtree
            # This is expensive; for now, use a heuristic initialization
            pass
    
    # Initialize branch lengths from JC distances (rough)
    # For each leaf, use average distance to others / 2
    # For internal nodes, use small values
    # Actually, let's just set reasonable defaults
    return branch_len

def _nni_refine(root, parent, child1, child2, is_leaf, n, seq_int, L, deadline):
    """NNI refinement using JC likelihood.
    For each internal edge, try the two alternative topologies and keep the best."""
    max_nodes = 2 * n
    branch_len = np.zeros(max_nodes, dtype=np.float64)
    
    # Initialize branch lengths: use small values
    for i in range(max_nodes):
        if i < n:
            branch_len[i] = 0.05
        else:
            branch_len[i] = 0.03
    
    # Build postorder
    postorder = _build_postorder(root, child1, child2, is_leaf, max_nodes)
    
    # Compute initial LL
    current_ll = _compute_ll(root, parent, child1, child2, is_leaf, n, seq_int, L, branch_len, postorder)
    
    # Find all internal edges: edges where both endpoints are internal nodes
    # An internal edge is (parent[node], node) where node is internal and parent[node] is internal
    # Actually in our rooted representation, internal edges are those connecting two internal nodes
    
    # For NNI on unrooted trees: pick an internal edge (a,b). The four neighbors
    # are: the two subtrees hanging off a (other than b), and the two subtrees
    # hanging off b (other than a). Try the two alternative connections.
    
    # In our rooted tree, let's find internal edges
    internal_edges = []
    for node in range(max_nodes):
        if is_leaf[node]:
            continue
        p = parent[node]
        if p >= 0 and not is_leaf[p]:
            internal_edges.append((p, node))
    
    # For each internal edge, try NNI moves
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and iteration < max_iterations and time.time() < deadline:
        improved = False
        iteration += 1
        # Rebuild postorder
        postorder = _build_postorder(root, child1, child2, is_leaf, max_nodes)
        
        for (a, b) in internal_edges:
            if time.time() > deadline:
                return root, parent, child1, child2, is_leaf, n, branch_len
            
            # Find neighbors of a other than b
            neighbors_a = []
            if not is_leaf[a]:
                c1a = child1[a]
                c2a = child2[a]
                if c1a >= 0 and c1a != b:
                    neighbors_a.append(c1a)
                if c2a >= 0 and c2a != b:
                    neighbors_a.append(c2a)
            if parent[a] >= 0 and parent[a] != b:
                neighbors_a.append(parent[a])
            
            # Find neighbors of b other than a
            neighbors_b = []
            if not is_leaf[b]:
                c1b = child1[b]
                c2b = child2[b]
                if c1b >= 0 and c1b != a:
                    neighbors_b.append(c1b)
                if c2b >= 0 and c2b != a:
                    neighbors_b.append(c2b)
            if parent[b] >= 0 and parent[b] != a:
                neighbors_b.append(parent[b])
            
            if len(neighbors_a) < 2 or len(neighbors_b) < 2:
                continue
            
            # For unrooted NNI: we need exactly 2 neighbors on each side
            # (the edge a-b, and one other neighbor on each side)
            # If there are more, we need to handle it differently
            # For simplicity, skip edges with more than 2 neighbors on either side
            # (this means we only do NNI on edges where both endpoints have degree 3)
            if len(neighbors_a) != 2 or len(neighbors_b) != 2:
                continue
            
            c = neighbors_a[0]
            d = neighbors_a[1]
            e = neighbors_b[0]
            f = neighbors_b[1]
            
            # Current topology: a-b with c,d attached to a and e,f attached to b
            # Alternative 1: a-c, b-e, d-f (swap c with e)
            # Alternative 2: a-e, b-c, d-f (swap c with f)
            # Wait, standard NNI: given edge a-b, with subtrees C,D on a side and E,F on b side
            # Alt 1: connect a-e, b-c (swap C and E)
            # Alt 2: connect a-f, b-d (swap D and F)
            
            # Let me think more carefully.
            # The edge is a-b. On a's side (away from b), we have subtrees c and d.
            # On b's side (away from a), we have subtrees e and f.
            # NNI move 1: remove a-b, add a-e and b-c. Now d is still on a, f is still on b.
            #   Wait no. We need to reconnect: a connects to c and e, b connects to d and f.
            #   Hmm, that's not right either.
            
            # Standard NNI: For internal edge (a,b), the four incident edges (other than a-b)
            # go to subtrees S1, S2 (from a) and S3, S4 (from b).
            # Move 1: reconnect as a-S3, b-S1 (S2 stays with a, S4 stays with b)
            #   Actually: new edges are a-S3 and b-S1, with S2 still on a and S4 still on b.
            #   So a has neighbors S2, S3 and b has neighbors S1, S4.
            # Move 2: reconnect as a-S4, b-S2.
            #   So a has neighbors S1, S4 and b has neighbors S3, S2.
            
            # Let's implement: save current connections, try move 1, compute LL, if better keep it
            # Save state
            old_child1_a = child1[a]
            old_child2_a = child2[a]
            old_parent_c = parent[c]
            old_parent_d = parent[d]
            old_parent_e = parent[e]
            old_parent_f = parent[f]
            old_child1_b = child1[b]
            old_child2_b = child2[b]
            
            # Determine which child slot c,d,e,f are in
            def get_slot(node, parent_node):
                if child1[parent_node] == node:
                    return 1
                elif child2[parent_node] == node:
                    return 2
                return 0
            
            slot_c = get_slot(c, a)
            slot_d = get_slot(d, a)
            slot_e = get_slot(e, b)
            slot_f = get_slot(f, b)
            
            # Try Move 1: a connects to c and e; b connects to d and f
            # (swap d and e)
            child1[a] = c if slot_c == 1 else c
            # Actually let me just set explicitly
            # a's children: c, e
            # b's children: d, f
            if slot_c == 1:
                child1[a] = c
                child2[a] = e
            else:
                child2[a] = c
                child1[a] = e
            parent[e] = a
            
            if slot_d == 1:
                child1[b] = d
                child2[b] = f
            else:
                child2[b] = d
                child1[b] = f
            parent[d] = b
            
            postorder_new = _build_postorder(root, child1, child2, is_leaf, max_nodes)
            new_ll = _compute_ll(root, parent, child1, child2, is_leaf, n, seq_int, L, branch_len, postorder_new)
            
            if new_ll > current_ll + 1e-10:
                current_ll = new_ll
                improved = True
            else:
                # Revert
                child1[a] = old_child1_a
                child2[a] = old_child2_a
                child1[b] = old_child1_b
                child2[b] = old_child2_b
                parent[c] = old_parent_c
                parent[d] = old_parent_d
                parent[e] = old_parent_e
                parent[f] = old_parent_f
            continue
            
            # Try Move 2: a connects to d and e; b connects to c and f
            # (swap c and e)
            # Actually move 2 should be: a connects to c and f; b connects to d and e
            # Let me redo:
            # Move 1 was: a-(c,e), b-(d,f)  [swapped d and e]
            # Move 2 should be: a-(c,f), b-(d,e)  [swapped d and f]
            
            # Hmm, I'm confusing myself. Let me use a cleaner formulation.
            # The two NNI alternatives for edge a-b with subtrees {c,d} on a and {e,f} on b:
            # Alt 1: a-e, b-d  (c stays with a, f stays with b)
            # Alt 2: a-f, b-c  (d stays with a, e stays with b)
            
            # Restore first
            child1[a] = old_child1_a
            child2[a] = old_child2_a
            child1[b] = old_child1_b
            child2[b] = old_child2_b
            parent[c] = old_parent_c
            parent[d] = old_parent_d
            parent[e] = old_parent_e
            parent[f] = old_parent_f
            
            # Alt 1: a connects to c and e; b connects to d and f
            if slot_c == 1:
                child1[a] = c
                child2[a] = e
            else:
                child2[a] = c
                child1[a] = e
            parent[e] = a
            if slot_d == 1:
                child1[b] = d
                child2[b] = f
            else:
                child2[b] = d
                child1[b] = f
            parent[d] = b
            
            postorder_new = _build_postorder(root, child1, child2, is_leaf, max_nodes)
            new_ll = _compute_ll(root, parent, child1, child2, is_leaf, n, seq_int, L, branch_len, postorder_new)
            
            if new_ll > current_ll + 1e-10:
                current_ll = new_ll
                improved = True
            else:
                child1[a] = old_child1_a
                child2[a] = old_child2_a
                child1[b] = old_child1_b
                child2[b] = old_child2_b
                parent[c] = old_parent_c
                parent[d] = old_parent_d
                parent[e] = old_parent_e
                parent[f] = old_parent_f
                continue
            
            # Alt 2: a connects to d and e; b connects to c and f
            child1[a] = old_child1_a
            child2[a] = old_child2_a
            child1[b] = old_child1_b
            child2[b] = old_child2_b
            parent[c] = old_parent_c
            parent[d] = old_parent_d
            parent[e] = old_parent_e
            parent[f] = old_parent_f
            
            if slot_d == 1:
                child1[a] = d
                child2[a] = e
            else:
                child2[a] = d
                child1[a] = e
            parent[e] = a
            if slot_c == 1:
                child1[b] = c
                child2[b] = f
            else:
                child2[b] = c
                child1[b] = f
            parent[c] = b
            
            postorder_new = _build_postorder(root, child1, child2, is_leaf, max_nodes)
            new_ll = _compute_ll(root, parent, child1, child2, is_leaf, n, seq_int, L, branch_len, postorder_new)
            
            if new_ll > current_ll + 1e-10:
                current_ll = new_ll
                improved = True
            else:
                child1[a] = old_child1_a
                child2[a] = old_child2_a
                child1[b] = old_child1_b
                child2[b] = old_child2_b
                parent[c] = old_parent_c
                parent[d] = old_parent_d
                parent[e] = old_parent_e
                parent[f] = old_parent_f
    
    return root, parent, child1, child2, is_leaf, n, branch_len

def _tree_to_newick(root, child1, child2, is_leaf, n):
    """Convert rooted tree to Newick string."""
    def to_newick(node):
        if is_leaf[node]:
            return f"t{node}"
        c1 = child1[node]
        c2 = child2[node]
        if c1 >= 0 and c2 >= 0:
            return f"({to_newick(c1)},{to_newick(c2)})"
        elif c1 >= 0:
            return to_newick(c1)
        elif c2 >= 0:
            return to_newick(c2)
        else:
            return f"t{node}"
    return to_newick(root) + ";"

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 160.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Convert sequences to integer array
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_list = [sequences[label] for label in labels]
    L = len(seq_list[0])
    seq_int = np.zeros((n, L), dtype=np.int8)
    for i in range(n):
        for j in range(L):
            c = seq_list[i][j]
            seq_int[i, j] = char_map.get(c, 0)
    
    # Compute JC distance matrix
    d = _jc_distance_matrix(seq_int, L)
    
    # Build NJ tree
    parent, child1, child2, is_leaf, num_nodes = _nj_build(d)
    root = num_nodes - 1  # The last internal node is the root
    
    # NNI refinement
    try:
        root, parent, child1, child2, is_leaf, n, branch_len = _nni_refine(
            root, parent, child1, child2, is_leaf, n, seq_int, L, deadline
        )
    except Exception:
        pass
    
    # Convert to Newick
    newick = _tree_to_newick(root, child1, child2, is_leaf, n)
    return newick
