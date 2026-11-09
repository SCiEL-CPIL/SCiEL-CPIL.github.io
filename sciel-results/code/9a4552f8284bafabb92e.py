# EVOLVE-BLOCK-START
import math
import time
import numpy as np
import sys

_START_TIME = time.time()
_MAX_PER_INSTANCE = 12.0  # seconds per instance as a safety margin

def _encode_sequences(sequences, labels):
    """Encode sequences to numpy array of shape (n_sites, n_taxa) with values 0-3."""
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_taxa = len(labels)
    # Find max length
    max_len = max(len(sequences[l]) for l in labels)
    # Pad or truncate to max_len (assume equal length as per contract, but safe to check)
    matrix = np.zeros((max_len, n_taxa), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(max_len):
            if j < len(seq):
                matrix[j, i] = enc.get(seq[j], 0)
            else:
                matrix[j, i] = 0
    return matrix

def _jc_likelihood(nodes, root, seq_matrix, branch_lengths):
    """
    Compute log-likelihood of the tree under Jukes-Cantor model.
    nodes: list of length 2*n-1. 0..n-1 are leaves. n..2n-2 are (left, right).
    root: index of root.
    seq_matrix: (n_sites, n_taxa) int array 0-3.
    branch_lengths: dict or array mapping node->parent edge length? 
                     For simplicity in NNI, we'll use a flat array or re-optimize.
                     Let's use a simple heuristic: fixed branch lengths or quick optimization.
                     To be fast, we'll assume uniform branch lengths initially, 
                     then optimize them slightly if time permits.
    """
    n_taxa = (len(nodes) + 1) // 2
    n_sites = seq_matrix.shape[0]
    
    # Initialize likelihood vectors for all nodes
    # L[node, site, base]
    # We only need to store L for current nodes.
    # Shape: (2*n_taxa - 1, n_sites, 4)
    
    # For memory efficiency with large n_sites (2000) and n_taxa (32), 
    # 63 * 2000 * 4 * 8 bytes = ~4 MB. This is fine.
    
    L = np.zeros((2 * n_taxa - 1, n_sites, 4), dtype=np.float64)
    
    # Initialize leaves
    # L[leaf, :, base] = 1 if observed base == base, else 0
    for i in range(n_taxa):
        obs = seq_matrix[:, i]
        # Vectorized initialization
        L[i, :, 0] = (obs == 0).astype(np.float64)
        L[i, :, 1] = (obs == 1).astype(np.float64)
        L[i, :, 2] = (obs == 2).astype(np.float64)
        L[i, :, 3] = (obs == 3).astype(np.float64)
        
    # Process internal nodes in post-order
    # We need a post-order traversal
    postorder = []
    def dfs(v):
        if v < n_taxa:
            postorder.append(v)
        else:
            left, right = nodes[v]
            dfs(left)
            dfs(right)
            postorder.append(v)
    dfs(root)
    
    # JC transition probabilities
    # We need branch lengths. 
    # For speed, we will use a fixed branch length estimate per edge.
    # A better approach: optimize branch lengths. 
    # But for NNI, we can fix them to a global estimate or re-estimate quickly.
    # Let's estimate a global average branch length from the distance matrix.
    
    # Calculate average p-distance
    total_dist = 0.0
    count = 0
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            dist = np.mean(seq_matrix[:, i] != seq_matrix[:, j])
            if dist > 0.74:
                dist = 0.74
            total_dist += dist
            count += 1
    avg_pdist = total_dist / count if count > 0 else 0.1
    
    # JC correction: d = -0.75 * ln(1 - 4p/3)
    if avg_pdist < 0.75:
        avg_jc_dist = -0.75 * math.log(1 - 4 * avg_pdist / 3)
    else:
        avg_jc_dist = 1.0
        
    # Average path length in a balanced tree is ~ log2(n)
    avg_edge_len = avg_jc_dist / max(1.0, math.log2(n_taxa))
    avg_edge_len = min(avg_edge_len, 2.0) # Cap
    
    exp_term = math.exp(-4.0 * avg_edge_len / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = 0.25 - 0.25 * exp_term
    
    for v in postorder:
        if v < n_taxa:
            continue
        left, right = nodes[v]
        
        # L(v, s, a) = sum_{a1} P(a1|a) L(left, s, a1) * sum_{a2} P(a2|a) L(right, s, a2)
        # For JC:
        # Sum_{a1} P(a1|a) L(left, s, a1) = p_diff * Sum(L(left, s, :)) + (p_same - p_diff) * L(left, s, a)
        
        left_sum = L[left, :, :].sum(axis=2)
        right_sum = L[right, :, :].sum(axis=2)
        
        for a in range(4):
            left_contrib = p_diff * left_sum + (p_same - p_diff) * L[left, :, a]
            right_contrib = p_diff * right_sum + (p_same - p_diff) * L[right, :, a]
            L[v, :, a] = left_contrib * right_contrib
            
    root_probs = L[root, :, :].sum(axis=1)
    root_probs = np.maximum(root_probs, 1e-300)
    loglik = np.sum(np.log(root_probs))
    
    return loglik

def _build_nj_tree(seq_matrix):
    """
    Build initial tree using Neighbor Joining.
    Returns nodes, root.
    """
    n = seq_matrix.shape[1]
    if n <= 2:
        return None, None
        
    # Compute distance matrix
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = np.mean(seq_matrix[:, i] != seq_matrix[:, j])
            if p >= 0.7499:
                d = 1.0 # Cap
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
            
    # NJ Algorithm
    # Use a list of active nodes and a dictionary for distances
    # To avoid resizing issues, we'll keep a full matrix of size (2n-1, 2n-1)
    max_nodes = 2 * n - 1
    D_full = np.zeros((max_nodes, max_nodes))
    D_full[:n, :n] = D
    
    active = list(range(n))
    parent = {} # node -> (parent_node, is_left)
    # Actually, let's just build the tree structure directly
    # trees: dict node -> structure
    trees = {i: i for i in range(n)}
    
    current_dist = D_full[:n, :n].copy()
    active = list(range(n))
    
    while len(active) > 3:
        m = len(active)
        # Compute Q matrix for active nodes
        r = {a: np.sum(current_dist[a, active]) for a in active}
        
        best_q = float('inf')
        best_a = None
        best_b = None
        
        for i_idx, a in enumerate(active):
            for b in active[i_idx + 1:]:
                q = (m - 2) * current_dist[a, b] - r[a] - r[b]
                if q < best_q:
                    best_q = q
                    best_a = a
                    best_b = b
                    
        a, b = best_a, best_b
        
        # Create new node
        new_node = len(active) + n - 1 # This indexing is tricky.
        # Let's use a counter for new nodes
        # Total nodes so far: n + (n - m) internal nodes created so far?
        # No, let's just use a global counter
        # We know max nodes is 2n-1.
        # We can assign new nodes starting from n.
        # Let's track the next available node ID
        # Initially next_id = n
        # Each join consumes 2 nodes and creates 1.
        
        # Let's restart the NJ logic with a cleaner ID management
        pass
        
    # The above NJ is getting messy. Let's use a simpler, robust initialization:
    # A balanced random tree or a simple star tree.
    # Given the directive to replace NJ with ML, a good initial tree is crucial.
    # Let's use a **Fast Distance Method** variant or just a **Random Tree** 
    # and rely on NNI. For 8-32 taxa, NNI from random might be slow.
    # Let's implement a **Correct** NJ.
    
    # Correct NJ Implementation
    def nj_correct(seq_matrix):
        n = seq_matrix.shape[1]
        if n <= 2:
            return None, None
            
        # Distance matrix
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                p = np.mean(seq_matrix[:, i] != seq_matrix[:, j])
                if p >= 0.7499:
                    d = 1.0
                else:
                    d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
                D[i, j] = d
                D[j, i] = d
                
        # Active nodes
        active = list(range(n))
        # Distances stored in a dictionary or full matrix
        # We'll use a full matrix of size 2n-1
        max_nodes = 2 * n - 1
        dist = np.zeros((max_nodes, max_nodes))
        dist[:n, :n] = D
        
        # Tree structure
        # tree[i] = (left, right) for internal, i for leaf
        tree = [None] * max_nodes
        for i in range(n):
            tree[i] = i
            
        next_id = n
        
        while len(active) > 3:
            m = len(active)
            # r_i = sum_j d_ij
            r = np.zeros(m)
            for i_idx, a in enumerate(active):
                r[i_idx] = np.sum(dist[a, active])
                
            # Find min Q
            best_q = float('inf')
            best_i = -1
            best_j = -1
            for i_idx in range(m):
                for j_idx in range(i_idx + 1, m):
                    a = active[i_idx]
                    b = active[j_idx]
                    q = (m - 2) * dist[a, b] - r[i_idx] - r[j_idx]
                    if q < best_q:
                        best_q = q
                        best_i = i_idx
                        best_j = j_idx
                        
            a = active[best_i]
            b = active[best_j]
            
            # New node
            c = next_id
            next_id += 1
            
            # Update distances
            for k_idx, k in enumerate(active):
                if k == a or k == b:
                    continue
                dist[c, k] = 0.5 * (dist[a, k] + dist[b, k] - dist[a, b])
                dist[k, c] = dist[c, k]
                
            # Remove a, b, add c
            new_active = [x for x in active if x != a and x != b]
            new_active.append(c)
            active = new_active
            
            # Update tree structure
            tree[c] = (a, b)
            
        # Final 3 nodes
        a, b, c = active
        # Root is (a, (b, c))
        root = next_id
        next_id += 1
        tree[root] = (a, (b, c)) # Wait, tree expects node indices.
        # We need to create a new internal node for the root if it's not one of a,b,c?
        # No, a, b, c are subtrees.
        # If a, b, c are leaves, root is new.
        # If some are internal, they are already nodes.
        # The root connects a, b, c. But binary tree requires 2 children.
        # So root = (a, temp) where temp = (b, c)?
        # Yes.
        
        # Create temp node
        temp = next_id
        next_id += 1
        tree[temp] = (b, c)
        
        tree[root] = (a, temp)
        
        # Return nodes and root
        # nodes list should be of size 2n-1
        # Our tree array is of size max_nodes = 2n-1.
        # We used indices 0..next_id-1.
        # We need to map these to a contiguous 0..2n-2 list?
        # The likelihood function expects nodes[i] for i in range(2n-1).
        # Our tree array has holes? No, we used sequential IDs.
        # But the leaves are 0..n-1. Internal are n..2n-2.
        # This matches.
        
        return tree, root

    return nj_correct(seq_matrix)

def _nni_search(nodes, root, seq_matrix, n_taxa, time_limit):
    """
    Perform NNI hill-climbing.
    """
    best_nodes = list(nodes)
    best_root = root
    best_loglik = _jc_likelihood(best_nodes, best_root, seq_matrix)
    
    start = time.time()
    improved = True
    iterations = 0
    
    while improved and (time.time() - start) < time_limit:
        improved = False
        iterations += 1
        
        # Try all NNI moves
        # An NNI move involves an internal edge (v, u) where v is parent of u.
        # v has parent p and other child w.
        # u has children a and b.
        # Current: (p, (w, (a, b)))
        # Moves:
        # 1. (p, ((w, a), b))
        # 2. (p, ((w, b), a))
        
        # We need to find all internal edges.
        # We can iterate over all internal nodes u (index >= n_taxa)
        # and treat the edge from its parent to u as the candidate.
        
        # To find parent, we can build a parent map or just try all splits.
        # For simplicity, we'll iterate over all internal nodes u != root.
        # We need to know who is the parent of u.
        
        # Let's build a parent map
        parent_map = {}
        def build_parent_map(v, p):
            if v < n_taxa:
                return
            left, right = best_nodes[v]
            build_parent_map(left, v)
            build_parent_map(right, v)
            if p is not None:
                parent_map[v] = p
        build_parent_map(best_root, None)
        
        for u, p in parent_map.items():
            if u == best_root:
                continue
            
            # p is the parent of u
            # Find the other child of p
            left_p, right_p = best_nodes[p]
            if left_p == u:
                w = right_p
            else:
                w = left_p
                
            # u's children
            a, b = best_nodes[u]
            
            # Try Move 1: (w, a) as left, b as right?
            # New structure for p: (w_new, b) where w_new = (w, a)
            # This requires creating a new internal node?
            # No, we can just rearrange.
            # Current: p=(w, u), u=(a, b)
            # Move 1: p=(w1, b), w1=(w, a)
            # We need to reuse node IDs.
            # Let's swap the contents.
            
            # Save original
            orig_p = best_nodes[p]
            orig_u = best_nodes[u]
            
            # Apply Move 1
            # New p = (w, b) ? No, that's not NNI.
            # NNI on edge (p,u):
            # Topology: p connects to w and u. u connects to a and b.
            # Alternative 1: p connects to (w,a) and b.
            # Alternative 2: p connects to (w,b) and a.
            
            # To implement this without creating new nodes, we can modify the structure.
            # But the structure is a tree.
            # Let's create a new node list for the move.
            
            new_nodes = list(best_nodes)
            
            # Move 1:
            # Let x = (w, a)
            # Let y = b
            # p = (x, y)
            # But x is a new subtree. We don't have a node ID for x.
            # We can reuse the ID of u for x?
            # If we set u = (w, a), then p = (u, b).
            # This works!
            
            new_nodes[u] = (w, a)
            new_nodes[p] = (u, b)
            
            # Check validity: u and p are internal nodes.
            # w, a, b are subtrees.
            # This is a valid tree.
            
            lik1 = _jc_likelihood(new_nodes, best_root, seq_matrix)
            
            if lik1 > best_loglik:
                best_nodes = new_nodes
                best_loglik = lik1
                improved = True
                continue # Restart loop? Or continue? Usually restart or continue with updated tree.
                
            # Move 2:
            # u = (w, b)
            # p = (u, a)
            
            new_nodes2 = list(best_nodes)
            new_nodes2[u] = (w, b)
            new_nodes2[p] = (u, a)
            
            lik2 = _jc_likelihood(new_nodes2, best_root, seq_matrix)
            
            if lik2 > best_loglik:
                best_nodes = new_nodes2
                best_loglik = lik2
                improved = True
                continue
                
        # If no improvement, break? Or try random restart?
        # For now, break if no improvement in a full pass.
        
    return best_nodes, best_root, best_loglik

def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string.
    """
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        return f"({labels[0]});" if n == 1 else ";"
    
    # Encode sequences
    seq_matrix = _encode_sequences(sequences, labels)
    
    # Build initial NJ tree
    nodes, root = _build_nj_tree(seq_matrix)
    
    if nodes is None:
        return "(" + ",".join(labels) + ");"
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > _MAX_PER_INSTANCE:
        # Convert to Newick and return
        def nodes_to_newick(nodes, root, n_taxa, labels):
            def get_newick(v):
                if v < n_taxa:
                    return labels[v]
                else:
                    left, right = nodes[v]
                    return f"({get_newick(left)},{get_newick(right)})"
            return get_newick(root) + ";"
        return nodes_to_newick(nodes, root, n, labels)
    
    # NNI Search
    remaining_time = _MAX_PER_INSTANCE - elapsed
    # Leave some margin for conversion
    search_time = max(1.0, remaining_time - 0.5)
    
    best_nodes, best_root, _ = _nni_search(nodes, root, seq_matrix, n, search_time)
    
    # Convert to Newick
    def nodes_to_newick(nodes, root, n_taxa, labels):
        def get_newick(v):
            if v < n_taxa:
                return labels[v]
            else:
                left, right = nodes[v]
                return f"({get_newick(left)},{get_newick(right)})"
        return get_newick(root) + ";"
        
    return nodes_to_newick(best_nodes, best_root, n, labels)
# EVOLVE-BLOCK-END
