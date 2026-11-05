"""Reconstruct phylogenetic topology using Fast NJ + Quartet-Score NNI refinement."""
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 170.0  # Safety margin for 180s limit

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Use vectorized comparison if possible, but loop is okay for N<=32
    # For speed, we can use numpy arrays
    try:
        seq_matrix = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    except:
        seq_matrix = np.array([[0 if c=='A' else 1 if c=='C' else 2 if c=='G' else 3 for c in s] for s in seqs], dtype=np.int8)
        
    # Vectorized difference count
    # seq_matrix is (n, L)
    # We want (n, n) matrix of differences
    # diff[i, j, k] = 1 if seq[i, k] != seq[j, k]
    # This creates a (n, n, L) array which might be large if L is 2000 and n is 32.
    # 32 * 32 * 2000 = 2,048,000 ints. This is fine.
    diff = (seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]).sum(axis=-1)
    
    p = diff / length
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(d, 0.0)
    
    return d

def _nj_build(D):
    """
    Fast Neighbor Joining.
    Returns: (adj, branch_lengths)
    adj: dict node_id -> list of neighbors
    branch_lengths: dict (u, v) -> length
    """
    n = D.shape[0]
    if n <= 2:
        return {}, {}
    
    active = list(range(n))
    D_cur = D.copy()
    
    adj = {i: [] for i in range(n)}
    branch_lengths = {}
    
    next_node_id = n
    
    for _ in range(n - 2):
        m = len(active)
        if m == 3:
            break
        
        sub_D = D_cur[np.ix_(active, active)]
        r = sub_D.sum(axis=1)
        
        q = (m - 2) * sub_D - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(q, np.inf)
        
        idx_flat = np.argmin(q)
        i_idx, j_idx = np.unravel_index(idx_flat, q.shape)
        
        a = active[i_idx]
        b = active[j_idx]
        
        # Compute branch lengths
        # delta = (r_a - r_b) / (2*(m-2))
        delta = (r[i_idx] - r[j_idx]) / (2.0 * (m - 2))
        
        ab = D_cur[a, b]
        len_a = 0.5 * ab + delta
        len_b = 0.5 * ab - delta
        
        if len_a < 0: len_a = 0
        if len_b < 0: len_b = 0
            
        new_node = next_node_id
        next_node_id += 1
        
        adj[new_node] = [a, b]
        adj[a].append(new_node)
        adj[b].append(new_node)
        
        branch_lengths[(a, new_node)] = len_a
        branch_lengths[(new_node, a)] = len_a
        branch_lengths[(b, new_node)] = len_b
        branch_lengths[(new_node, b)] = len_b
        
        # Update distances
        rem_active = [x for x in active if x != a and x != b]
        
        if rem_active:
            da_rem = D_cur[a, rem_active]
            db_rem = D_cur[b, rem_active]
            new_d = 0.5 * (da_rem + db_rem - ab)
            D_cur[new_node, rem_active] = new_d
            D_cur[rem_active, new_node] = new_d
            
        active = rem_active + [new_node]
        
        if time.time() > _DEADLINE - 5.0:
            break
            
    # Handle the final 3 nodes
    if len(active) == 3:
        a, b, c = active
        new_node = next_node_id
        next_node_id += 1
        
        d_ab = D_cur[a, b]
        d_bc = D_cur[b, c]
        d_ac = D_cur[a, c]
        
        len_a = 0.5 * (d_ab + d_ac - d_bc)
        len_b = 0.5 * (d_ab + d_bc - d_ac)
        len_c = 0.5 * (d_ac + d_bc - d_ab)
        
        len_a = max(0, len_a)
        len_b = max(0, len_b)
        len_c = max(0, len_c)
        
        adj[new_node] = [a, b, c]
        adj[a].append(new_node)
        adj[b].append(new_node)
        adj[c].append(new_node)
        
        branch_lengths[(a, new_node)] = len_a
        branch_lengths[(new_node, a)] = len_a
        branch_lengths[(b, new_node)] = len_b
        branch_lengths[(new_node, b)] = len_b
        branch_lengths[(c, new_node)] = len_c
        branch_lengths[(new_node, c)] = len_c
        
    return adj, branch_lengths

def _get_leaf_sets(adj, n):
    """
    Compute the set of leaves for each node in the tree.
    Returns: dict node_id -> set of leaf indices
    """
    leaf_sets = {}
    
    def get_leaves(node, parent):
        if node < n:
            return {node}
        
        leaves = set()
        for v in adj[node]:
            if v != parent:
                leaves |= get_leaves(v, node)
        return leaves
    
    # We need a root to traverse. Pick an internal node.
    # Find an internal node
    root = None
    for i in range(n, len(adj)):
        root = i
        break
        
    if root is None:
        # Star tree?
        for i in range(n):
            leaf_sets[i] = {i}
        return leaf_sets
        
    # Calculate leaf sets for all nodes
    # We can do a DFS from the root
    def dfs(node, parent):
        leaves = set()
        for v in adj[node]:
            if v == parent:
                continue
            sub_leaves = dfs(v, node)
            leaves |= sub_leaves
            leaf_sets[v] = sub_leaves
        if node < n:
            leaves = {node}
        leaf_sets[node] = leaves
        return leaves
        
    dfs(root, -1)
    
    return leaf_sets

def _nni_refine_quartet(D, adj, branch_lengths, n):
    """
    NNI refinement using Quartet Score.
    """
    # Precompute leaf sets
    leaf_sets = _get_leaf_sets(adj, n)
    
    improved = True
    max_iter = 50
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time.time() > _DEADLINE - 2.0:
            break
            
        edges_checked = set()
        for u in range(n, len(adj)):
            for v in adj[u]:
                if v < n:
                    continue
                if (u, v) in edges_checked or (v, u) in edges_checked:
                    continue
                edges_checked.add((u, v))
                
                u_neighbors = [x for x in adj[u] if x != v]
                if len(u_neighbors) != 2:
                    continue
                u1, u2 = u_neighbors
                
                v_neighbors = [x for x in adj[v] if x != u]
                if len(v_neighbors) != 2:
                    continue
                v1, v2 = v_neighbors
                
                # Get leaf sets
                L_u1 = leaf_sets[u1]
                L_u2 = leaf_sets[u2]
                L_v1 = leaf_sets[v1]
                L_v2 = leaf_sets[v2]
                
                n_u1 = len(L_u1)
                n_u2 = len(L_u2)
                n_v1 = len(L_v1)
                n_v2 = len(L_v2)
                
                # If any clade is too large, the computation might be slow?
                # But N is small (<=32).
                
                # Compute sums of distances
                # S_AC = sum(D[a, c] for a in L_u1, c in L_v1)
                # We can use numpy for this
                
                def sum_dist(set1, set2):
                    if not set1 or not set2:
                        return 0.0
                    idx1 = list(set1)
                    idx2 = list(set2)
                    # D is a numpy array
                    # D[np.ix_(idx1, idx2)].sum()
                    return D[np.ix_(idx1, idx2)].sum()
                
                S_u1_v1 = sum_dist(L_u1, L_v1) # S_AC
                S_u1_v2 = sum_dist(L_u1, L_v2) # S_AD
                S_u2_v1 = sum_dist(L_u2, L_v1) # S_BC
                S_u2_v2 = sum_dist(L_u2, L_v2) # S_BD
                S_u1_u2 = sum_dist(L_u1, L_u2) # S_AB
                S_v1_v2 = sum_dist(L_v1, L_v2) # S_CD
                
                # Current Topology: (u1, u2) | (v1, v2) => (AB | CD)
                # Score contribution for quartets a in A, b in B, c in C, d in D:
                # Q_old = sum( D_ab + D_cd - D_ac - D_bd )
                # = n_C n_D S_AB + n_A n_B S_CD - n_B n_D S_AC - n_A n_C S_BD
                
                term_old = n_v1 * n_v2 * S_u1_u2 + n_u1 * n_u2 * S_v1_v2 - n_u2 * n_v2 * S_u1_v1 - n_u1 * n_v1 * S_u2_v2
                
                # Alternative 1: (u1, v1) | (u2, v2) => (AC | BD)
                # Q_new1 = sum( D_ac + D_bd - D_ab - D_cd )
                # = n_B n_D S_AC + n_A n_C S_BD - n_C n_D S_AB - n_A n_B S_CD
                
                term_new1 = n_u2 * n_v2 * S_u1_v1 + n_u1 * n_v1 * S_u2_v2 - n_v1 * n_v2 * S_u1_u2 - n_u1 * n_u2 * S_v1_v2
                
                # Alternative 2: (u1, v2) | (u2, v1) => (AD | BC)
                # Q_new2 = sum( D_ad + D_bc - D_ab - D_cd )
                # = n_B n_C S_AD + n_A n_D S_BC - n_C n_D S_AB - n_A n_B S_CD
                
                term_new2 = n_u2 * n_v1 * S_u1_v2 + n_u1 * n_v2 * S_u2_v1 - n_v1 * n_v2 * S_u1_u2 - n_u1 * n_u2 * S_v1_v2
                
                # We want to minimize the total score.
                # The change in score is term_new - term_old.
                # Note: The terms S_u1_u2 and S_v1_v2 are the same in all 3 cases?
                # Let's check.
                # Old: (AB|CD). Pairs (A,B) and (C,D) are "close".
                # New1: (AC|BD). Pairs (A,C) and (B,D) are "close".
                # New2: (AD|BC). Pairs (A,D) and (B,C) are "close".
                
                # The "far" pairs in Old are (A,C), (A,D), (B,C), (B,D).
                # In the formula Q = D_close - D_far.
                # For Old: D_close = D_AB, D_CD. D_far = D_AC, D_AD, D_BC, D_BD.
                # Wait, my formula above:
                # Q_old = S_AB (scaled) + S_CD (scaled) - S_AC (scaled) - S_BD (scaled) ... 
                # Let's re-verify the quartet score sum.
                # Sum over a in A, b in B, c in C, d in D of (D_ab + D_cd - D_ac - D_bd).
                # Term 1: sum D_ab = n_C n_D S_AB.
                # Term 2: sum D_cd = n_A n_B S_CD.
                # Term 3: sum D_ac = n_B n_D S_AC.
                # Term 4: sum D_bd = n_A n_C S_BD.
                # So Q_old = n_C n_D S_AB + n_A n_B S_CD - n_B n_D S_AC - n_A n_C S_BD.
                
                # For New1 (AC|BD):
                # Topology is (A, C) | (B, D).
                # Close pairs: (A, C) and (B, D).
                # Far pairs: (A, B), (A, D), (C, B), (C, D).
                # Q_new1 = n_B n_D S_AC + n_A n_C S_BD - n_C n_D S_AB - n_A n_B S_CD.
                
                # For New2 (AD|BC):
                # Topology is (A, D) | (B, C).
                # Close pairs: (A, D) and (B, C).
                # Far pairs: (A, B), (A, C), (D, B), (D, C).
                # Q_new2 = n_B n_C S_AD + n_A n_D S_BC - n_C n_D S_AB - n_A n_B S_CD.
                
                # We want to minimize the total score.
                # Delta 1 = Q_new1 - Q_old
                # Delta 2 = Q_new2 - Q_old
                
                delta1 = term_new1 - term_old
                delta2 = term_new2 - term_old
                
                best_delta = min(0, delta1, delta2)
                
                if best_delta < -1e-9: # Significant improvement
                    improved = True
                    if best_delta == delta1:
                        # Apply Move 1: (u1, v1) | (u2, v2)
                        # Swap u2 and v1
                        adj[u].remove(u2)
                        adj[u].append(v1)
                        adj[v].remove(v1)
                        adj[v].append(u2)
                        adj[u2].remove(u)
                        adj[u2].append(v)
                        adj[v1].remove(v)
                        adj[v1].append(u)
                        
                        # Update leaf sets for affected nodes
                        # This is tricky. We need to recompute leaf sets for u, v, u2, v1
                        # Or just recompute all leaf sets? O(N^2) which is fine.
                        leaf_sets = _get_leaf_sets(adj, n)
                        
                    elif best_delta == delta2:
                        # Apply Move 2: (u1, v2) | (u2, v1)
                        # Swap u2 and v2
                        adj[u].remove(u2)
                        adj[u].append(v2)
                        adj[v].remove(v2)
                        adj[v].append(u2)
                        adj[u2].remove(u)
                        adj[u2].append(v)
                        adj[v2].remove(v)
                        adj[v2].append(u)
                        
                        leaf_sets = _get_leaf_sets(adj, n)
                    
                    break # Restart edge loop
        
    return adj, branch_lengths

def _build_newick(adj, n, labels):
    """Build Newick string from adjacency list."""
    def build_rec(node, parent):
        if node < n:
            return labels[node]
        
        parts = []
        for v in adj[node]:
            if v == parent:
                continue
            parts.append(build_rec(v, node))
            
        if not parts:
            return "()"
            
        return "(" + ",".join(parts) + ")"
        
    # Find a root
    root = 0
    # If 0 is a leaf, we need to handle it.
    # The function build_rec works if we start from any node.
    # However, if we start from a leaf, it will treat the leaf as the root of the recursion,
    # which is fine for generating the Newick string as long as we don't add extra parentheses.
    # But standard Newick requires internal nodes to be parenthesized.
    # If we start from a leaf, the leaf is the root.
    # Example: Tree (0, (1, 2)). Start at 0.
    # build_rec(0, -1) -> "0" (if 0 is leaf) -> Wait, if 0 is leaf, it returns "0".
    # But the tree is (0, (1, 2)).
    # If we start at 0, we see neighbor 1 (internal).
    # We need to ensure the structure is correct.
    
    # Better: Pick an internal node as root if possible.
    internal_roots = [i for i in adj if i >= n]
    if internal_roots:
        root = internal_roots[0]
    else:
        root = 0
        
    return build_rec(root, -1) + ";"

def reconstruct(sequences):
    global _START_TIME, _DEADLINE
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    try:
        # 1. Distance Matrix
        D = _jc_distance_matrix(sequences, labels)
        
        # 2. NJ
        adj, branch_lengths = _nj_build(D)
        
        if not adj:
            return "(" + ",".join(labels) + ");"
            
        # 3. NNI Refinement with Quartet Score
        if time.time() < _DEADLINE - 5.0:
            adj, branch_lengths = _nni_refine_quartet(D, adj, branch_lengths, n)
            
        # 4. Build Newick
        newick = _build_newick(adj, n, labels)
        
        return newick
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
