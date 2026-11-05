"""Reconstruct phylogenetic topology using FastNJ with NNI refinement."""
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 170.0  # Safety margin for 180s limit

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances."""
    n = len(labels)
    # Map labels to indices
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Align sequences
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    # Convert ACGT to integers: A=0, C=1, G=2, T=3
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Create a matrix of shape (n, length)
    try:
        seq_matrix = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    except:
        # Fallback if sequences contain unexpected chars
        seq_matrix = np.array([[0 if c=='A' else 1 if c=='C' else 2 if c=='G' else 3 for c in s] for s in seqs], dtype=np.int8)
        
    # Pairwise hamming distances
    diff = (seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]).sum(axis=-1)
    
    # Convert to JC distance
    p = diff / length
    # Clip p to avoid log(0) or negative
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(d, 0.0)
    
    return d

def _fast_nj(D):
    """
    Fast Neighbor Joining algorithm.
    D: numpy array of distances, shape (n, n)
    Returns: list of joins (i, j, d_i, d_j)
    """
    n = D.shape[0]
    if n <= 2:
        return []
    
    # Active nodes
    active = list(range(n))
    # Distance matrix (we will update this)
    D_cur = D.copy()
    
    joins = []
    # Branch lengths for new nodes
    
    for _ in range(n - 2):
        m = len(active)
        if m == 3:
            break
        
        # Calculate r[i] = sum of distances from i to all other active nodes
        sub_D = D_cur[np.ix_(active, active)]
        r = sub_D.sum(axis=1)
        
        # Calculate q[i, j] = (m-2)*d[i,j] - r[i] - r[j]
        q = (m - 2) * sub_D - r[:, np.newaxis] - r[np.newaxis, :]
        
        # Mask diagonal (i != j)
        np.fill_diagonal(q, np.inf)
        
        # Find min
        idx_flat = np.argmin(q)
        i_idx, j_idx = np.unravel_index(idx_flat, q.shape)
        
        a = active[i_idx]
        b = active[j_idx]
        
        # Calculate branch lengths for a and b
        delta = D_cur[a, b] / 2.0
        r_a = r[i_idx]
        r_b = r[j_idx]
        
        len_a = delta + (r_a - r_b) / (2.0 * (m - 2))
        len_b = delta + (r_b - r_a) / (2.0 * (m - 2))
        
        joins.append((a, b, len_a, len_b))
        
        # Create new node
        new_node = n + len(joins) - 1
        
        # Update distance matrix
        rem_active = [x for x in active if x != a and x != b]
        
        if rem_active:
            da_x = D_cur[a, rem_active]
            db_x = D_cur[b, rem_active]
            ab = D_cur[a, b]
            
            new_d = 0.5 * (da_x + db_x - ab)
            
            D_cur[new_node, rem_active] = new_d
            D_cur[rem_active, new_node] = new_d
        
        # Update active list
        active = rem_active + [new_node]
        
        # Check time
        if time.time() > _DEADLINE - 5.0:
            break
            
    # Construct the tree structure for NNI
    return joins

def _build_topology(joins, n):
    """
    Build a topology representation from NJ joins.
    Returns:
    - adj: dict mapping node -> list of neighbors
    - root: root node index (arbitrary for unrooted)
    """
    adj = {i: [] for i in range(n + len(joins))}
    
    # Initialize leaves
    for i in range(n):
        adj[i] = []
        
    # Process joins
    for k, (a, b, la, lb) in enumerate(joins):
        new_node = n + k
        adj[new_node] = [a, b]
        adj[a].append(new_node)
        adj[b].append(new_node)
        
    # Find root (any node with degree 2 or 3, usually the last created internal node has degree 2 in NJ if we don't close the loop, but NJ creates a binary tree)
    # In a standard NJ, the last step connects two nodes to form a star of 3 if we stop at 3, or connects the last 2 to the existing internal node.
    # The `joins` list has n-2 entries.
    # The last join connects two active nodes. One might be a leaf, one internal, or both internal?
    # Actually, NJ stops when 3 nodes remain. The last join combines 2 of them, leaving 1 internal node connected to the 3rd?
    # No, standard NJ:
    # Iteration 1: join 2 leaves -> new internal node. Active: n-1 nodes.
    # ...
    # Iteration n-2: join 2 nodes from the remaining 3.
    # The remaining 3 nodes are: A, B, C.
    # Join A and B -> new node X.
    # Active: X, C.
    # The loop ends.
    # So we have X connected to A, B. And C is still active.
    # We need to connect C to X?
    # In standard NJ, the last step creates a node X connected to A and B. The remaining active nodes are X and C.
    # The edge between X and C is the last branch.
    # So we must add the edge between the last two active nodes.
    
    # Let's check the active list at the end of _fast_nj.
    # The function returns `joins`. It doesn't return the final active list.
    # We need to infer the final connection.
    # The last join created node `last_node = n + len(joins) - 1`.
    # The nodes involved in the last join were `a` and `b`.
    # The other node in the final set of 3 was `c`.
    # We didn't record `c`.
    
    # Let's re-construct the final edge.
    # The set of all nodes is 0..n+len(joins)-1.
    # All leaves 0..n-1 have degree 1 (connected to their parent).
    # Internal nodes have degree 3, except the "root" of the unrooted tree which effectively has degree 2 in the adjacency list if we don't close it, or we treat the tree as unrooted.
    # In an unrooted binary tree with N leaves, there are N-2 internal nodes.
    # Total nodes = 2N-2.
    # Sum of degrees = 2 * Edges = 2 * (2N-3).
    # Leaves: N * 1 = N.
    # Internal: (N-2) * 3 = 3N-6.
    # Total = 4N-6.
    # 2 * (2N-3) = 4N-6. Matches.
    # So every internal node must have degree 3.
    # In my `adj` construction:
    # When I create `new_node` from `a` and `b`, I set `adj[new_node] = [a, b]` (degree 2).
    # I add `new_node` to `adj[a]` and `adj[b]`.
    # If `a` was a leaf, its degree becomes 1.
    # If `a` was an internal node created previously, its degree was 2 (connected to its two children), now it becomes 3.
    # So, internal nodes created in steps 1 to n-3 will end up with degree 3.
    # The last internal node created (from the last join) will have degree 2 (connected to `a` and `b`).
    # The remaining leaf/internal node `c` from the final set of 3 is not connected to anything yet in my loop!
    # I need to connect the last internal node to the remaining node `c`.
    
    # How to find `c`?
    # `c` is the only node in the final active set that is not `a` or `b` of the last join.
    # But I don't have the active list.
    # However, I can find the node with degree < 3 that is not the last internal node?
    # Actually, all leaves have degree 1.
    # The last internal node `last_int` has degree 2.
    # The node `c` has degree 1 (if it's a leaf) or 2 (if it's an internal node from a previous step? No, if it was internal, it would have been joined already).
    # Wait, in the last step, we join 2 nodes from a set of 3.
    # The 3 nodes are:
    # Case 1: 3 Leaves. Join L1, L2 -> X. X connected to L1, L2. L3 is alone. Connect X-L3.
    # Case 2: 2 Leaves, 1 Internal (X_prev). Join L1, L2 -> X_new. X_new connected to L1, L2. X_prev is alone. Connect X_new-X_prev.
    # Case 3: 1 Leaf, 2 Internal. Join I1, I2 -> X_new. X_new connected to I1, I2. L1 is alone. Connect X_new-L1.
    # In all cases, the "remaining" node `c` has degree 1 (if leaf) or 2 (if internal? No, if it was internal, it was already connected to 2 children. If it wasn't joined, it's still degree 2? No, internal nodes are created by joining. If it wasn't joined, it's a "pending" internal node? No, NJ joins pairs. The nodes in the active list are either leaves or internal nodes that represent clades.
    # An internal node in the active list has already absorbed its children. So it has degree 2 in the "partial tree" sense (connected to its two children).
    # So `c` has degree 2 (if internal) or 1 (if leaf) in the current `adj` structure.
    # The last created node `last_int` has degree 2 (connected to `a` and `b`).
    # We need to find the node `c` that is not `a`, `b`, or `last_int`? No, `c` is the 3rd node.
    # Actually, we can just find any node with degree 1 or 2 that is not part of the last join's immediate children?
    # Simpler: The node `c` is the only node that is not `a`, not `b`, and not `last_int`? No, `a` and `b` are connected to `last_int`.
    # Let's just find the node with the smallest index that has degree < 3 and is not `last_int`?
    # No, `c` is the only node that hasn't been connected to `last_int` yet.
    # We can identify `c` by checking which node in the range 0..last_int has degree 1 (if leaf) or 2 (if internal) and is not `a` or `b`.
    # Wait, `a` and `b` are now degree 2 (if leaf) or 3 (if internal).
    # If `a` was a leaf, it is now degree 1.
    # If `a` was internal, it is now degree 3.
    # So `c` is the only node with degree 1 (if leaf) or 2 (if internal) that is NOT `a` and NOT `b`.
    # But `last_int` is also degree 2.
    # So we look for a node `x` such that `x != a`, `x != b`, `x != last_int`, and `len(adj[x]) < 3`.
    # There should be exactly one such node.
    
    if joins:
        last_int = n + len(joins) - 1
        a, b, _, _ = joins[-1]
        
        c = None
        for x in range(n + len(joins)):
            if x == a or x == b or x == last_int:
                continue
            if len(adj[x]) < 3:
                # It could be a leaf (deg 1) or an internal node that hasn't been fully connected (deg 2)
                # In the final step, only one such node remains.
                c = x
                break
        
        if c is not None:
            adj[last_int].append(c)
            adj[c].append(last_int)
    
    # Find a root (any node)
    root = 0
    return adj, root

def _nni_refine(D, adj, root, labels, n):
    """
    Perform NNI refinement to minimize Sum of Squared Distances (SSD).
    """
    # Precompute clades (sets of leaves) for each node
    clades = {}
    
    def compute_clades(node, visited=None):
        if visited is None:
            visited = set()
        if node in clades:
            return clades[node]
        if node in visited:
            return set()
        visited.add(node)
        
        if node < n:
            clades[node] = {node}
        else:
            # Internal node. Find neighbors.
            # In a binary tree, internal node has 3 neighbors.
            # One neighbor is the "parent" (towards root), two are "children".
            # To compute clade, we need to define direction.
            # Easier: Use DFS/BFS from leaves?
            # Or: For each internal edge, we can compute the split.
            
            # Let's use a different approach:
            # For each internal edge (u, v), the split is defined by the leaves on one side.
            # We can compute the set of leaves reachable from u without going through v.
            
            pass
            
    # This recursive clade computation is tricky with adjacency lists.
    # Alternative: Use the property that NNI only changes the topology locally.
    # We can evaluate the quartet score for each internal edge.
    
    # To get the leaf sets for the 4 subtrees attached to an internal edge (u, v):
    # 1. Identify neighbors of u: u1, u2, v.
    # 2. Identify neighbors of v: v1, v2, u.
    # 3. The leaf set for u1 is all leaves reachable from u1 without passing through u.
    
    # We can precompute these "directional clades" for every directed edge.
    # directed_clade[u][v] = set of leaves reachable from u without passing through v.
    
    directed_clades = {}
    
    def get_directed_clade(u, v):
        key = (u, v)
        if key in directed_clades:
            return directed_clades[key]
        
        # DFS/BFS from u, avoiding v
        stack = [u]
        visited = {v}
        leaves = set()
        
        while stack:
            node = stack.pop()
            if node < n:
                leaves.add(node)
            else:
                for neighbor in adj[node]:
                    if neighbor != v and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        
        directed_clades[key] = leaves
        return leaves

    # Precompute all directed clades
    for u in adj:
        for v in adj[u]:
            get_directed_clade(u, v)

    # NNI Loop
    improved = True
    max_iter = 50
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time.time() > _DEADLINE - 2.0:
            break
            
        # Iterate over all internal edges
        # An internal edge connects two internal nodes.
        # We can iterate over all edges (u, v) where u < v to avoid duplicates.
        # Check if both are internal (>= n).
        
        edges_to_check = []
        for u in adj:
            if u < n:
                continue
            for v in adj[u]:
                if v > u and v >= n:
                    edges_to_check.append((u, v))
                    
        for u, v in edges_to_check:
            # Get neighbors
            u_neighbors = [x for x in adj[u] if x != v]
            v_neighbors = [x for x in adj[v] if x != u]
            
            if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                continue
                
            u1, u2 = u_neighbors
            v1, v2 = v_neighbors
            
            # Get leaf sets
            L_u1 = get_directed_clade(u1, u)
            L_u2 = get_directed_clade(u2, u)
            L_v1 = get_directed_clade(v1, v)
            L_v2 = get_directed_clade(v2, v)
            
            # Calculate scores for the 3 quartets
            # Score is sum of distances within the pairs? No.
            # For SSD, we can approximate by checking if the split is supported.
            # Simple heuristic: Minimize sum of distances across the split?
            # Let's use the Quartet Distance:
            # S1 = d(L_u1, L_u2) + d(L_v1, L_v2)  (Current: (u1,u2)|(v1,v2))
            # S2 = d(L_u1, L_v1) + d(L_u2, L_v2)  (Alt 1: (u1,v1)|(u2,v2))
            # S3 = d(L_u1, L_v2) + d(L_u2, L_v1)  (Alt 2: (u1,v2)|(u2,v1))
            
            def avg_dist(set1, set2):
                if not set1 or not set2:
                    return 0
                s = 0.0
                cnt = 0
                for a in set1:
                    for b in set2:
                        s += D[a, b]
                        cnt += 1
                return s / cnt if cnt > 0 else 0
            
            score_t1 = avg_dist(L_u1, L_u2) + avg_dist(L_v1, L_v2)
            score_t2 = avg_dist(L_u1, L_v1) + avg_dist(L_u2, L_v2)
            score_t3 = avg_dist(L_u1, L_v2) + avg_dist(L_u2, L_v1)
            
            best_score = min(score_t1, score_t2, score_t3)
            
            if best_score < score_t1 - 1e-6:
                # Perform NNI move
                if best_score == score_t2:
                    # Swap u2 and v1
                    # Remove edges (u, u2), (v, v1)
                    # Add edges (u, v1), (v, u2)
                    
                    adj[u].remove(u2)
                    adj[v].remove(v1)
                    adj[u].append(v1)
                    adj[v].append(u2)
                    adj[u2].remove(u)
                    adj[u2].append(v)
                    adj[v1].remove(v)
                    adj[v1].append(u)
                    
                elif best_score == score_t3:
                    # Swap u2 and v2
                    adj[u].remove(u2)
                    adj[v].remove(v2)
                    adj[u].append(v2)
                    adj[v].append(u2)
                    adj[u2].remove(u)
                    adj[u2].append(v)
                    adj[v2].remove(v)
                    adj[v2].append(u)
                    
                # Invalidate directed clades cache
                directed_clades.clear()
                # Re-compute affected clades?
                # For simplicity, clear all and let lazy evaluation handle it?
                # No, `get_directed_clade` is called in the next iteration.
                # We must clear the cache so it recomputes.
                
                improved = True
                break # Restart edge loop

    return adj, root

def _build_newick(adj, root, labels, n):
    """Build Newick string from adjacency list."""
    def build(node, parent):
        if node < n:
            return labels[node]
        
        # Get children (neighbors excluding parent)
        children = [x for x in adj[node] if x != parent]
        
        # In a binary tree, there should be 2 children (if not root) or 2 children (if root, parent is -1)
        # If root, parent is None.
        if parent is None:
            children = adj[node]
        
        if len(children) == 0:
            return labels[node] # Should not happen for internal
        
        # Sort children to ensure deterministic output?
        # children.sort()
        
        parts = [build(child, node) for child in children]
        return "(" + ",".join(parts) + ")"
    
    return build(root, None) + ";"

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
        
        # 2. FastNJ
        joins = _fast_nj(D)
        
        # 3. Build Topology
        if not joins:
             return "(" + ",".join(labels) + ");"
             
        adj, root = _build_topology(joins, n)
        
        # 4. NNI Refinement
        if time.time() < _DEADLINE - 5.0:
            adj, root = _nni_refine(D, adj, root, labels, n)
            
        # 5. Build Newick
        newick = _build_newick(adj, root, labels, n)
        
        return newick
        
    except Exception as e:
        # Fallback: Star tree
        import traceback
        traceback.print_exc()
        return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
