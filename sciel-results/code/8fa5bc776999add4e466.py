import time
import numpy as np

_DEADLINE = 265.0  # seconds; leave 5s margin before the 270s kill
_START = time.time()

def _jc_distances(sequences, labels):
    """Compute pairwise JC-corrected distances using NumPy."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    if length == 0:
        return np.zeros((n, n))

    # Map DNA to integers for faster comparison? 
    # Or just use string comparison which is optimized in C.
    # np.array of chars is fine.
    dna_arr = np.array([list(s) for s in seqs])
    
    # Compute pairwise Hamming distances
    mismatches = np.sum(dna_arr[:, None, :] != dna_arr[None, :, :], axis=2)
    
    # Proportion of differences
    p = mismatches.astype(float) / length
    
    # JC correction: -0.75 * ln(1 - 4p/3)
    p_clipped = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(dist, 0.0)
    
    return dist

def _nj_tree(dist_mat, labels):
    """
    Neighbor Joining.
    Returns a dictionary representing the tree:
    nodes: list of active node IDs
    trees: dict mapping node ID to (child1, child2, len1, len2) for internal nodes,
           or the leaf label for leaves.
    Actually, let's return the adjacency list directly.
    """
    n = len(labels)
    if n <= 2:
        # Handle trivial cases
        if n == 1:
            return {0: []} # Leaf 0, no edges
        else:
            return {0: [(1, 0.0)], 1: [(0, 0.0)]}
    
    D = dist_mat.copy()
    # trees[i] will store the structure. 
    # For leaves, trees[i] = i (leaf id)
    # For internal nodes, trees[i] = (child1, child2, len1, len2)
    trees = {i: i for i in range(n)}
    
    # Active nodes
    active = list(range(n))
    m = n
    
    # We need to track the actual IDs of the nodes in the final tree.
    # Leaves are 0..n-1.
    # Internal nodes will be n, n+1, ...
    next_id = n
    
    # We will maintain the adjacency list at the end.
    # But NJ builds the tree hierarchically.
    # Let's build the `trees` dict and then convert to adjacency list.
    
    while m > 3:
        if time.time() > _DEADLINE:
            break
            
        # Compute row sums
        r = np.sum(D, axis=1)
        
        # Compute Q matrix
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find min Q
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        # Map indices in D to actual node IDs
        node_i = active[i]
        node_j = active[j]
        
        d_ij = D[i, j]
        
        # Calculate branch lengths
        # len_i = 0.5 * (d_ij + (r[i] - r[j]) / (m - 2))
        # len_j = 0.5 * (d_ij - (r[i] - r[j]) / (m - 2))
        delta = (r[i] - r[j]) / (m - 2)
        len_i = 0.5 * (d_ij + delta)
        len_j = 0.5 * (d_ij - delta)
        
        # Ensure non-negative? NJ can produce negative lengths. 
        # For topology, we can clamp to 0 or keep as is. 
        # Keeping as is is better for distance calculation? 
        # Usually clamping is safer for validity, but for NNI, 
        # the topology is what matters.
        if len_i < 0: len_i = 0.0
        if len_j < 0: len_j = 0.0
        
        # Create new node
        new_node_id = next_id
        next_id += 1
        
        # Store structure
        trees[new_node_id] = (node_i, node_j, len_i, len_j)
        
        # Update active list and D
        others_idx = [k for k in range(m) if k != i and k != j]
        others_nodes = [active[k] for k in others_idx]
        
        # New distances from new_node to others
        d_new_others = np.zeros(len(others_nodes))
        for idx, k in enumerate(others_idx):
            node_k = active[k]
            d_new_others[idx] = 0.5 * (D[i, k] + D[j, k] - d_ij)
            
        # Construct new D
        new_m = len(others_nodes) + 1
        D_new = np.zeros((new_m, new_m))
        
        if others_nodes:
            # Fill distances among others
            D_new[:len(others_nodes), :len(others_nodes)] = D[np.ix_(others_idx, others_idx)]
            
            # Fill distances from new node to others
            D_new[:len(others_nodes), len(others_nodes)] = d_new_others
            D_new[len(others_nodes), :len(others_nodes)] = d_new_others
            
            # Update active
            new_active = others_nodes + [new_node_id]
        else:
            new_active = [new_node_id]
            
        D = D_new
        active = new_active
        m = new_m
        
    # Handle final 3 nodes
    # We have 3 nodes in 'active'.
    # They form a star.
    # The center of the star is a new internal node?
    # Or we can just connect them in a triangle? No, unrooted tree with 3 leaves is a star.
    # The center is an internal node.
    if m == 3:
        node_1, node_2, node_3 = active
        # Create center node
        center_id = next_id
        next_id += 1
        
        # Calculate branch lengths to center
        # d(1,2) = L1 + L2
        # d(2,3) = L2 + L3
        # d(1,3) = L1 + L3
        d12 = D[0, 1]
        d23 = D[1, 2]
        d13 = D[0, 2]
        
        L1 = 0.5 * (d12 + d13 - d23)
        L2 = 0.5 * (d12 + d23 - d13)
        L3 = 0.5 * (d13 + d23 - d12)
        
        if L1 < 0: L1 = 0.0
        if L2 < 0: L2 = 0.0
        if L3 < 0: L3 = 0.0
        
        # Store structure for center
        # The center connects to node_1, node_2, node_3.
        # But our `trees` dict expects pairs.
        # We can represent the star as two internal nodes?
        # Or just handle the conversion to adjacency list specially.
        # Let's add the center to trees as a special case or just build adjacency directly.
        
        # For simplicity, let's just build the adjacency list from the `trees` dict and the final star.
        pass
        
    # Convert `trees` to adjacency list
    # `trees` contains entries for internal nodes created during NJ.
    # Leaves are 0..n-1.
    # Internal nodes are n..next_id-1.
    
    adj = {i: [] for i in range(next_id)}
    
    # Process internal nodes
    for node_id in range(n, next_id):
        if node_id in trees:
            c1, c2, l1, l2 = trees[node_id]
            adj[node_id].append((c1, l1))
            adj[node_id].append((c2, l2))
            adj[c1].append((node_id, l1))
            adj[c2].append((node_id, l2))
            
    # Handle the final 3 nodes star
    if m == 3:
        # The center node was created above if we implemented it.
        # In the loop above, if m becomes 3, we exit.
        # The 3 nodes are in `active`.
        # We need to connect them.
        # Let's add the center node logic here properly.
        
        # We didn't add the center to `trees` in the loop above.
        # Let's add it now.
        center_id = next_id - 1 # It was incremented in the loop? No, the loop stops at m=3.
        # Actually, the loop runs while m > 3.
        # So if we start with m=4, we join 2, m becomes 3.
        # The new node is in `trees`.
        # The 3 active nodes are: the new node, and the other 2?
        # No.
        # If m=4, active = [a, b, c, d].
        # Join a, b -> new_node.
        # active becomes [c, d, new_node].
        # m becomes 3.
        # Loop ends.
        # So `active` has 3 nodes.
        # Two of them might be leaves, one is internal (new_node).
        # Or all 3 could be internal? No, we start with leaves.
        
        # The 3 nodes in `active` need to be connected in a star.
        # We create a center node.
        center_id = next_id
        next_id += 1
        adj[center_id] = []
        
        node_1, node_2, node_3 = active
        d12 = D[0, 1]
        d23 = D[1, 2]
        d13 = D[0, 2]
        
        L1 = 0.5 * (d12 + d13 - d23)
        L2 = 0.5 * (d12 + d23 - d13)
        L3 = 0.5 * (d13 + d23 - d12)
        
        if L1 < 0: L1 = 0.0
        if L2 < 0: L2 = 0.0
        if L3 < 0: L3 = 0.0
        
        adj[center_id].append((node_1, L1))
        adj[center_id].append((node_2, L2))
        adj[center_id].append((node_3, L3))
        
        adj[node_1].append((center_id, L1))
        adj[node_2].append((center_id, L2))
        adj[node_3].append((center_id, L3))
        
        # Update next_id to include center
        # (It's already incremented)
        
    # Ensure all nodes are in adj
    for i in range(next_id):
        if i not in adj:
            adj[i] = []
            
    return adj

def _tree_distances(adj, n_leaves):
    """Compute pairwise distances between all leaves."""
    dists = np.zeros((n_leaves, n_leaves))
    # For each leaf, BFS to all other leaves
    for i in range(n_leaves):
        # BFS from i
        visited = {i: 0.0}
        stack = [(i, 0.0)]
        while stack:
            node, dist = stack.pop()
            for neighbor, length in adj[node]:
                if neighbor not in visited:
                    visited[neighbor] = dist + length
                    stack.append((neighbor, dist + length))
        for j in range(n_leaves):
            if j in visited:
                dists[i, j] = visited[j]
    return dists

def _sse(dist_obs, dist_tree):
    """Sum of squared errors."""
    # Only consider upper triangle to avoid double counting? 
    # SSE is usually sum over all i,j.
    # Since dist is symmetric, sum over i < j is half.
    # Let's just sum all.
    return np.sum((dist_obs - dist_tree) ** 2)

def _nni_refine(adj, n_leaves, dist_obs, time_limit):
    """Perform NNI refinement."""
    # Identify internal edges
    # An edge (u, v) is internal if u >= n_leaves and v >= n_leaves
    # Or if we consider the center of the star?
    # The center node is internal.
    
    # We need a list of internal edges.
    # Since the tree is unrooted, each edge is counted twice in adj.
    # We can iterate u < v to avoid duplicates.
    
    # Current SSE
    dist_tree = _tree_distances(adj, n_leaves)
    current_sse = _sse(dist_obs, dist_tree)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Get list of internal edges
        internal_edges = []
        for u in range(n_leaves, len(adj)):
            for v, _ in adj[u]:
                if v > u and v >= n_leaves:
                    internal_edges.append((u, v))
                    
        for u, v in internal_edges:
            # Check if u, v are still connected?
            # NNI modifies the tree, so the edge list might become invalid.
            # It's safer to regenerate internal edges in each iteration?
            # Or just check if the edge exists.
            
            # Find neighbors of u and v
            # u neighbors: v, a, b
            # v neighbors: u, c, d
            
            u_neighbors = [n for n, _ in adj[u] if n != v]
            v_neighbors = [n for n, _ in adj[v] if n != u]
            
            if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                continue # Not a valid internal edge for NNI (degree must be 3)
                
            a, b = u_neighbors
            c, d = v_neighbors
            
            # Get lengths
            len_u_a = next(l for n, l in adj[u] if n == a)
            len_u_b = next(l for n, l in adj[u] if n == b)
            len_v_c = next(l for n, l in adj[v] if n == c)
            len_v_d = next(l for n, l in adj[v] if n == d)
            len_u_v = next(l for n, l in adj[u] if n == v)
            
            # Original SSE is known.
            
            # Try NNI 1: ((a, c), (b, d))
            # New nodes u', v'
            # u' connects to a, c, v'
            # v' connects to b, d, u'
            
            # We need to calculate the new distances.
            # Instead of modifying adj and recalculating, 
            # we can just calculate the change in SSE for the affected pairs?
            # No, let's just modify a copy of adj and recalculate.
            # It's safer and N=32 is small.
            
            adj_copy = {k: list(v) for k, v in adj.items()}
            
            # Remove old edges
            adj_copy[u] = [e for e in adj_copy[u] if e[0] not in [v, a, b]]
            adj_copy[v] = [e for e in adj_copy[v] if e[0] not in [u, c, d]]
            adj_copy[a] = [e for e in adj_copy[a] if e[0] != u]
            adj_copy[b] = [e for e in adj_copy[b] if e[0] != u]
            adj_copy[c] = [e for e in adj_copy[c] if e[0] != v]
            adj_copy[d] = [e for e in adj_copy[d] if e[0] != v]
            
            # Create new nodes
            new_u = max(adj_copy.keys()) + 1
            new_v = max(adj_copy.keys()) + 1
            
            adj_copy[new_u] = []
            adj_copy[new_v] = []
            
            # Connect new_u to a, c, new_v
            # We need to estimate new branch lengths.
            # Let's keep the old lengths as a first approximation?
            # Or calculate optimal lengths for the 4 subtrees?
            # For speed, let's just use 0 or average?
            # Actually, keeping the old lengths is a common heuristic.
            # len(a, new_u) = len_u_a
            # len(c, new_u) = len_v_c
            # len(b, new_v) = len_u_b
            # len(d, new_v) = len_v_d
            # len(new_u, new_v) = len_u_v
            
            adj_copy[new_u].append((a, len_u_a))
            adj_copy[new_u].append((c, len_v_c))
            adj_copy[new_u].append((new_v, len_u_v))
            
            adj_copy[new_v].append((b, len_u_b))
            adj_copy[new_v].append((d, len_v_d))
            adj_copy[new_v].append((new_u, len_u_v))
            
            adj_copy[a].append((new_u, len_u_a))
            adj_copy[c].append((new_u, len_v_c))
            adj_copy[b].append((new_v, len_u_b))
            adj_copy[d].append((new_v, len_v_d))
            
            # Calculate SSE
            dist_tree_new = _tree_distances(adj_copy, n_leaves)
            sse_new = _sse(dist_obs, dist_tree_new)
            
            if sse_new < current_sse:
                # Accept move
                adj = adj_copy
                current_sse = sse_new
                improved = True
                # Break to restart edge iteration? 
                # Or continue?
                # If we accept, the tree changed, so other edges are invalid.
                # It's safer to restart the loop.
                break
                
            # Try NNI 2: ((a, d), (b, c))
            adj_copy = {k: list(v) for k, v in adj.items()}
            
            # Remove old edges
            adj_copy[u] = [e for e in adj_copy[u] if e[0] not in [v, a, b]]
            adj_copy[v] = [e for e in adj_copy[v] if e[0] not in [u, c, d]]
            adj_copy[a] = [e for e in adj_copy[a] if e[0] != u]
            adj_copy[b] = [e for e in adj_copy[b] if e[0] != u]
            adj_copy[c] = [e for e in adj_copy[c] if e[0] != v]
            adj_copy[d] = [e for e in adj_copy[d] if e[0] != v]
            
            new_u = max(adj_copy.keys()) + 1
            new_v = max(adj_copy.keys()) + 1
            
            adj_copy[new_u] = []
            adj_copy[new_v] = []
            
            # Connect new_u to a, d, new_v
            adj_copy[new_u].append((a, len_u_a))
            adj_copy[new_u].append((d, len_v_d))
            adj_copy[new_u].append((new_v, len_u_v))
            
            adj_copy[new_v].append((b, len_u_b))
            adj_copy[new_v].append((c, len_v_c))
            adj_copy[new_v].append((new_u, len_u_v))
            
            adj_copy[a].append((new_u, len_u_a))
            adj_copy[d].append((new_u, len_v_d))
            adj_copy[b].append((new_v, len_u_b))
            adj_copy[c].append((new_v, len_v_c))
            
            dist_tree_new = _tree_distances(adj_copy, n_leaves)
            sse_new = _sse(dist_obs, dist_tree_new)
            
            if sse_new < current_sse:
                adj = adj_copy
                current_sse = sse_new
                improved = True
                break
                
    return adj

def _to_newick(adj, n_leaves):
    """Convert adjacency list to Newick string."""
    # Find a root
    # Pick any node
    root = 0
    # Build parent pointers via DFS
    parent = {root: -1}
    stack = [root]
    visited = {root}
    
    while stack:
        node = stack.pop()
        for neighbor, _ in adj[node]:
            if neighbor not in visited:
                visited[neighbor] = True
                parent[neighbor] = node
                stack.append(neighbor)
                
    # Build Newick
    def build(node):
        if node < n_leaves:
            return f"t{node}"
        children = [n for n, _ in adj[node] if n != parent[node]]
        if not children:
            return f"t{node}" # Should not happen for internal
        child_strs = [build(c) for c in children]
        return "(" + ",".join(child_strs) + ")"
        
    return build(root) + ";"

def reconstruct(sequences):
    global _START, _DEADLINE
    
    _START = time.time()
    _DEADLINE = _START + 265.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"t0;"
        else:
            return f"(t0,t1);"
    
    # Compute distance matrix
    D = _jc_distances(sequences, labels)
    
    # Run NJ
    adj = _nj_tree(D, labels)
    
    # NNI Refinement
    adj = _nni_refine(adj, n, D, _DEADLINE - 5.0) # Leave 5s margin
    
    # Convert to Newick
    newick = _to_newick(adj, n)
    
    return newick
