import math
import time
import numpy as np
import sys

# Increase recursion limit just in case, though we use iterative methods
sys.setrecursionlimit(10000)

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    idx_map = {label: i for i, label in enumerate(labels)}
    
    # Convert sequences to numpy arrays of integers 0-3
    # A=0, C=1, G=2, T=3
    seq_arrays = []
    for label in labels:
        seq = sequences[label]
        # Map chars to ints efficiently
        # Using a lookup table or simple replace is faster for small strings
        # But np.frombuffer or manual loop is okay for n<=32, L<=2000
        arr = np.zeros(len(seq), dtype=np.int8)
        for i, c in enumerate(seq):
            if c == 'A': arr[i] = 0
            elif c == 'C': arr[i] = 1
            elif c == 'G': arr[i] = 2
            elif c == 'T': arr[i] = 3
        seq_arrays.append(arr)
    
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            # Hamming distance
            diff = np.sum(seq_arrays[i] != seq_arrays[j])
            L = len(seq_arrays[0])
            p = diff / L
            if p >= 0.7499:
                d = 3.0 # Cap to avoid log(0) or large errors, though JC max is -0.75 log(1-4/3) which is inf, but p<1 usually
                # Actually if p=0.75, 1-4p/3 = 0, log(0) is -inf, so d is inf.
                # In practice, p < 0.75.
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat

def _nj_tree(dist_mat, labels):
    """
    Standard Neighbor Joining.
    Returns a tree structure as a list of tuples (node_a, node_b, len_a, len_b) for internal nodes,
    and a mapping of leaf indices to labels.
    We will represent the tree as a set of edges for the refinement step.
    """
    n = len(labels)
    if n == 1:
        return [(0, 0, 0.0, 0.0)] # Dummy
    if n == 2:
        return [(0, 1, dist_mat[0][1]/2, dist_mat[0][1]/2)]
    if n == 3:
        # 3 taxa tree
        d01 = dist_mat[0][1]
        d02 = dist_mat[0][2]
        d12 = dist_mat[1][2]
        l0 = 0.5 * (d01 + d02 - d12)
        l1 = 0.5 * (d01 + d12 - d02)
        l2 = 0.5 * (d02 + d12 - d01)
        # Structure: (0,1) joined, then joined with 2? 
        # Unrooted 3-taxon tree is a star. 
        # To fit binary structure: ( (0,1), 2 )
        # Internal node 3 connects 0,1. Internal node 4 connects 3,2.
        # This is not quite right for unrooted binary.
        # An unrooted binary tree with 3 leaves is not possible (it's a star).
        # But our internal node structure assumes binary branching.
        # For n=3, we can just return a specific structure.
        # Let's create a dummy internal node.
        # Actually, for n=3, the topology is fixed. 
        # We can represent it as edges: (0,3), (1,3), (2,3) where 3 is central.
        # But our refinement code expects a binary tree structure (2n-3 edges).
        # For n=3, edges = 3. 2*3-3 = 3. Correct.
        # So we need 3 edges.
        # Let's create node 3.
        # Edges: (0,3), (1,3), (2,3).
        # This is not a standard binary tree representation (which usually has internal nodes of degree 3).
        # A star is a valid unrooted tree.
        # Let's handle n=3 separately in the main function or ensure the tree representation supports it.
        # My refinement code will assume a set of edges.
        return None # Handled in main

    # Standard NJ for n >= 4
    nodes = list(range(n))
    # dist is a 2D list indexed by node IDs
    # Node IDs 0..n-1 are leaves. New nodes get IDs n, n+1, ...
    dist = [[0.0 for _ in range(2*n)] for _ in range(2*n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = dist_mat[i][j]
    
    active = list(range(n))
    next_id = n
    internal_edges = [] # List of (u, v, len_u, len_v) where u,v are children of a new node? 
    # Better: Store the tree as a set of edges (u, v, length).
    # NJ creates a new node W connecting A and B.
    # Edges added: (A, W, len_A), (B, W, len_B).
    # But W is a new node.
    # We need to track the topology.
    
    tree_edges = [] # List of (u, v, length)
    
    while len(active) > 3:
        k = len(active)
        # Calculate row sums
        r = [0.0] * k
        for i, node in enumerate(active):
            for j, other in enumerate(active):
                if i != j:
                    r[i] += dist[node][other]
        
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(k):
            for j in range(i + 1, k):
                node_i = active[i]
                node_j = active[j]
                q = (k - 2) * dist[node_i][node_j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        node_a = active[best_i]
        node_b = active[best_j]
        
        # Calculate lengths
        delta = (r[best_i] - r[best_j]) / (k - 2)
        len_a = 0.5 * (dist[node_a][node_b] + delta)
        len_b = 0.5 * (dist[node_a][node_b] - delta)
        
        # Create new node
        new_node = next_id
        next_id += 1
        
        # Add edges
        tree_edges.append((node_a, new_node, len_a))
        tree_edges.append((node_b, new_node, len_b))
        
        # Update distances
        for x in active:
            if x != node_a and x != node_b:
                dist[new_node][x] = 0.5 * (dist[node_a][x] + dist[node_b][x] - dist[node_a][node_b])
                dist[x][new_node] = dist[new_node][x]
        
        # Remove a and b, add new_node
        active = [x for x in active if x != node_a and x != node_b]
        active.append(new_node)
        
    # Handle remaining 3 nodes
    if len(active) == 3:
        u, v, w = active
        # Calculate optimal lengths for the star topology (u,v,w)
        # This is a 3-taxon star.
        # But we need to connect them to the rest of the tree?
        # No, these are the last 3 active nodes.
        # The tree is not complete yet?
        # In NJ, when 3 nodes remain, we form a star among them.
        # However, if there are internal nodes connected to these, the topology is defined.
        # The last 3 nodes in the active list are connected to each other and potentially to the rest of the tree via previous steps?
        # No, the "active" list contains the current representatives of the subtrees.
        # When 3 remain, they form the core of the tree.
        # We need to connect u, v, w.
        # Since they are the last 3, we can form a star with a central node?
        # Or just connect them in a binary way?
        # Standard NJ finishes by connecting the 3 remaining nodes with a star.
        # Let's create a central node C.
        c = next_id
        next_id += 1
        
        d_uv = dist[u][v]
        d_uw = dist[u][w]
        d_vw = dist[v][w]
        
        l_u = 0.5 * (d_uv + d_uw - d_vw)
        l_v = 0.5 * (d_uv + d_vw - d_uw)
        l_w = 0.5 * (d_uw + d_vw - d_uv)
        
        tree_edges.append((u, c, l_u))
        tree_edges.append((v, c, l_v))
        tree_edges.append((w, c, l_w))
        
    return tree_edges

def _calculate_ls_cost_and_lengths(edges, n_leaves, d_obs):
    """
    edges: list of (u, v, length_init)
    n_leaves: number of leaves (0 to n_leaves-1)
    d_obs: n_leaves x n_leaves matrix
    Returns: (cost, lengths_dict)
    """
    # Build adjacency list to find paths
    # For small n, we can just do BFS/DFS for each pair.
    # But we need to build the incidence matrix A for LS.
    # Number of edges E = 2*n_leaves - 3 (for binary unrooted)
    # Number of pairs K = n_leaves*(n_leaves-1)/2
    
    # Identify edges
    E = len(edges)
    # Map edge index to (u, v)
    edge_list = [(u, v) for u, v, _ in edges]
    
    # Build A matrix (K x E)
    # A[k, e] = 1 if edge e is on path between leaf i and leaf j (where k is index of pair i,j)
    
    # To build A, we need to know which edges are on the path between any two leaves.
    # We can do this by running BFS from each leaf to all other leaves.
    
    # Adjacency list: node -> list of (neighbor, edge_index)
    adj = {i: [] for i in range(2 * n_leaves)} # Nodes 0..n_leaves-1 are leaves, others are internal
    # We don't know the max node ID exactly, but it's < 2*n
    # Let's find all nodes involved
    all_nodes = set()
    for u, v, _ in edges:
        all_nodes.add(u)
        all_nodes.add(v)
    
    for node in all_nodes:
        if node not in adj:
            adj[node] = []
            
    for idx, (u, v) in enumerate(edge_list):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # Precompute paths
    # path_edges[i][j] = set of edge indices on path between i and j
    # Or just build A directly.
    
    K = n_leaves * (n_leaves - 1) // 2
    A = np.zeros((K, E), dtype=np.float64)
    
    pair_idx = 0
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            # BFS from i to j
            # Since n is small, simple BFS is fine
            visited = {i}
            queue = [(i, [])]
            found_path = None
            while queue:
                node, path = queue.pop(0)
                if node == j:
                    found_path = path
                    break
                for neighbor, e_idx in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [e_idx]))
            
            if found_path is not None:
                for e_idx in found_path:
                    A[pair_idx, e_idx] = 1.0
            pair_idx += 1
            
    # Solve LS: min || A*l - d_vec ||^2
    d_vec = d_obs[:n_leaves, :n_leaves].copy()
    # Flatten upper triangle
    d_flat = []
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            d_flat.append(d_vec[i, j])
    d_flat = np.array(d_flat)
    
    # Use lstsq
    # A is K x E
    # We want l such that A*l approx d_flat
    # Solution: l = (A^T A)^-1 A^T d
    # numpy.linalg.lstsq(A, d_flat, rcond=None)
    
    try:
        lengths, residuals, rank, s = np.linalg.lstsq(A, d_flat, rcond=None)
        # Ensure lengths are non-negative? LS can give negative lengths.
        # For topology evaluation, negative lengths are bad but we can clip them for the cost calculation?
        # Or just use the raw LS solution for cost.
        # The cost is the residual sum of squares.
        cost = np.sum((A @ lengths - d_flat) ** 2)
    except:
        # Fallback if singular (should not happen for valid tree)
        lengths = np.zeros(E)
        cost = float('inf')
        
    return cost, lengths

def _get_nni_neighbors(edges, n_leaves):
    """
    Given a list of edges (u, v, len), generate NNI neighbors.
    Returns a list of (new_edges_list, edge_index_swapped)
    """
    # Identify internal edges.
    # An edge is internal if it is not incident to a leaf.
    # Leaf nodes are 0..n_leaves-1.
    
    # Build degree map and adjacency
    adj = {}
    for u, v, _ in edges:
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append((v, edges.index((u,v,edges[(u,v,0) if (u,v,0) in edges else (v,u,0)][2]) if (u,v,0) in edges else edges[(v,u,0)][2]))) # This is getting messy.
        
    # Let's use a cleaner representation.
    # edges: list of tuples (u, v, length)
    # We need to find the 4 edges incident to the internal edge (u,v).
    # Let's map node to list of (neighbor, edge_index)
    
    node_to_edges = {}
    for idx, (u, v, _) in enumerate(edges):
        if u not in node_to_edges: node_to_edges[u] = []
        if v not in node_to_edges: node_to_edges[v] = []
        node_to_edges[u].append((v, idx))
        node_to_edges[v].append((u, idx))
        
    neighbors = []
    
    for idx, (u, v, _) in enumerate(edges):
        # Check if u and v are internal
        # A node is a leaf if it is in 0..n_leaves-1
        if u < n_leaves or v < n_leaves:
            continue # Not an internal edge
        
        # Get neighbors of u (excluding v) and v (excluding u)
        u_neighbors = [n for n, e_idx in node_to_edges[u] if n != v]
        v_neighbors = [n for n, e_idx in node_to_edges[v] if n != u]
        
        if len(u_neighbors) != 1 or len(v_neighbors) != 1:
            continue # Should not happen in binary tree
        
        a = u_neighbors[0]
        b = v_neighbors[0]
        
        # The NNI swap replaces (u,v) with (a,b) and connects u-b, v-a?
        # Original: a-u-v-b
        # New: a-u-b-v ? No.
        # NNI on edge (u,v) where u connects to a and v connects to b.
        # The other connections are:
        # u connects to a and (some other node c? No, degree 3).
        # Wait, in an unrooted binary tree, internal nodes have degree 3.
        # So u is connected to a, v, and c.
        # v is connected to b, u, and d.
        # My previous logic `u_neighbors` excluding v gives [a, c].
        # So `len(u_neighbors)` should be 2.
        
        if len(u_neighbors) != 2 or len(v_neighbors) != 2:
            continue
            
        a, c = u_neighbors
        b, d = v_neighbors
        
        # There are 2 possible NNI swaps for the edge (u,v) given the 4 surrounding edges.
        # The "spine" is a-u-v-b. The other edges are c-u and d-v.
        # Swap 1: Remove (u,v), (a,u), (v,b). Add (u,b), (v,a), (c,d)? No.
        # Standard NNI:
        # Topology: ( (a, c) u, (b, d) v ) with edge u-v.
        # Neighbor 1: ( (a, b) x, (c, d) y ) with edge x-y.
        # Neighbor 2: ( (a, d) x, (b, c) y ) with edge x-y.
        
        # Let's construct the new edge lists.
        # We need to identify the specific edges to remove and add.
        
        # Edges to remove:
        # 1. (u, v) [idx]
        # 2. Edge connecting u to a
        # 3. Edge connecting v to b
        
        # Find indices
        def find_edge_idx(n1, n2):
            for i, (eu, ev, _) in enumerate(edges):
                if (eu == n1 and ev == n2) or (eu == n2 and ev == n1):
                    return i
            return -1
            
        idx_uv = idx
        idx_ua = find_edge_idx(u, a)
        idx_vb = find_edge_idx(v, b)
        
        # New edges:
        # 1. (u, b)
        # 2. (v, a)
        # 3. (c, d)
        
        # We need to preserve lengths? No, we will re-optimize lengths.
        # So we just need the topology (the set of edges).
        
        new_edges_set = set()
        for e in edges:
            if e[0] in [u, v] and e[1] in [u, v] and (e[0]==u and e[1]==v or e[0]==v and e[1]==u):
                continue
            if e[0]==u and e[1]==a: continue
            if e[0]==a and e[1]==u: continue
            if e[0]==v and e[1]==b: continue
            if e[0]==b and e[1]==v: continue
            new_edges_set.add((e[0], e[1]))
            
        new_edges_set.add((u, b))
        new_edges_set.add((v, a))
        new_edges_set.add((c, d))
        
        # Convert to list of (u, v, 0.0)
        new_edges_list = [(u, v, 0.0) for u, v in new_edges_set]
        neighbors.append(new_edges_list)
        
        # Second NNI
        # Remove (u,v), (u,a), (v,b) -> Add (u,d), (v,c), (a,b)
        new_edges_set2 = set()
        for e in edges:
            if e[0] in [u, v] and e[1] in [u, v] and (e[0]==u and e[1]==v or e[0]==v and e[1]==u):
                continue
            if e[0]==u and e[1]==a: continue
            if e[0]==a and e[1]==u: continue
            if e[0]==v and e[1]==b: continue
            if e[0]==b and e[1]==v: continue
            new_edges_set2.add((e[0], e[1]))
            
        new_edges_set2.add((u, d))
        new_edges_set2.add((v, c))
        new_edges_set2.add((a, b))
        
        new_edges_list2 = [(u, v, 0.0) for u, v in new_edges_set2]
        neighbors.append(new_edges_list2)
        
    return neighbors

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0 # Safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        else:
            return f"({labels[0]},{labels[1]})"
            
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    # 1. Compute Distance Matrix
    dist_mat = _jc_distance_matrix(sequences, labels)
    
    # 2. Neighbor Joining
    initial_edges = _nj_tree(dist_mat, labels)
    
    if not initial_edges:
        # Fallback star
        return "(" + ",".join(labels) + ")"

    current_edges = initial_edges
    current_cost, _ = _calculate_ls_cost_and_lengths(current_edges, n, dist_mat)
    
    # 3. NNI Refinement
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        neighbors = _get_nni_neighbors(current_edges, n)
        
        best_neighbor = None
        best_neighbor_cost = current_cost
        
        for neighbor_edges in neighbors:
            if (time.time() - start_time) > time_limit:
                break
            cost, _ = _calculate_ls_cost_and_lengths(neighbor_edges, n, dist_mat)
            if cost < best_neighbor_cost:
                best_neighbor_cost = cost
                best_neighbor = neighbor_edges
        
        if best_neighbor is not None:
            current_edges = best_neighbor
            current_cost = best_neighbor_cost
            improved = True
            
    # 4. Convert to Newick
    # Build adjacency list
    adj = {}
    for u, v, _ in current_edges:
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append((v, 0.0)) # Lengths not needed for topology Newick if we just want labels
        adj[v].append((u, 0.0))
        
    # We need to generate a Newick string.
    # We can root the tree arbitrarily (e.g., at leaf 0) and do a DFS.
    # Or just pick any internal node as root.
    
    # Let's find an internal node (degree 3) to root at.
    root = None
    for node in adj:
        if node >= n: # Internal node
            root = node
            break
            
    if root is None:
        # Should not happen for n>3
        root = 0
        
    visited = set()
    
    def get_newick(node, parent):
        visited.add(node)
        if node < n:
            return labels[node]
        
        children = [neighbor for neighbor, _ in adj[node] if neighbor != parent]
        
        if len(children) == 0:
            return labels[node] # Should not happen for internal
        
        child_strs = []
        for child in children:
            if child not in visited:
                child_strs.append(get_newick(child, node))
        
        if len(child_strs) == 1:
            return f"({child_strs[0]})"
        else:
            return "(" + ",".join(child_strs) + ")"
            
    newick = get_newick(root, -1)
    newick += ";"
    
    return newick

# Test/Debug helper (not used in production)
if __name__ == "__main__":
    # Simple test
    seqs = {
        't0': 'ACGTACGTACGT',
        't1': 'ACGTACGTACGT',
        't2': 'ACGTACGTACGT',
        't3': 'ACGTACGTACGT'
    }
    # print(reconstruct(seqs))
    pass
