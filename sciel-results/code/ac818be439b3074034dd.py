import math
import time
import numpy as np

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    # Convert sequences to numpy arrays of integers 0-3
    # A=0, C=1, G=2, T=3
    seq_arrays = []
    for label in labels:
        seq = sequences[label]
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
            p = diff / len(seq_arrays[0])
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat

def _nj_tree(dist_mat, labels):
    n = len(labels)
    nodes = list(range(n))
    trees = {i: i for i in range(n)}
    
    current_dist = dist_mat.copy()
    
    while len(nodes) > 3:
        k = len(nodes)
        # Calculate row sums
        row_sums = np.zeros(k)
        for i, node in enumerate(nodes):
            for j, other in enumerate(nodes):
                if i != j:
                    row_sums[i] += current_dist[node, other]
        
        best_criterion = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(k):
            for j in range(i + 1, k):
                node_i = nodes[i]
                node_j = nodes[j]
                q = (k - 2) * current_dist[node_i, node_j] - row_sums[i] - row_sums[j]
                if q < best_criterion:
                    best_criterion = q
                    best_i = i
                    best_j = j
        
        node_a = nodes[best_i]
        node_b = nodes[best_j]
        
        delta = (row_sums[best_i] - row_sums[best_j]) / (k - 2)
        len_a = 0.5 * (current_dist[node_a, node_b] + delta)
        len_b = 0.5 * (current_dist[node_a, node_b] - delta)
        
        new_node = len(trees)
        
        trees[new_node] = (node_a, node_b, len_a, len_b)
        
        rest_nodes = [x for idx, x in enumerate(nodes) if idx != best_i and idx != best_j]
        
        # Rebuild distance matrix for active nodes
        active_nodes = rest_nodes + [new_node]
        new_size = len(active_nodes)
        new_dist = np.zeros((new_size, new_size), dtype=np.float64)
        
        # Map old node to new index
        idx_map = {node: i for i, node in enumerate(active_nodes)}
        
        for i, node_i in enumerate(active_nodes):
            for j, node_j in enumerate(active_nodes):
                if i == j:
                    continue
                if node_i == new_node:
                    if node_j == new_node:
                        continue
                    # d(new, x) = 0.5 * (d(a, x) + d(b, x) - d(a, b))
                    # We need the original indices for node_a, node_b, and node_j
                    # node_a and node_b are removed, so we need to look up their distances from current_dist
                    # But we don't have the original indices in the new array.
                    # We need to store the distances before removing.
                    # Let's do this properly.
                    pass
                elif node_j == new_node:
                    pass
                else:
                    # Both are original nodes still in the set
                    # Find their indices in the old current_dist
                    # This is inefficient but correct for small n
                    old_i = nodes.index(node_i) if node_i in nodes else -1
                    old_j = nodes.index(node_j) if node_j in nodes else -1
                    if old_i != -1 and old_j != -1:
                        new_dist[i, j] = current_dist[old_i, old_j]
        
        # This approach is flawed because we lose the distances to node_a and node_b.
        # Let's use a different strategy: maintain a full distance matrix and just mark inactive nodes.
        # Or, simpler: use a dictionary for distances.
        
        # Given the complexity, let's use a pure Python implementation that is known to work.
        pass
    
    # Fallback: return a star tree if NJ fails
    return None

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0 # 5s margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        else:
            return f"({labels[0]},{labels[1]})"

    # 1. Compute Distance Matrix
    dist_mat = _jc_distance_matrix(sequences, labels)
    
    # 2. Neighbor Joining (Robust Implementation)
    # Use a list of active nodes and a dictionary for distances to handle dynamic removals.
    # Or, more simply, use the standard NJ algorithm with a 2D array and careful index management.
    
    # Let's use a standard, clean NJ implementation.
    # Source: Standard NJ algorithm
    num_nodes = n
    # D is a 2D array of size (2n) x (2n) to accommodate new nodes
    # Initialize with 0
    D = np.zeros((2 * n, 2 * n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            D[i, j] = dist_mat[i, j]
    
    # Active nodes list
    active = list(range(n))
    
    # Tree structure: list of tuples (parent, child1, child2, len1, len2)
    # Or dictionary: parent -> (child1, child2, len1, len2)
    tree_edges = {}
    next_id = n
    
    while len(active) > 3:
        k = len(active)
        
        # Calculate row sums
        r = np.zeros(k)
        for i, node in enumerate(active):
            for j, other in enumerate(active):
                if i != j:
                    r[i] += D[node, other]
        
        # Find best pair
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(k):
            for j in range(i + 1, k):
                node_i = active[i]
                node_j = active[j]
                q = (k - 2) * D[node_i, node_j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        node_a = active[best_i]
        node_b = active[best_j]
        
        # Calculate lengths
        delta = (r[best_i] - r[best_j]) / (k - 2)
        len_a = 0.5 * (D[node_a, node_b] + delta)
        len_b = 0.5 * (D[node_a, node_b] - delta)
        
        # Create new node
        new_node = next_id
        next_id += 1
        
        tree_edges[new_node] = (node_a, node_b, len_a, len_b)
        
        # Update distances
        # d(new, x) = 0.5 * (d(a, x) + d(b, x) - d(a, b))
        for x in active:
            if x != node_a and x != node_b:
                D[new_node, x] = 0.5 * (D[node_a, x] + D[node_b, x] - D[node_a, node_b])
                D[x, new_node] = D[new_node, x]
        
        # Remove a and b, add new_node
        active = [x for x in active if x != node_a and x != node_b]
        active.append(new_node)
        
        # Time check
        if time.time() - start_time > time_limit * 0.5:
            break

    # Build final tree structure
    def get_newick(node):
        if node < n:
            return labels[node]
        else:
            if node not in tree_edges:
                return labels[node] # Fallback
            child_a, child_b, len_a, len_b = tree_edges[node]
            return f"({get_newick(child_a)}:{len_a:.4f},{get_newick(child_b)}:{len_b:.4f})"
    
    if len(active) == 3:
        u, v, w = active
        len_u = 0.5 * (D[u, v] + D[u, w] - D[v, w])
        len_v = 0.5 * (D[u, v] + D[v, w] - D[u, w])
        len_w = 0.5 * (D[u, w] + D[v, w] - D[u, v])
        
        new_node_1 = next_id
        next_id += 1
        tree_edges[new_node_1] = (u, v, len_u, len_v)
        
        new_node_2 = next_id
        next_id += 1
        # Estimate length for the edge connecting new_node_1 and w
        # In a 3-taxon tree, the central node has 3 edges.
        # In a binary tree representation, we join (u,v) with w.
        # The length of the edge from the (u,v) clade to w is 0 in the binary representation?
        # No, the distance from the new_node_1 to w is D[new_node_1, w] which we didn't calculate.
        # Let's estimate it. The "central" node is the MRCA of u, v, w.
        # Let L_u, L_v, L_w be the lengths from the central node to u, v, w.
        # L_u = 0.5 * (D[u, v] + D[u, w] - D[v, w])
        # L_v = 0.5 * (D[u, v] + D[v, w] - D[u, w])
        # L_w = 0.5 * (D[u, w] + D[v, w] - D[u, v])
        # The binary tree is ((u, v), w).
        # The node new_node_1 represents the clade (u, v).
        # The edge from new_node_1 to w has length 0? No.
        # The distance from new_node_1 to w is L_w + (length of internal edge)?
        # In NJ, the last step just connects the 3 nodes.
        # Let's just use 0 for the internal edge length in the binary representation,
        # as the topology is what matters for RF distance.
        tree_edges[new_node_2] = (new_node_1, w, 0.0, len_w)
        
        newick = get_newick(new_node_2)
    else:
        # Should not happen
        newick = ",".join(labels)

    # Check time
    if time.time() - start_time > time_limit:
        return newick

    return newick
