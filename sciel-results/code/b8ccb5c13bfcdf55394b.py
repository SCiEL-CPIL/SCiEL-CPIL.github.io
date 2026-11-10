import math
import time
import numpy as np

def _jc_distance_matrix(sequences, labels):
    """
    Compute the Jukes-Cantor corrected distance matrix.
    Uses vectorized NumPy operations for speed.
    """
    n = len(labels)
    # Map ACGT to integers
    # A=0, C=1, G=2, T=3
    seq_arrays = []
    for label in labels:
        seq = sequences[label]
        # Use a lookup table for faster conversion if needed, but direct comparison is often fine for small n
        # For speed, we can convert the whole sequence to an array of bytes or integers
        # Let's use a simple loop as it's O(L) and L is max 2000, n is max 32. 
        # Vectorizing the comparison between all pairs is O(n^2 * L).
        # 32^2 * 2000 = 2M operations. This is fast in NumPy.
        
        # Convert string to numpy array of integers
        # 'A' -> 0, 'C' -> 1, 'G' -> 2, 'T' -> 3
        # We can use np.frombuffer with a translation table, or just loop.
        # Given n is small, loop is acceptable, but let's try to be efficient.
        arr = np.array([ord(c) for c in seq], dtype=np.uint8)
        # Map ord values to 0-3
        # A=65, C=67, G=71, T=84
        # This is a bit slow. Let's use a dictionary lookup or a precomputed table.
        # Better: Convert to bytes and use a translation table.
        trans_table = bytes.maketrans(b'ACGT', b'\x00\x01\x02\x03')
        arr = seq.encode('utf-8').translate(trans_table)
        arr = np.frombuffer(arr, dtype=np.uint8)
        seq_arrays.append(arr)
    
    # Stack into a single matrix (n, L)
    seq_matrix = np.vstack(seq_arrays)
    L = seq_matrix.shape[1]
    
    # Compute Hamming distances
    # diff[i, j] is the number of positions where seq_i and seq_j differ
    # We can compute this by broadcasting
    # seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]
    # This creates a (n, n, L) boolean array.
    # For n=32, L=2000, this is 32*32*2000 = 2M booleans. 2MB. Very manageable.
    
    diffs = np.sum(seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :], axis=2)
    
    # Convert to proportions
    p = diffs / L
    
    # Apply JC correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 (saturation)
    dist_mat = np.zeros((n, n), dtype=np.float64)
    
    # Mask for valid p
    valid_mask = p < 0.7499
    dist_mat[valid_mask] = -0.75 * np.log(1.0 - 4.0 * p[valid_mask] / 3.0)
    
    # For saturated distances, set to a large value or 3.0 (max possible for 4 bases is 3.0? 
    # Actually, -0.75 * ln(1 - 4*0.75/3) = -0.75 * ln(0) = inf. 
    # The max finite value before log(0) is when p is close to 0.75.
    # Typically, we cap it or set to a high value. 3.0 is a common heuristic cap, but mathematically it goes to infinity.
    # For NJ, large distances are fine, they just won't be joined first.
    # Let's set them to a large value, e.g., 10.0 or just leave as inf? 
    # Inf might cause issues in subtraction. Let's cap at a high value like 100.0 or just use 3.0 as a proxy for "very distant".
    # Actually, the formula breaks down. Let's just set to 0.0? No, that's wrong.
    # Let's set to a large constant, e.g., 10.0.
    dist_mat[~valid_mask] = 10.0
    
    # Diagonal is 0
    np.fill_diagonal(dist_mat, 0.0)
    
    return dist_mat

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0 # 5s margin from 270s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        else:
            return f"({labels[0]},{labels[1]})"

    # 1. Compute Distance Matrix
    dist_mat = _jc_distance_matrix(sequences, labels)
    
    # 2. Neighbor Joining
    # We maintain a list of active node indices (original indices)
    # And a distance matrix D of size (active_count, active_count)
    # Initially, active = [0, 1, ..., n-1], D = dist_mat
    
    active_nodes = list(range(n))
    D = dist_mat.copy()
    
    # Tree structure storage
    # We will store the Newick string for each node as we merge.
    # newick[i] will be the Newick representation of the clade rooted at node i (if it's a leaf, just the label)
    # For internal nodes, we create a new index.
    
    # Since we are reducing the matrix, we need to keep track of which original nodes form which clades.
    # Let's use a dictionary: node_id -> (child1_id, child2_id, len1, len2)
    # But we also need to generate the Newick string.
    # It's easier to build the tree structure as we go and then convert to Newick at the end.
    # Or, build Newick strings incrementally.
    
    # Let's use a list `newicks` where index corresponds to the current "node id" in the reduced matrix.
    # However, the node ids change as we reduce the matrix.
    # Better approach:
    # Keep a list `active` of original node IDs.
    # Keep a dictionary `tree_structure` mapping a unique ID for each clade to its children.
    # Actually, the standard way is to assign new IDs to merged nodes.
    
    next_id = n
    # Map from current index in D to the "ID" of the node (which is either an original index or a new_id)
    # Initially, id_map[i] = i
    id_map = list(range(n))
    
    # newick_strings[id] = the Newick string for that node
    newick_strings = {i: labels[i] for i in range(n)}
    
    # tree_edges[id] = (child1_id, child2_id, len1, len2)
    tree_edges = {}
    
    current_size = n
    
    while current_size > 3:
        if time.time() - start_time > time_limit:
            # Time out: return a star tree or whatever we have
            # For now, break and handle final assembly with what we have
            break
            
        k = current_size
        
        # Calculate row sums r
        # r[i] = sum(D[i, j] for j != i)
        # D is (k, k)
        # r = D.sum(axis=1) - np.diag(D)
        r = np.sum(D, axis=1)
        
        # Calculate Q matrix
        # Q[i, j] = (k - 2) * D[i, j] - r[i] - r[j]
        # We only need the lower triangle (i < j)
        
        # Vectorized calculation:
        # Q = (k-2)*D - r[:, np.newaxis] - r[np.newaxis, :]
        Q = (k - 2) * D - r[:, np.newaxis] - r[np.newaxis, :]
        
        # Find the minimum Q in the lower triangle
        # We need to ignore the diagonal and upper triangle
        # Create a mask for lower triangle
        mask = np.triu(np.ones((k, k), dtype=bool), k=1)
        Q_masked = np.where(mask, Q, np.inf)
        
        min_val = np.min(Q_masked)
        # Get indices
        min_idx = np.unravel_index(np.argmin(Q_masked), Q_masked.shape)
        i, j = min_idx
        
        # Ensure i < j
        if i > j:
            i, j = j, i
            
        node_a_id = id_map[i]
        node_b_id = id_map[j]
        
        # Calculate branch lengths
        # delta = (r[i] - r[j]) / (k - 2)
        delta = (r[i] - r[j]) / (k - 2)
        len_a = 0.5 * (D[i, j] + delta)
        len_b = 0.5 * (D[i, j] - delta)
        
        # Clamp lengths to 0 if negative (due to numerical errors or non-additive distances)
        if len_a < 0: len_a = 0.0
        if len_b < 0: len_b = 0.0
        
        # Create new node
        new_node_id = next_id
        next_id += 1
        
        # Store tree structure
        tree_edges[new_node_id] = (node_a_id, node_b_id, len_a, len_b)
        newick_strings[new_node_id] = f"({newick_strings[node_a_id]},{newick_strings[node_b_id]})"
        
        # Update distance matrix
        # New distances: d(new, x) = 0.5 * (d(a, x) + d(b, x) - d(a, b))
        # We need to remove rows/cols i and j, and add a new row/col for the new node.
        
        # Indices of nodes to keep
        keep_indices = [idx for idx in range(k) if idx != i and idx != j]
        
        # New size
        new_k = k - 1
        
        # Construct new D matrix
        new_D = np.zeros((new_k, new_k), dtype=np.float64)
        
        # Fill the part corresponding to existing nodes
        # We need to map old indices to new indices
        # The new matrix will have the kept nodes in the first new_k-1 positions, and the new node in the last position.
        
        # Let's just rebuild it for clarity, as k is small
        for new_i, old_i in enumerate(keep_indices):
            for new_j, old_j in enumerate(keep_indices):
                new_D[new_i, new_j] = D[old_i, old_j]
                
        # Calculate distances from new node to kept nodes
        # The new node is at index new_k - 1 in the new matrix
        new_idx = new_k - 1
        for new_i, old_i in enumerate(keep_indices):
            d_new_old = 0.5 * (D[i, old_i] + D[j, old_i] - D[i, j])
            if d_new_old < 0: d_new_old = 0.0
            new_D[new_idx, new_i] = d_new_old
            new_D[new_i, new_idx] = d_new_old
            
        # Update active nodes list and id_map
        # The new active nodes are the kept ones plus the new node
        new_active_ids = [id_map[old_idx] for old_idx in keep_indices] + [new_node_id]
        
        D = new_D
        id_map = new_active_ids
        current_size = new_k

    # Final assembly
    # If we have 3 nodes left, they form a trifurcation (star tree) in unrooted sense, 
    # but NJ usually outputs a binary tree. 
    # For 3 taxa, the unrooted tree is a star. 
    # The splits are {a}|{b,c}, {b}|{a,c}, {c}|{a,b}.
    # A Newick string for a 3-taxon unrooted tree is often written as (a,b,c);
    # However, standard Newick parsers might expect binary. 
    # The task says "unrooted tree topology". 
    # If we return (a,b,c);, it represents a trifurcation.
    # If we return ((a,b),c);, it represents a binary tree with a zero-length branch potentially?
    # Actually, for 3 taxa, any binary tree has one internal branch of length 0 if the distances are additive? No.
    # In a 3-taxon tree, there are no internal branches in the unrooted sense, just the central node.
    # So (a,b,c); is the correct representation of the topology.
    
    if current_size == 3:
        # Get the newick strings for the 3 remaining nodes
        n1 = newick_strings[id_map[0]]
        n2 = newick_strings[id_map[1]]
        n3 = newick_strings[id_map[2]]
        return f"({n1},{n2},{n3});"
    elif current_size == 2:
        # Should not happen if n>=3 and loop runs to 3
        n1 = newick_strings[id_map[0]]
        n2 = newick_strings[id_map[1]]
        return f"({n1},{n2});"
    elif current_size == 1:
        return f"{newick_strings[id_map[0]]};"
    else:
        # Fallback: Star tree
        return "(" + ",".join(newick_strings[id] for id in id_map) + ");"
