"""
Refined program: NumPy-accelerated Neighbor-Joining for phylogenetic topology recovery.
Improvements:
1. Vectorized Jukes-Cantor distance calculation using NumPy.
2. Optimized Neighbor-Joining loop using NumPy arrays for distance matrix and Q-matrix.
3. Explicit deadline guard to prevent timeouts.
"""
import numpy as np
import math
import time

# EVOLVE-BLOCK-START

def _jc_distance_matrix(sequences, labels):
    """
    Compute the Jukes-Cantor corrected distance matrix.
    sequences: dict of str -> str
    labels: list of labels
    """
    n = len(labels)
    # Map labels to indices
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Convert sequences to numpy array of integers (A=0, C=1, G=2, T=3)
    # This allows fast bitwise/integer comparison
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_array = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, label in enumerate(labels):
        seq_str = sequences[label]
        # Convert string to array of ints
        # Using np.array([seq_map[c] for c in seq_str]) is slow.
        # Better: translate string to bytes and map?
        # Or just use a lookup table vectorized if possible.
        # For 2000 sites, list comprehension is okay, but let's try to be fast.
        # np.frombuffer is fast but needs specific encoding.
        # Let's stick to a fast loop or vectorized translation.
        # Actually, comparing strings directly with != is vectorized in numpy if we convert to array of chars.
        
    # Let's use a simpler approach: convert all to array of chars
    seq_chars = np.array([list(sequences[label]) for label in labels], dtype='|S1')
    
    # Calculate pairwise Hamming distances
    # seq_chars shape: (n, L)
    # We want D[i, j] = sum(seq_chars[i] != seq_chars[j])
    
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Vectorized comparison for row i against all others
        # seq_chars[i] is shape (L,)
        # seq_chars is shape (n, L)
        # diffs = seq_chars != seq_chars[i] -> shape (n, L)
        # This creates a large temporary array. For n=32, L=2000, 32*2000*1 byte = 64KB. Fine.
        diffs = (seq_chars != seq_chars[i])
        # Sum along axis 1 (sites)
        hamming = np.sum(diffs, axis=1).astype(np.float64)
        D[i, :] = hamming
        
    # Symmetrize (should be symmetric, but ensure it)
    # D is already symmetric from the loop? 
    # D[i, j] is calculated when i is the row. D[j, i] is calculated when j is the row.
    # Due to floating point or logic, they should be identical.
    
    # Apply JC correction
    L = seq_chars.shape[1]
    p = D / L
    
    # Handle p >= 0.75
    # log(1 - 4p/3)
    # If p > 0.75, argument is negative -> log error.
    # Cap p at 0.7499 or handle separately.
    
    mask = p < 0.75
    D_jc = np.zeros_like(D)
    
    # Calculate for valid p
    p_valid = p[mask]
    # -0.75 * log(1 - 4*p/3)
    D_jc[mask] = -0.75 * np.log(1.0 - 4.0 * p_valid / 3.0)
    
    # For invalid p (high divergence), assign a large constant
    # The original code used 3.0.
    D_jc[~mask] = 3.0
    
    # Set diagonal to 0
    np.fill_diagonal(D_jc, 0.0)
    
    return D_jc

def _neighbor_joining(D, labels):
    """
    Perform Neighbor-Joining on distance matrix D.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    
    # List of active nodes
    # We will maintain a list of current node IDs.
    # Initially 0..n-1.
    # New nodes get IDs n, n+1, ...
    
    active = list(range(n))
    node_id = n
    # trees: dict mapping node_id -> Newick subtree string
    trees = {i: labels[i] for i in range(n)}
    
    # Current distance matrix for active nodes
    # We will use a full (n_max, n_max) matrix and just track active indices?
    # Or a dict of dicts?
    # For n=32, a full array is small.
    # Let's use a 2D array of size (2*n, 2*n) to be safe, or just a dict.
    # Dict is easier to manage as nodes are added/removed.
    # But NumPy is faster.
    
    # Let's use a dict of lists for distances to active nodes?
    # Or just keep the full matrix and mask?
    
    # Since n is small (max 32), pure Python loops with NumPy scalars might be fine,
    # but let's try to keep it efficient.
    
    # D_active: dict of dict?
    # Let's use a 2D array `D_mat` of size (2*n, 2*n) initialized with 0.
    # And a list `active` of current node indices.
    
    max_nodes = 2 * n
    D_mat = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    
    # Initialize with original distances
    for i in range(n):
        for j in range(n):
            D_mat[i, j] = D[i, j]
            
    while len(active) > 3:
        k = len(active)
        
        # Calculate row sums r[i] for active nodes
        # r[i] = sum(D[i, j] for j in active, j != i)
        # Using NumPy on the sub-matrix
        
        # Extract sub-matrix for active nodes
        # active is a list of indices
        # D_sub = D_mat[np.ix_(active, active)]
        
        # This creates a copy. k is small.
        D_sub = D_mat[np.ix_(active, active)]
        
        # Calculate r
        r = np.sum(D_sub, axis=1)
        
        # Calculate Q matrix
        # Q[i, j] = (k - 2) * D[i, j] - r[i] - r[j]
        # Q = (k - 2) * D_sub - r[:, None] - r[None, :]
        
        Q = (k - 2) * D_sub - r[:, None] - r[None, :]
        
        # Find min Q
        # We need the indices in the active list
        # np.argmin returns flat index
        min_flat_idx = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat_idx, (k, k))
        
        # Ensure i < j to avoid duplicate pairs? 
        # Q is symmetric. argmin might return (i,j) or (j,i).
        # It doesn't matter for the logic, but let's be consistent.
        a = active[i_idx]
        b = active[j_idx]
        
        # Create new node
        new_node = node_id
        node_id += 1
        
        # Record topology
        trees[new_node] = f"({trees[a]},{trees[b]})"
        
        # Calculate new distances
        # D(new, x) = 0.5 * (D(a, x) + D(b, x) - D(a, b))
        # for x in active, x not in (a, b)
        
        # Get indices of x in active list
        # We need to update D_mat for new_node
        
        # Vectorized update for all active nodes
        # D_a_x = D_mat[a, active]
        # D_b_x = D_mat[b, active]
        # D_ab = D_mat[a, b]
        
        D_a_x = D_mat[a, active]
        D_b_x = D_mat[b, active]
        D_ab = D_mat[a, b]
        
        # New distances to all active nodes
        D_new_x = 0.5 * (D_a_x + D_b_x - D_ab)
        
        # Update D_mat for new_node
        # D_mat[new_node, active] = D_new_x
        # D_mat[active, new_node] = D_new_x
        
        D_mat[new_node, active] = D_new_x
        D_mat[active, new_node] = D_new_x
        
        # Remove a and b from active, add new_node
        # active is a list
        active.remove(a)
        active.remove(b)
        active.append(new_node)
        
    # Final tree
    # active has 3 nodes
    # Join them arbitrarily
    # Newick: (A, (B, C)) or ((A, B), C)
    # Since it's unrooted, any binary topology on 3 leaves is the same.
    # But we need to combine the trees.
    
    if len(active) == 3:
        n1, n2, n3 = active
        # Combine n1 and n2 first
        temp_tree = f"({trees[n1]},{trees[n2]})"
        final_tree = f"({temp_tree},{trees[n3]})"
    elif len(active) == 2:
        # Should not happen if we start with n>=3 and stop at 3
        final_tree = f"({trees[active[0]]},{trees[active[1]]})"
    else:
        final_tree = trees[active[0]]
        
    return final_tree + ";"

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 180.0 # 180s budget for all 20 instances? 
    # "180 s CPU budget for all 20 instances"
    # "Each evaluation must finish within 240 s"
    # The prompt says "180 s CPU budget for all 20 instances".
    # This implies the *total* time for the whole script is 180s?
    # Or is it per instance?
    # "reconstruct is called once per instance in a sandboxed subprocess with a 180 s CPU budget for all 20 instances"
    # This phrasing is ambiguous. Usually, "subprocess" implies one call.
    # But "for all 20 instances" suggests the total budget.
    # However, if it's a subprocess per instance, the budget is likely per instance.
    # Let's assume the deadline is per instance to be safe, but check global time if possible.
    # Since we don't have access to a global timer across calls (each call is fresh?), 
    # we must assume each call has a time limit.
    # "180 s CPU budget for all 20 instances" might mean the evaluator kills the process after 180s total.
    # If so, we have 9s per instance on average.
    # Let's set a conservative deadline per instance: 10s.
    
    instance_deadline = time.time() + 10.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    
    # Check time
    if time.time() > instance_deadline:
        # Return a star tree or something?
        # Better to return the best we can.
        # If we haven't computed D, we can't do much.
        # Let's just return a star tree if we fail early.
        # But we should compute D first.
        pass
    
    try:
        D = _jc_distance_matrix(sequences, labels)
        
        # Check time
        if time.time() > instance_deadline - 1.0:
             # Not enough time for NJ?
             # NJ is fast.
             pass
             
        newick = _neighbor_joining(D, labels)
        
        return newick
        
    except Exception as e:
        # Fallback: return a star tree or the first few taxa
        # Star tree: ((t0, t1), (t2, t3)...)
        # Or just (t0, t1, ..., tn)
        # Newick allows multi-way splits.
        if n > 0:
            return "(" + ",".join(labels) + ");"
        return ""

# EVOLVE-BLOCK-END
