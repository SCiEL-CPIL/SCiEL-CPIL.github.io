"""
Phylogenetic topology recovery using optimized Neighbor Joining.
Combines vectorized distance calculation with efficient NJ algorithm.
"""
import time
import math
import numpy as np

# Global deadline: 270s wall-clock limit, use 250s as safety margin
_DEADLINE = time.time() + 250.0

def _time_exceeded():
    return time.time() > _DEADLINE

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distance matrix using vectorized NumPy."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    
    # Get sequences in label order
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0]) if seqs else 0
    
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)

    # Convert to numpy array of shape (n, L)
    # Using a lookup table for speed
    # A=0, C=1, G=2, T=3
    lookup = np.array([0, 1, 2, 3], dtype=np.int8)
    # Map characters to indices
    # We can use np.vectorize or a simple loop for conversion if L is small, 
    # but for L up to 2000 and n up to 32, 
    # let's use a fast conversion.
    
    # Efficient conversion:
    # Create a 2D array of chars, then map.
    # Since we only have 4 chars, we can use np.where or a lookup.
    # Let's stack the sequences.
    seq_array = np.empty((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        # Manual loop for filling might be faster than vectorize for small L?
        # Or use bytes conversion.
        # s.encode() gives bytes.
        # We can map bytes to ints.
        # Let's use a simple vectorized approach with a lookup table.
        # Create a table mapping ASCII codes to 0,1,2,3
        # But np.vectorize is slow. 
        # Let's just do:
        # arr = np.frombuffer(s.encode(), dtype=np.uint8)
        # Then map.
        # Since we have only 4 values, we can use:
        # 0 if A, 1 if C, 2 if G, 3 if T
        # 
        # Actually, for N=32 and L=2000, the conversion is 64k operations.
        # A simple Python loop is fine, but let's try to be smart.
        
        # Using a pre-computed mapping for bytes
        # A=65, C=67, G=71, T=84
        # We can use np.take with a lookup table.
        
        # Let's stick to a robust method:
        # Convert string to bytes
        b = s.encode('ascii')
        # Create a lookup table for all 256 possible bytes
        # This is overkill.
        
        # Simplest fast way:
        # seq_array[i] = np.array([0 if c=='A' else 1 if c=='C' else 2 if c=='G' else 3 for c in s], dtype=np.int8)
        # This is slow in Python.
        
        # Better:
        # Use numpy's frombuffer and then a lookup table.
        # lookup_table = np.zeros(256, dtype=np.int8)
        # lookup_table[ord('A')] = 0
        # lookup_table[ord('C')] = 1
        # lookup_table[ord('G')] = 2
        # lookup_table[ord('T')] = 3
        
        # We can define this globally.
        
    # Let's define the lookup table globally for efficiency.
    
    # For now, let's use the vectorized comparison directly on the strings?
    # No, numpy doesn't handle strings well for element-wise comparison without object dtype (slow).
    
    # Let's use the lookup table approach.
    
    pass # Placeholder

def _init_lookup():
    global _LOOKUP
    _LOOKUP = np.zeros(256, dtype=np.int8)
    _LOOKUP[ord('A')] = 0
    _LOOKUP[ord('C')] = 1
    _LOOKUP[ord('G')] = 2
    _LOOKUP[ord('T')] = 3
    _LOOKUP[ord('a')] = 0
    _LOOKUP[ord('c')] = 1
    _LOOKUP[ord('g')] = 2
    _LOOKUP[ord('t')] = 3

_init_lookup()

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0]) if seqs else 0
    
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)

    seq_array = np.empty((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        # Convert string to bytes, then to numpy array of uint8
        b = s.encode('ascii')
        # Use lookup table with proper numpy indexing
        seq_array[i] = _LOOKUP[np.frombuffer(b, dtype=np.uint8)]
        
    # Compute pairwise Hamming distances
    # diff_matrix[i, j] = number of positions where seq_array[i] != seq_array[j]
    # Using broadcasting: (n, 1, L) vs (1, n, L) -> (n, n, L)
    # This might be memory intensive for large L, but L<=2000, n<=32.
    # 32*32*2000 = 2,048,000 bytes. Very small.
    
    diff_matrix = np.sum(seq_array[:, None, :] != seq_array[None, :, :], axis=2)
    
    # Convert to proportions
    P = diff_matrix / L
    
    # Apply JC correction
    # d = -0.75 * ln(1 - 4/3 * p)
    # Handle p >= 0.75 (log of negative or zero)
    # Clip P to avoid log(0) or log(negative)
    # If p >= 0.75, the log term is undefined. Standard practice is to cap the distance or set to a large value.
    # The current code sets it to 3.0.
    
    # We want to calculate d for all elements.
    # Create a mask for p < 0.75
    mask = P < 0.75
    
    D = np.zeros((n, n), dtype=np.float64)
    # Calculate for valid entries
    # 1 - 4/3 * P
    term = 1.0 - (4.0 / 3.0) * P[mask]
    D[mask] = -0.75 * np.log(term)
    
    # Set invalid entries (p >= 0.75) to 3.0
    D[~mask] = 3.0
    
    # Diagonal should be 0
    np.fill_diagonal(D, 0.0)
    
    return D

def _neighbor_joining(D, labels):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    dist_matrix = D.copy()
    tree_substrings = list(labels)
    num_nodes = n
    
    while num_nodes > 3:
        if _time_exceeded():
            break
            
        m = num_nodes
        # Compute total distances (row sums)
        r = np.sum(dist_matrix, axis=1)
        
        # Find the pair that minimizes the NJ criterion
        # Criterion Q(i, j) = (m - 2) * d(i, j) - r(i) - r(j)
        # We want to find min Q.
        
        # Vectorized computation of Q matrix?
        # Q[i, j] = (m-2)*D[i,j] - r[i] - r[j]
        # We can compute the full Q matrix and find the min.
        
        Q = (m - 2) * dist_matrix - r[:, None] - r[None, :]
        
        # We only care about i < j
        # Set diagonal and lower triangle to inf
        Q_lower = np.triu(Q, k=1)
        Q_lower = np.where(Q_lower == 0, float('inf'), Q_lower) # Diagonal is 0 in Q? No, D is 0, so Q[i,i] = -2*r[i]. We don't want self.
        # Actually, we just need to find the min in the upper triangle.
        
        # Flatten and find min, or use a mask
        # For small N, a simple loop is faster than creating masks and argmin on large arrays?
        # N is max 32. 32x32 is small.
        
        # Let's stick to the loop for clarity and to avoid edge cases with Q matrix construction.
        best_criterion = float('inf')
        best_i, best_j = 0, 1
        
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * dist_matrix[i, j] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        
        # Compute branch lengths
        delta = (r[i] - r[j]) / (2.0 * (m - 2))
        d_ij = dist_matrix[i, j]
        len_i = 0.5 * d_ij + 0.5 * delta
        len_j = 0.5 * d_ij - 0.5 * delta
        
        if len_i < 0: len_i = 0.0
        if len_j < 0: len_j = 0.0
        
        # Create new node
        new_name = f"({tree_substrings[i]}:{len_i:.6f},{tree_substrings[j]}:{len_j:.6f})"
        
        # Compute distances from new node to all other nodes
        new_distances = np.zeros(m - 2)
        new_active = []
        for k in range(m):
            if k == i or k == j:
                continue
            new_distances[len(new_active)] = 0.5 * (dist_matrix[i, k] + dist_matrix[j, k] - d_ij)
            new_active.append(k)
            
        # Build new distance matrix
        # We can use slicing to copy the submatrix faster
        # new_dist = np.zeros((m-1, m-1))
        # The submatrix of dist_matrix for new_active x new_active
        # We need to reorder rows and columns.
        
        # Create a mask or index list
        # new_active is a list of indices.
        # dist_matrix[np.ix_(new_active, new_active)] gives the submatrix.
        
        sub_matrix = dist_matrix[np.ix_(new_active, new_active)]
        
        new_dist = np.zeros((m - 1, m - 1), dtype=np.float64)
        new_dist[:m-2, :m-2] = sub_matrix
        new_dist[:m-2, m-2] = new_distances
        new_dist[m-2, :m-2] = new_distances
        new_dist[m-2, m-2] = 0.0
        
        # Update structures
        tree_substrings = [tree_substrings[k] for k in new_active] + [new_name]
        dist_matrix = new_dist
        num_nodes = m - 1
        
    # Combine remaining nodes
    if num_nodes == 3:
        a, b, c = range(3)
        len_a = 0.5 * (dist_matrix[0, 1] + dist_matrix[0, 2] - dist_matrix[1, 2])
        len_b = 0.5 * (dist_matrix[0, 1] + dist_matrix[1, 2] - dist_matrix[0, 2])
        len_c = 0.5 * (dist_matrix[0, 2] + dist_matrix[1, 2] - dist_matrix[0, 1])
        
        if len_a < 0: len_a = 0.0
        if len_b < 0: len_b = 0.0
        if len_c < 0: len_c = 0.0
        
        result = f"({tree_substrings[0]}:{len_a:.6f},{tree_substrings[1]}:{len_b:.6f},{tree_substrings[2]}:{len_c:.6f})"
    elif num_nodes == 2:
        result = f"({tree_substrings[0]}:{dist_matrix[0,1]/2:.6f},{tree_substrings[1]}:{dist_matrix[0,1]/2:.6f})"
    else:
        # If we broke out of the loop early, we might have > 3 nodes.
        # Join them all in a star to ensure a valid tree is returned.
        result = "(" + ",".join(tree_substrings) + ")"
        
    return result + ";"

def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    result = _neighbor_joining(D, labels)
    
    return result
