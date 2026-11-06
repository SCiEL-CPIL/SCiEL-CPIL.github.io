<<<<<<< SEARCH
"""Optimized Neighbor-Joining with vectorized distance computation."""
import numpy as np
import math


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using NumPy vectorization."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Convert to binary matrix: 1 if A, 0 otherwise (we'll do this for each base)
    # Actually, for efficiency, we can compute differences directly
    # Let's encode each sequence as a numpy array of integers
    # A=0, C=1, G=2, T=3
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Create matrix of shape (n, length)
    seq_matrix = np.zeros((n, length), dtype=np.int8)
    for i, seq in enumerate(seq_list):
        for j, ch in enumerate(seq):
            seq_matrix[i, j] = base_to_int[ch]
    
    # Compute pairwise differences
    # For each pair (i, j), count positions where seq_matrix[i] != seq_matrix[j]
    # Use broadcasting: diff[i, j, k] = seq_matrix[i, k] != seq_matrix[j, k]
    # This would be (n, n, length) which for n=32, length=2000 is 32*32*2000 = 2M entries, fine
    
    # More efficient: compute sum of differences
    # diff_matrix[i, j] = sum over k of (seq_matrix[i, k] != seq_matrix[j, k])
    
    # Use: (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2)
    diff_matrix = (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2).astype(np.float64)
    
    # Compute p = diff / length
    p = diff_matrix / length
    
    # Jukes-Cantor correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 (saturation) by capping
    np.clip(p, 0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d
=======
"""Optimized Neighbor-Joining with vectorized distance computation."""
import numpy as np
import math


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using one-hot encoding and matrix multiplication."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    if n == 0:
        return np.array([])
        
    # Encode sequences to one-hot matrices for A, C, G, T
    # M_A, M_C, M_G, M_T of shape (n, length)
    
    # For efficiency, we can create a single (n, length, 4) array or 4 separate arrays.
    # We'll use 4 separate arrays and sum their dot products.
    
    # Initialize
    M_A = np.zeros((n, length), dtype=np.float32)
    M_C = np.zeros((n, length), dtype=np.float32)
    M_G = np.zeros((n, length), dtype=np.float32)
    M_T = np.zeros((n, length), dtype=np.float32)
    
    for i, seq in enumerate(seq_list):
        # Vectorized conversion
        # Using numpy arrays for speed
        s = np.array(list(seq), dtype='|S1')
        M_A[i, :] = (s == b'A').astype(np.float32)
        M_C[i, :] = (s == b'C').astype(np.float32)
        M_G[i, :] = (s == b'G').astype(np.float32)
        M_T[i, :] = (s == b'T').astype(np.float32)
        
    # Compute matches
    # Matches[i, j] = sum_k (M_A[i,k]*M_A[j,k] + ...)
    # This is M_A @ M_A.T + M_C @ M_C.T + ...
    
    matches = M_A @ M_A.T + M_C @ M_C.T + M_G @ M_G.T + M_T @ M_T.T
    
    # Mismatches = length - matches
    mismatches = length - matches
    
    # p = mismatches / length
    p = mismatches / length
    
    # Jukes-Cantor correction
    np.clip(p, 0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d
>>>>>>> REPLACE
