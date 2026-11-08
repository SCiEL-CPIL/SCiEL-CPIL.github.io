import numpy as np

def construct():
    D = 11
    # Generate all combinations of indices
    # We need vectors with two 1s (or -1s) and rest 0.
    # There are 4 signs per pair: (1,1), (1,-1), (-1,1), (-1,-1)
    
    # Number of pairs: 11*10/2 = 55
    # Total vectors: 55 * 4 = 220
    
    # Create an array of indices for all pairs
    # Using np.triu_indices to get i, j pairs
    i_idx, j_idx = np.triu_indices(D, k=1)
    
    # We need to expand each pair into 4 vectors
    # i_idx has shape (55,), j_idx has shape (55,)
    
    # Repeat each index 4 times
    i_rep = np.repeat(i_idx, 4)
    j_rep = np.repeat(j_idx, 4)
    
    # Signs: 00, 01, 10, 11 -> map to 1, -1
    # For the i-th position in the pair
    # Pattern for signs of (val_i, val_j):
    # (1, 1)
    # (1, -1)
    # (-1, 1)
    # (-1, -1)
    
    # Signs for i: [1, 1, -1, -1] repeated 55 times
    signs_i = np.tile([1, 1, -1, -1], 55)
    # Signs for j: [1, -1, 1, -1] repeated 55 times
    signs_j = np.tile([1, -1, 1, -1], 55)
    
    # Initialize empty array (220, 11)
    vectors = np.zeros((220, D), dtype=np.int64)
    
    # Set values
    # i_rep contains the row index where the value goes? No, i_rep is the COLUMN index in the vector.
    # Wait, vectors are rows.
    # vectors[row, col] = val
    
    # i_rep[k] is the column index for the first non-zero element in vector k
    # j_rep[k] is the column index for the second non-zero element in vector k
    
    vectors[np.arange(220), i_rep] = signs_i
    vectors[np.arange(220), j_rep] = signs_j
    
    return vectors
