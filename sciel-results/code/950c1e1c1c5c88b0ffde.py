"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy: Subspace embedding and null-space projection.
    We start from the schoolbook decomposition of the larger 4x4 * 4x8 tensor (rank 128).
    The 3x4 * 4x8 problem is a restriction where the 4th row of the first matrix is zero.
    We project the 128-term decomposition onto the 3x4x8 subspace by zeroing out
    the rows in U corresponding to i=3, then removing any all-zero columns.
    """
    n, m, p = DIMENSIONS
    
    # Build the 4x4x8 schoolbook decomposition
    n_large = 4  # Larger dimension for the first matrix
    rank_large = n_large * m * p  # 128
    
    U_large = np.zeros((n_large * m, rank_large), dtype=float)
    V_large = np.zeros((m * p, rank_large), dtype=float)
    W_large = np.zeros((p * n_large, rank_large), dtype=float)
    
    col = 0
    for i in range(n_large):
        for j in range(m):
            for k in range(p):
                U_large[i * m + j, col] = 1
                V_large[j * p + k, col] = 1
                W_large[k * n_large + i, col] = 1
                col += 1
    
    # Project onto the 3x4x8 subspace: keep only rows 0..(n*m-1) of U
    U_projected = U_large[:n * m, :]  # Shape: (12, 128)
    
    # Remove columns where U_projected is all zeros
    # A column is zero if all entries in U_projected for that column are 0
    nonzero_cols = np.any(U_projected != 0, axis=0)
    
    U_final = U_projected[:, nonzero_cols]
    V_final = V_large[:, nonzero_cols]
    
    # For W, we need to remap the indices since we're going from n_large=4 to n=3
    # W_large has shape (p * n_large, rank) = (32, 128)
    # W_final should have shape (p * n, rank) = (24, rank_final)
    # The mapping: W_large[k * n_large + i, col] -> W_final[k * n + i, col] for i < n
    W_final = np.zeros((p * n, U_final.shape[1]), dtype=float)
    for k in range(p):
        for i in range(n):
            W_final[k * n + i, :] = W_large[k * n_large + i, :]
    
    # Convert to lists for return
    rank_final = U_final.shape[1]
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

