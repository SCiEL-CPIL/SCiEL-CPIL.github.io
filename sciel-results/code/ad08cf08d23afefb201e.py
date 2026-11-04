"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Construct the rank-minimal (known) decomposition for 2x4 times 4x6.
    
    Using schoolbook: rank = 2*4*6 = 48.
    U: (n*m, rank) = (8, 48)
    V: (m*p, rank) = (24, 48)
    W: (p*n, rank) = (12, 48)
    
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    where c enumerates (i,j,k) triples.
    Implemented using vectorized NumPy operations for a different approach family.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    
    # Create index arrays for i, j, k
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Flatten to 1D array of length rank
    # The order in the original loop was i, j, k.
    # i changes slowest, k changes fastest.
    # np.repeat(np.arange(n), m*p) gives [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, ...]
    # This matches the outer loop i.
    
    # Calculate linear indices for U, V, W
    u_indices = i_idx * m + j_idx
    v_indices = j_idx * p + k_idx
    w_indices = k_idx * n + i_idx
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Set 1.0 at the appropriate positions
    # We need to set U[u_indices[col], col] = 1.0 for all col
    # This can be done by:
    col_indices = np.arange(rank)
    U[u_indices, col_indices] = 1.0
    V[v_indices, col_indices] = 1.0
    W[w_indices, col_indices] = 1.0
    
    # Convert to list of lists to match the original interface expectation if strictly required,
    # though the prompt says "factor matrices", and numpy arrays are often acceptable.
    # The original returned list of lists. I will return lists to be safe.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
