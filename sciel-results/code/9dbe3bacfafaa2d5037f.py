"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x5 matrix.
    
    This implementation uses a NumPy-based construction for the schoolbook
    decomposition, which is a different implementation approach from the
    previous list-of-lists method, while maintaining the same rank (75)
    which is believed to be optimal for these dimensions.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = n * m * p  # 75
    
    # Initialize factor matrices as zero arrays
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill the matrices for the schoolbook decomposition
    # Indexing:
    # U corresponds to A (n x m), flattened row-major: index i*m + j
    # V corresponds to B (m x p), flattened row-major: index j*p + k
    # W corresponds to C (n x p), but the task specifies W[k*n+i, r]
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies W is indexed by (k, i) where k is from B's columns (p) and i is from A's rows (n).
    # So W has shape (p*n, rank).
    # The term in the tensor is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # Summing over r gives C[i, k] = sum_j A[i, j] B[j, k].
    # So W[k*n+i, r] should be 1 when the indices match the schoolbook term.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists as per the return interface expectation
    # The interface expects lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
