"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    # Start with schoolbook and try to reduce
    
    # Build the matrix multiplication tensor
    # T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # The tensor is in the space of (n*m) x (m*p) x (p*n)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r
    
    # For the decomposition to be exact:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # and = 0 otherwise (but the tensor only has 1s at valid (i,j,k) positions)
    
    # Actually, the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # and the multiplication is: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor decomposes as: sum_{i,j,k} E_{i,j} (in U-space) * E_{j,k} (in V-space) * E_{k,i} (in W-space)
    
    # Let's try to find a lower-rank decomposition using a search
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = U.shape[1]
        # Compute the reconstructed tensor
        # For each (i,j,k), check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # This is expensive, so let's do it efficiently
        # Create the tensor T of shape (n, m, p)
        T = np.zeros((n, m, p))
        for r in range(r):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # The contribution to T[i,j,k] is u[i,j] * v[j,k] * w[k,i]
            # This is not a simple outer product in 3D
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # T should be all 1s
        if not np.allclose(T, np.ones((n,m,p)), atol=1e-10):
            return False
        return True
    
    # Try to reduce rank by finding linear dependencies
    # Approach: try removing columns and see if we can adjust the remaining ones
    
    # For now, let's try a few known techniques
    # 1. Try to use Strassen-like ideas for blocks
    
    # Let's try a direct search for rank < 120
    # We'll try rank 119, 118, etc. and see if we can find a solution
    
    # Actually, let's try a different approach: use the fact that for small matrices,
    # there are known sub-multiplicative algorithms
    
    # For 4x5x6, let's try to find a good decomposition
    # One approach: use the tensor rank known for smaller cases
    
    # Let me try to use a block decomposition
    # Split 4 = 2+2, 5 = 2+3, 6 = 3+3
    
    # The blocks are:
    # A11 (2x2) * B11 (2x3) -> C11 (2x3): 2x2x3, rank 9
    # A11 (2x2) * B12 (2x3) -> C12 (2x3): 2x2x3, rank 9
    # A12 (2x3) * B21 (3x3) -> C13 (2x3): 2x3x3, rank 18 (schoolbook)
    # A12 (2x3) * B22 (3x3) -> C14 (2x3): 2x3x3, rank 18 (schoolbook)
    # A21 (2x2) * B11 (2x3) -> C21 (2x3): 2x2x3, rank 9
    # A21 (2x2) * B12 (2x3) -> C22 (2x3): 2x2x3, rank 9
    # A22 (2x3) * B21 (3x3) -> C23 (2x3): 2x3x3, rank 18 (schoolbook)
    # A22 (2x3) * B22 (3x3) -> C24 (2x3): 2x3x3, rank 18 (schoolbook)
    
    # Total rank: 4*9 + 4*18 = 36 + 72 = 108
    
    # But wait, the C blocks need to be combined:
    # C11 and C21 contribute to the first 2 rows of the first 3 columns of C
    # This is getting complicated. Let me just return the schoolbook for now
    # and try to improve.
    
    # For the time being, return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
