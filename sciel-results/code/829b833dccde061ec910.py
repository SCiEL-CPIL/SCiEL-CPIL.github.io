"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    Uses a rank-15 construction based on decomposing each column of B independently
    using a rank-3 decomposition for 2x2 * 2x1 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    # Rank is 3 * p = 3 * 5 = 15
    rank = 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each column k in 0..p-1
    for k in range(p):
        # Base index for this column's terms
        base = 3 * k
        
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        # U index for A[0,0] is 0*m + 0 = 0
        # V index for B[0,k] is 0*p + k = k
        # W index for C[0,k] is k*n + 0 = 2k
        U[0][base] = 1
        V[k][base] = 1
        W[2 * k][base] = 1
        
        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        # U index for A[0,1] is 0*m + 1 = 1
        # V index for B[1,k] is 1*p + k = p + k = 5 + k
        # W index for C[0,k] is 2k
        U[1][base + 1] = 1
        V[5 + k][base + 1] = 1
        W[2 * k][base + 1] = 1
        
        # Term 3: (A[1,0] * B[0,k] + A[1,1] * B[1,k]) -> C[1,k]
        # This term handles the second row of C.
        # We combine A[1,0] and A[1,1] with B[0,k] and B[1,k] into a single product?
        # No, the tensor decomposition requires rank-1 terms.
        # My previous derivation:
        # y_1 = c x_0 + d x_1
        # Used:
        # U[2] (c) and U[3] (d)
        # V[0] (x_0) and V[1] (x_1)
        # W[1] (y_1)
        # In the decomposition:
        # Term 3 was: U[2]=1, U[3]=1, V[0]=1, V[1]=1, W[1]=1.
        # This corresponds to (c+d)(x_0+x_1)? No.
        # The tensor sum is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
        # If U[2,c]=1, U[3,c]=1, V[0,c]=1, V[1,c]=1, W[1,c]=1:
        # Contribution to C[1,0] (i=1, k=0):
        # j=0: U[2,0] V[0,0] W[0,0] = 1*1*1 = 1 (c * x_0)
        # j=1: U[3,0] V[5,0] W[0,0] = 1*1*1 = 1 (d * x_1)
        # Sum = c x_0 + d x_1. Correct.
        
        # So for term 3 (index base + 2):
        # U[2] (A[1,0]) = 1
        # U[3] (A[1,1]) = 1
        # V[k] (B[0,k]) = 1
        # V[5+k] (B[1,k]) = 1
        # W[2*k+1] (C[1,k]) = 1
        
        U[2][base + 2] = 1
        U[3][base + 2] = 1
        V[k][base + 2] = 1
        V[5 + k][base + 2] = 1
        W[2 * k + 1][base + 2] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
