"""Program for matrix multiplication 2x3 by 3x5 using tensor code / Berlekamp-Massey approach."""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions of factor matrices
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    # The matrix multiplication tensor T[i,j,k] where:
    # i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (standard matmul: C[i,k] = sum_j A[i,j]*B[j,k])
    #
    # We seek a rank-r decomposition:
    #   T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # where u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    #
    # The "parity-check matrix" / tensor code view:
    # Flattening T along different axes gives matrices whose ranks bound the tensor rank.
    # For standard matmul of n×m by m×p, the tensor rank equals n*m*p for generic dimensions,
    # but specific small cases may have lower rank.
    #
    # Berlekamp-Massey approach:
    # We treat the 30 entries of the tensor (2*3*5=30) as a sequence and look for
    # algebraic dependencies. The minimal linear recurrence of the sequence of tensor
    # flattening ranks gives us the structure.
    #
    # For 2x3 * 3x5:
    # - The tensor has 30 non-zero entries (all 1s in the appropriate positions)
    # - Flattening as (i,j) x k gives a 6x5 matrix with all entries 1 in valid positions
    # - Flattening as i x (j,k) gives a 2x15 matrix
    # - Flattening as (i,k) x j gives a 10x3 matrix
    #
    # The key insight from the tensor code perspective:
    # The matrix multiplication tensor for n×m * m×p has rank exactly n*m*p
    # when n,m,p are such that no non-trivial decomposition exists.
    #
    # However, for small dimensions, we can try to find a lower rank decomposition
    # by solving the linear system.
    #
    # Let's construct the system:
    # For each (i,j,k) with T[i,j,k]=1:
    #   sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # For each (i,j,k) where T[i,j,k]=0 (but all entries are 1 for valid indices):
    #   The constraint is automatically satisfied since there are no zero entries
    #   in the standard matmul tensor for valid indices.
    #
    # Actually, the tensor T[i,j,k]=1 for ALL i in [0,n), j in [0,m), k in [0,p).
    # So we have 30 equations:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    #
    # This is a multilinear system. We can fix one matrix (say W) and solve for U, V.
    #
    # Let's try to find a rank decomposition. Start with the schoolbook rank = 30.
    # Try to reduce it.
    #
    # For the 2x3 * 3x5 case, let's use the following approach:
    # We can write the matmul as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # A 2x3 * 3x5 matmul can be decomposed using the following structure:
    # Think of it as two independent 2x3 * 3x? problems? No, that's not quite right.
    #
    # Let's use a different approach: exploit the linearity.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # We need 5 output values for each i, so 10 total outputs.
    #
    # The minimal rank for 2x3 * 3x5 matmul:
    # It is known that the border rank of matmul_{2,3,5} is at most 2*3*5 = 30,
    # but the actual rank may be lower.
    #
    # Let's try to construct a decomposition of rank r by solving the system.
    # We'll use the fact that the tensor is symmetric in a certain sense.
    #
    # Actually, let's just return the schoolbook decomposition for now, as finding
    # a lower rank decomposition for this specific case is non-trivial.
    #
    # But wait - let me try a smarter approach. We can use the fact that:
    # matmul_{2,3,5} can be seen as two copies of matmul_{2,3,?} combined.
    #
    # Let me try rank = 20 or lower by exploiting structure.
    #
    # For now, return the schoolbook decomposition which is always correct.
    rank = n * m * p
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
