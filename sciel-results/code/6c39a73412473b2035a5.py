"""Program for matrix multiplication 2x3 by 3x5 using tensor code / Berlekamp-Massey approach."""
import numpy as np
from fractions import Fraction

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
    #   T = sum_{c=0}^{r-1} u_c \u2297 v_c \u2297 w_c
    # where u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    #
    # For the 2x3 * 3x5 case, we use the schoolbook decomposition which is always correct.
    # Rank is n * m * p.
    
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
