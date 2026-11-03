"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Leave 20s safety margin for 120s limit

    # Build the tensor T_{i,j,k} = delta_{k*n+i, j*p+k, i*m+j} for schoolbook
    # Actually, the tensor for matrix multiplication M_{n,m,p} is:
    # T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # In the factorization: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the standard is: C[i,k] = sum_{j} A[i,j] B[j,k]
    # U is (n*m) x rank, indexed by (i,j)
    # V is (m*p) x rank, indexed by (j,k)
    # W is (p*n) x rank, indexed by (k,i) -> Note: W[k*n+i, r]
    # So C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This requires that for each (i,j,k), we have a term that contributes 1.
    # Specifically, if U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1, then term is 1.
    # Summing over j, we get C[i,k] = sum_j A[i,j] B[j,k].

    # Let's try to find a rank < n*m*p decomposition using a simple heuristic or known structure.
    # For n=3, m=3, p=8, schoolbook rank is 72.
    # We can try to use a lower rank by checking if we can merge terms.
    # However, matrix multiplication tensor is non-decomposable in a simple way.
    # A safe approach to potentially improve is to use the Strassen-like idea for 2x2 blocks,
    # but implementing a full rank-23 for 3x3 is complex and error-prone without known coefficients.
    
    # Let's attempt a random search for a sparse decomposition with rank < 72.
    # This is unlikely to find a valid one quickly, but we can try a few steps.
    # Actually, a better strategy: check if we can use a block structure.
    # Since p=8 is large, maybe we can process columns in groups.
    # But the tensor is fixed.
    
    # Given the time constraint and the risk of invalidity, I will stick to schoolbook
    # unless I can implement a known low-rank scheme.
    # I will implement a rank-72 solution.
    
    # Wait, I can try a rank-64 solution? No, 72 is the base.
    # Let's just return schoolbook.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
