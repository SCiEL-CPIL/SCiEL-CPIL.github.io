"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n=4, m=4, p=5):
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

def _check_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta(i,j,k) for all i,j,k."""
    n, m, p = int(n), int(m), int(p)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need T[i,j,k] = 1 if (i,j,k) is a valid triple? No, the matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] should equal delta(i,j,k)? No.
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k? No.
    # Actually, T is the structure tensor of matrix multiplication.
    # T[i,j,k] = A[i,j] * B[j,k] * C[k,i]? No.
    # The standard tensor for multiplying A (n x m) by B (m x p) to get C (n x p) is:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = A[i,j] * B[j,k] -> this gives C[i,k]
    # So the tensor entries are: for each term A[i,j]*B[j,k], it contributes to C[i,k].
    # The structure tensor T has T[i,j,k] = 1 for all i,j,k? No.
    # Let me think again. The matrix multiplication tensor mu_{n,m,p} has:
    # mu[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # But actually, in the tensor decomposition:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a 3-way tensor T[i, j, k] where T[i,j,k] = delta(i,j,k)? No.
    # The tensor T is defined by T[i,j,k] = A[i,j] B[j,k] C[k,i]? That's the trace convention.
    # The task says: "The output factor follows the official trace convention W[k*n+i,r]."
    # So the decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] is the matrix multiplication tensor.
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k? 
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's the "all ones" tensor, not the matrix multiplication tensor.
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for multiplying an n x m matrix by an m x p matrix:
    # T[i, j, k] where i indexes rows of A/C, j indexes columns of A / rows of B, k indexes columns of B/C
    # T[i,j,k] = 1 if and only if this term appears in the product, which is always true.
    # So T[i,j,k] = 1 for all i,j,k? That would make it the all-ones tensor.
    # No, that's not right either. The matrix multiplication tensor is:
    # mu = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the product C[i,k] = sum_j A[i,j] B[j,k]
    # So actually, the tensor is: for each triple (i,j,k), the coefficient is 1.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Let me verify with the schoolbook: 
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # So sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = number of c where all three are 1
    # In schoolbook, each (i,j,k) triple gets exactly one c, so the sum = 1 for that (i,j,k) and 0 for others.
    # But we want T[i,j,k] = 1 for ALL i,j,k.
    # So the schoolbook gives sum = 1 for every (i,j,k) pair. That's correct!
    # So the check is: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.dot(Ua[i*m+j, :], Va[j*p+k, :]) * 0  # placeholder
    # Better: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is a 3-index contraction.
    # T[i,j,k] = np.einsum('rc,sc,tc->', U[i*m+j], V[j*p+k], W[k*n+i])
    # But that's 1x1. We need to vectorize.
    # Actually, let's just loop for correctness checking.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

# Hardcoded rank 72 decomposition for 4x4x5 if we can find one.
# For now, use schoolbook rank 80.
def construct():
    deadline = time.time() + 100  # 100s safety margin
    n, m, p = DIMENSIONS
    
    # Try to return a rank 72 decomposition if we have one hardcoded.
    # Since we don't have verified rank 72 coefficients, fall back to schoolbook.
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
