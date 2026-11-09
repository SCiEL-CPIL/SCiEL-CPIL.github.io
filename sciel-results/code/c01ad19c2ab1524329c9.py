"""Matrix multiplication 2x4x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the factorization is exact by comparing with schoolbook on random inputs."""
    np.random.seed(42)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for trial in range(20):
        A = np.random.randint(-3, 4, size=(n, m)).astype(float)
        B = np.random.randint(-3, 4, size=(m, p)).astype(float)
        C = A @ B
        
        # Compute C from factorization: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Actually the standard form: C[i,k] = sum_j A[i,j] B[j,k]
        # Factorization: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k]) ... no
        # The correct form for bilinear complexity:
        # C[i,k] = sum_r U[i,k,r] * (linear forms in A and B)
        # But the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # This means: C[i,k] = sum_r U[i*m + j, r] * A[i,j] * V[j*p + k, r] * B[j,k] ... that's not right either.
        
        # Let me re-read: U[n*m, rank], V[m*p, rank], W[p*n, rank]
        # The tensor is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]? No, W is (p*n, rank) indexed by [k*n+i, r]
        # So W[k*n + i, r]
        
        # Standard bilinear complexity form:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]?
        # That doesn't make sense dimensionally for a sum over r.
        
        # Actually the standard form is:
        # C[i,k] = sum_r U[i*m+j, r] ... no.
        
        # Let me think again. The tensor T has indices (i, j, k) where T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
        # The decomposition is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where U[i,j,r] = U[i*m+j, r], V[j,k,r] = V[j*p+k, r], W[k,i,r] = W[k*n+i, r]
        
        # So: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
        #           = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        
        # Wait, that would require a product of two sums, which is not linear in r.
        # The correct bilinear form is:
        # C[i,k] = sum_r f_r(A) * g_r(B) * h_r(i,k)
        # where f_r(A) = sum_j U[i*m+j, r] A[i,j], g_r(B) = sum_j V[j*p+k, r] B[j,k], h_r(i,k) = W[k*n+i, r]
        
        # Let's verify with schoolbook: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = (i*m+j)*p+k
        # Then f_r(A) = A[i,j_r], g_r(B) = B[j_r, k_r], h_r = 1
        # C[i,k] = sum_r A[i,j_r] * B[j_r,k_r] * 1 = sum_j A[i,j] B[j,k] ✓
        
        # So the formula is: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        
        # But wait, that's a sum of products of two linear forms, multiplied by W.
        # For the schoolbook, each r corresponds to a specific (i,j,k) triple, and W[k*n+i, r] = 1.
        
        # Let me just compute it directly for verification.
        C_fact = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    f = 0.0
                    for j in range(m):
                        f += U[i*m + j, r] * A[i, j]
                    g = 0.0
                    for j in range(m):
                        g += V[j*p + k, r] * B[j, k]
                    C_fact[i, k] += W[k*n + i, r] * f * g
        
        if np.max(np.abs(C - C_fact)) > tol:
            return False
    return True

def construct():
    # Schoolbook decomposition: rank = n*m*p = 2*4*6 = 48
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
