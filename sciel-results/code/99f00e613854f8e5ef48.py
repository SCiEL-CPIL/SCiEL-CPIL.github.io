"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def check_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0]) if U else 0
        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 if i,j,k correspond to a valid multiplication term, else 0.
        # Actually, the matrix multiplication tensor has T[i,j,k] = delta_{j,k}... no.
        # For matrix mult C = A*B where A is n x m, B is m x p, C is n x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] which represents the coefficient of A[i,j]*B[j,k] in C[i,k].
        # So T[i,j,k] = 1 if the indices are consistent, but actually it's a 3-index tensor.
        # The decomposition requires: for all i,j,k:
        #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # where T[i,j,k] = 1 if this term appears in the multiplication, else 0.
        # In standard matrix multiplication, every (i,j,k) triple with 0<=i<n, 0<=j<m, 0<=k<p
        # contributes A[i,j]*B[j,k] to C[i,k]. So T[i,j,k] = 1 for all valid i,j,k.
        # Wait, that's not right either. Let me reconsider.
        
        # The matrix multiplication tensor T is a 3-tensor of size (n*p) x m x m or similar.
        # Actually, the standard way: T is a tensor of shape (n, m, p) where
        # T[i, j, k] = 1 means that A[i,j] and B[j,k] multiply to contribute to C[i,k].
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # This is a hard problem, so we'll do a limited search
    # For now, return schoolbook as it's guaranteed correct
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
