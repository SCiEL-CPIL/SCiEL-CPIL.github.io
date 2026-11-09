"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, trials=20):
    """Verify that the decomposition correctly computes A·B."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(trials):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        # L_A[c] = sum_{i,j} U[i*m+j, c] * A[i,j]
        # L_B[c] = sum_{j,k} V[j*p+k, c] * B[j,k]
        # C[i,k] = sum_c W[k*n+i, c] * L_A[c] * L_B[c]
        
        A_flat = A.flatten()  # shape (n*m,)
        B_flat = B.flatten()  # shape (m*p,)
        
        L_A = U_arr @ A_flat  # shape (rank,)
        L_B = V_arr @ B_flat  # shape (rank,)
        
        # C[i,k] = sum_c W[k*n+i, c] * L_A[c] * L_B[c]
        prod = L_A * L_B  # shape (rank,)
        C_flat = W_arr @ prod  # shape (p*n,)
        C_computed = C_flat.reshape(p, n).T  # shape (n, p)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook
    best_rank = n * m * p
    best_result = None
    
    def make_schoolbook():
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
    
    best_result = make_schoolbook()
    
    # Try to find better decompositions
    # For each candidate rank, try random initialization and gradient-free optimization
    # This is a hard problem, so let's try a few structured approaches
    
    # Approach: try to merge schoolbook terms
    # If two terms have proportional U and V vectors, we can potentially merge
    
    # For now, return the schoolbook as baseline
    # Future: implement actual search
    
    return best_result
# EVOLVE-BLOCK-END
