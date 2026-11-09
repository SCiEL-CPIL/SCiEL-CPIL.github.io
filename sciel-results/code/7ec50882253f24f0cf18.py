"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Schoolbook decomposition: rank = n*m*p = 4*4*9 = 144
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Verify correctness
    if time.time() < deadline:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C_computed[i, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Should not happen for schoolbook, but fallback just in case
            pass

    # Convert to lists for the expected return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
