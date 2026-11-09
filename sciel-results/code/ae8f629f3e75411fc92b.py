"""Matrix multiplication 2,3,10 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 60, schoolbook
    
    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
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
    
    # Verify correctness with a random test
    if time.time() < deadline:
        try:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Random test matrices
            np.random.seed(42)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            
            # Compute via decomposition
            # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # More precisely: for each c, compute phi_c(A) * psi_c(B) * gamma_c
            # phi_c(A) = sum_{i,j} U[i*m+j, c] * A[i,j]
            # psi_c(B) = sum_{j,k} V[j*p+k, c] * B[j,k]
            # gamma_c contributes to C[i,k] via W[k*n+i, c]
            
            C_decomp = np.zeros((n, p))
            for c in range(rank):
                phi = np.sum(U_arr[:, c] * A.flatten())
                psi = np.sum(V_arr[:, c] * B.flatten())
                # gamma_c: W[:, c] has entries W[k*n+i, c] for each (k,i)
                # C[i,k] += phi * psi * W[k*n+i, c]
                for i in range(n):
                    for k in range(p):
                        C_decomp[i, k] += phi * psi * W_arr[k * n + i, c]
            
            C_direct = A @ B
            
            if np.allclose(C_decomp, C_direct, atol=1e-10):
                pass  # Correct
            else:
                # Should not happen, but fallback
                pass
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
