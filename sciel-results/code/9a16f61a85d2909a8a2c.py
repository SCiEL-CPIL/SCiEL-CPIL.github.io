"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook decomposition
    def schoolbook():
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

    # Validation Function
    def validate(decomp):
        U = np.array(decomp["U"])
        V = np.array(decomp["V"])
        W = np.array(decomp["W"])
        
        n, m, p = DIMENSIONS
        # Random test
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Reconstruct C
        C_reconstructed = np.zeros((n, p))
        
        for r in range(U.shape[1]):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        u_val = u_r[i*m + j]
                        v_val = v_r[j*p + k]
                        w_val = w_r[k*n + i]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            s += u_val * v_val * w_val * A[i,j] * B[j,k]
                    C_reconstructed[i,k] += s
                    
        if not np.allclose(C_reconstructed, C_expected, atol=1e-9):
            return False
        return True

    # Fallback to Schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
