"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank = n*m*p = 48
    schoolbook_rank = n * m * p
    
    # Build schoolbook as fallback
    U_fallback = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fallback = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fallback = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][column] = 1
                V_fallback[j * p + k][column] = 1
                W_fallback[k * n + i][column] = 1
                column += 1
    
    def verify(U, V, W, rank, n, m, p, timeout=10):
        start = time.time()
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        if rank <= 100:  # Full check feasible
            errors = 0
            for a in range(n*m):
                i, j = divmod(a, m)
                for b in range(m*p):
                    j2, k = divmod(b, p)
                    for c in range(p*n):
                        k2, i2 = divmod(c, n)
                        expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                        actual = np.dot(U_arr[a, :], V_arr[b, :] * W_arr[c, :])
                        if abs(actual - expected) > 1e-6:
                            errors += 1
                        if time.time() - start > timeout:
                            return False
            return errors == 0
        else:
            return False  # Skip expensive check
    
    # Return schoolbook as baseline
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
# EVOLVE-BLOCK-END
