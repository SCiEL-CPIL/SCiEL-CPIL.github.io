"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses Strassen's algorithm on 6 pairs of columns, giving rank 42.
    Verifies the decomposition and falls back to schoolbook if needed.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is correct."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def strassen_decomposition():
        """Construct Strassen-based rank-42 decomposition."""
        rank = 42
        U = [[0] * rank for _ in range(n * m)]  # 4 x 42
        V = [[0] * rank for _ in range(m * p)]  # 24 x 42
        W = [[0] * rank for _ in range(p * n)]  # 24 x 42
        
        num_pairs = p // 2  # 6
        
        for t in range(num_pairs):
            base = t * 7
            for s in range(7):
                c = base + s
                if s == 0:
                    # M1 = (A00 + A11) * (B00 + B11)
                    U[0][c] = 1
                    U[3][c] = 1
                    V[0*p + 2*t][c] = 1
                    V[1*p + 2*t+1][c] = 1
                    W[2*t*n + 0][c] = 1
                    W[(2*t+1)*n + 1][c] = 1
                elif s == 1:
                    # M2 = (A10 + A11) * B00
                    U[2][c] = 1
                    U[3][c] = 1
                    V[0*p + 2*t][c] = 1
                    W[2*t*n + 1][c] = 1
                    W[(2*t+1)*n + 1][c] = -1
                elif s == 2:
                    # M3 = A00 * (B01 - B11)
                    U[0][c] = 1
                    V[0*p + 2*t+1][c] = 1
                    V[1*p + 2*t+1][c] = -1
                    W[(2*t+1)*n + 0][c] = 1
                    W[(2*t+1)*n + 1][c] = 1
                elif s == 3:
                    # M4 = A11 * (B10 - B00)
                    U[3][c] = 1
                    V[1*p + 2*t][c] = 1
                    V[0*p + 2*t][c] = -1
                    W[2*t*n + 0][c] = 1
                    W[2*t*n + 1][c] = 1
                elif s == 4:
                    # M5 = (A00 + A01) * B11
                    U[0][c] = 1
                    U[1][c] = 1
                    V[1*p + 2*t+1][c] = 1
                    W[2*t*n + 0][c] = -1
                    W[(2*t+1)*n + 0][c] = 1
                elif s == 5:
                    # M6 = (A10 - A00) * (B00 + B01)
                    U[2][c] = 1
                    U[0][c] = -1
                    V[0*p + 2*t][c] = 1
                    V[0*p + 2*t+1][c] = 1
                    W[(2*t+1)*n + 1][c] = 1
                elif s == 6:
                    # M7 = (A01 - A11) * (B10 + B11)
                    U[1][c] = 1
                    U[3][c] = -1
                    V[1*p + 2*t][c] = 1
                    V[1*p + 2*t+1][c] = 1
                    W[2*t*n + 0][c] = 1
        
        return U, V, W, rank
    
    # Try Strassen rank-42 first
    U, V, W, rank = strassen_decomposition()
    max_err = verify_decomposition(U, V, W, rank)
    
    if max_err > 1e-9:
        # Fallback to schoolbook rank 48
        rank = n * m * p  # 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
