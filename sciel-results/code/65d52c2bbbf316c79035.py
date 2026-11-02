import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Must finish within 120s
    
    n, m, p = DIMENSIONS
    
    # Strassen-based construction: 6 pairs of columns, each with rank-7 Strassen
    # Total rank = 42
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
                # M1 = (A00 + A11) * (B0_{2t} + B1_{2t+1})
                U[0][c] = 1  # A00
                U[3][c] = 1  # A11
                V[0*p + 2*t][c] = 1  # B0, col 2t
                V[1*p + 2*t+1][c] = 1  # B1, col 2t+1
                W[2*t*n + 0][c] = 1  # C0, col 2t
                W[(2*t+1)*n + 1][c] = 1  # C1, col 2t+1
            elif s == 1:
                # M2 = (A10 + A11) * B0_{2t}
                U[2][c] = 1  # A10
                U[3][c] = 1  # A11
                V[0*p + 2*t][c] = 1  # B0, col 2t
                W[2*t*n + 1][c] = 1  # C1, col 2t
                W[(2*t+1)*n + 1][c] = -1  # C1, col 2t+1
            elif s == 2:
                # M3 = A00 * (B1_{2t+1} - B0_{2t+1})
                U[0][c] = 1  # A00
                V[0*p + 2*t+1][c] = 1  # B0, col 2t+1
                V[1*p + 2*t+1][c] = -1  # B1, col 2t+1
                W[(2*t+1)*n + 0][c] = 1  # C0, col 2t+1
                W[(2*t+1)*n + 1][c] = 1  # C1, col 2t+1
            elif s == 3:
                # M4 = A11 * (B1_{2t} - B0_{2t})
                U[3][c] = 1  # A11
                V[1*p + 2*t][c] = 1  # B1, col 2t
                V[0*p + 2*t][c] = -1  # B0, col 2t
                W[2*t*n + 0][c] = 1  # C0, col 2t
                W[2*t*n + 1][c] = 1  # C1, col 2t
            elif s == 4:
                # M5 = (A00 + A01) * B1_{2t+1}
                U[0][c] = 1  # A00
                U[1][c] = 1  # A01
                V[1*p + 2*t+1][c] = 1  # B1, col 2t+1
                W[2*t*n + 0][c] = -1  # C0, col 2t
                W[(2*t+1)*n + 0][c] = 1  # C0, col 2t+1
            elif s == 5:
                # M6 = (A10 - A00) * (B0_{2t} + B0_{2t+1})
                U[2][c] = 1  # A10
                U[0][c] = -1  # A00
                V[0*p + 2*t][c] = 1  # B0, col 2t
                V[0*p + 2*t+1][c] = 1  # B0, col 2t+1
                W[(2*t+1)*n + 1][c] = 1  # C1, col 2t+1
            elif s == 6:
                # M7 = (A01 - A11) * (B1_{2t} + B1_{2t+1})
                U[1][c] = 1  # A01
                U[3][c] = -1  # A11
                V[1*p + 2*t][c] = 1  # B1, col 2t
                V[1*p + 2*t+1][c] = 1  # B1, col 2t+1
                W[2*t*n + 0][c] = 1  # C0, col 2t
    
    # Verify the decomposition
    import numpy as np
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
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    if max_err > 1e-9:
        # Fallback to schoolbook
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
