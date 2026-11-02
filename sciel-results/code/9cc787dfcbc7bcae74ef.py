import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Laderman's 23-term decomposition for 3x3 * 3x3
    laderman_23 = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 0,0,1], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 1,0,0], [1,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 1,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,1,0, -1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,-1, 0,0,0], [0,0,0, 0,1,0, 0,0,-1], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, -1,0,0], [1,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,-1,0], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,-1], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 1,0,0, -1,0,0], [1,0,0, 0,0,0, 0,0,-1], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,1,0, 0,-1,0], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,1, 0,0,-1], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,0, 0,0,1]),
    ]
    
    # Build 46-term solution directly
    rank46 = 46
    U46 = np.zeros((9, rank46), dtype=int)
    V46 = np.zeros((18, rank46), dtype=int)
    W46 = np.zeros((18, rank46), dtype=int)
    
    for c in range(23):
        u, v, w = laderman_23[c]
        U46[:, c] = u
        for j in range(3):
            for k in range(3):
                V46[j*6+k, c] = v[j*3+k]
        for k in range(3):
            for i in range(3):
                W46[k*3+i, c] = w[k*3+i]
        
        c2 = c + 23
        U46[:, c2] = u
        for j in range(3):
            for k in range(3, 6):
                V46[j*6+k, c2] = v[j*3+(k-3)]
        for k in range(3, 6):
            for i in range(3):
                W46[k*3+i, c2] = w[(k-3)*3+i]
    
    # Verify 46-term solution
    def verify_solution(U, V, W, n, m, p, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_solution(U46, V46, W46, n, m, p):
        # Fallback to trivial 54-term solution
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=int)
        V_sb = np.zeros((m * p, rank_sb), dtype=int)
        W_sb = np.zeros((p * n, rank_sb), dtype=int)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, column] = 1
                    V_sb[j * p + k, column] = 1
                    W_sb[k * n + i, column] = 1
                    column += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Return the 46-term solution
    return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
# EVOLVE-BLOCK-END
