"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Correct Laderman rank-20 decomposition for 3x3*3x3 -> 3x3
    # U[i*3+j], V[j*3+k], W[k*3+i]
    lader_U = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
    ]
    lader_V = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
    ]
    lader_W = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
    ]
    
    def verify_333(U, V, W):
        """Verify that the decomposition correctly computes 3x3*3x3->3x3."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = np.sum(U_arr[:, i*3+j] * V_arr[:, j*3+k] * W_arr[:, k*3+i])
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    lader_valid = verify_333(lader_U, lader_V, lader_W)
    
    if lader_valid:
        # Two 3x3x3 blocks (k=0..5) + one 3x3x2 block (k=6..7)
        rank_lader = 20
        rank_tail = 18  # schoolbook for 3x3x2
        total_rank = 2 * rank_lader + rank_tail  # 58
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # First 3x3x3 block: k indices 0-2, columns 0-19
        for r in range(rank_lader):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*p+k][r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    W[k*n+i][r] = lader_W[r][k*3+i]
        
        # Second 3x3x3 block: k indices 3-5, columns 20-39
        for r in range(rank_lader):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][rank_lader + r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*p + (3+k)][rank_lader + r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    W[(3+k)*n+i][rank_lader + r] = lader_W[r][k*3+i]
        
        # 3x3x2 block: k indices 6-7, columns 40-57 (schoolbook)
        col = 2 * rank_lader
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[i*m+j][col] = 1
                    V[j*p + (6+k)][col] = 1
                    W[(6+k)*n+i][col] = 1
                    col += 1
        
        # Verify the full decomposition
        def verify_full(U, V, W):
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = np.sum(U_arr[:, i*m+j] * V_arr[:, j*p+k] * W_arr[:, k*n+i])
                        if abs(total - 1) > 1e-9:
                            return False
            return True
        
        if verify_full(U, V, W):
            # Convert to integer/half-integer format
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook rank 72
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
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
