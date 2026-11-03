"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2 x 2x2 -> 2x2 using Strassen (rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = -1; W[3, 1] = 1
    
    # M3: a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _strassen_4x4():
    """Return U, V, W for 4x4 x 4x4 -> 4x4 using Strassen on 2x2 blocks (rank 49).
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U2, V2, W2 = _strassen_2x2()
    
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)
    
    # Block-level Strassen coefficients: (block_row, block_col, sign)
    block_A = [
        [(0,0,1), (1,1,1)],   # s=0: M1
        [(1,0,1), (1,1,1)],   # s=1: M2
        [(0,0,1)],            # s=2: M3
        [(1,1,1)],            # s=3: M4
        [(0,0,1), (0,1,1)],   # s=4: M5
        [(1,0,1), (0,0,-1)],  # s=5: M6
        [(0,1,1), (1,1,-1)],  # s=6: M7
    ]
    block_B = [
        [(0,0,1), (1,1,1)],   # s=0: M1
        [(0,0,1)],            # s=1: M2
        [(0,1,1), (1,1,-1)],  # s=2: M3
        [(1,0,1), (0,0,-1)],  # s=3: M4
        [(1,1,1)],            # s=4: M5
        [(0,0,1), (0,1,1)],   # s=5: M6
        [(1,0,1), (1,1,1)],   # s=6: M7
    ]
    block_C = [
        [(0,0,1), (1,1,1)],   # s=0: M1
        [(1,0,-1), (1,1,1)],  # s=1: M2
        [(0,1,1), (1,1,1)],   # s=2: M3
        [(0,0,1), (1,0,1)],   # s=3: M4
        [(0,0,-1), (0,1,1)],  # s=4: M5
        [(1,1,1)],            # s=5: M6
        [(0,0,1)],            # s=6: M7
    ]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            for (bi, bj, sign) in block_A[s]:
                for ii in range(2):
                    for jj in range(2):
                        u_val = U2[ii*2+jj, t] * sign
                        if u_val != 0:
                            i = bi * 2 + ii
                            j = bj * 2 + jj
                            U[i*4+j, r] += u_val
            
            for (bj, bk, sign) in block_B[s]:
                for jj in range(2):
                    for kk in range(2):
                        v_val = V2[jj*2+kk, t] * sign
                        if v_val != 0:
                            j = bj * 2 + jj
                            k = bk * 2 + kk
                            V[j*4+k, r] += v_val
            
            for (bk, bi, sign) in block_C[s]:
                for kk in range(2):
                    for ii in range(2):
                        w_val = W2[kk*2+ii, t] * sign
                        if w_val != 0:
                            k = bk * 2 + kk
                            i = bi * 2 + ii
                            W[k*4+i, r] += w_val
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p  # 144
    U_fb = np.zeros((nm, rank_schoolbook))
    V_fb = np.zeros((mp, rank_schoolbook))
    W_fb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb.copy(), V_fb.copy(), W_fb.copy()
    
    # Strassen-based decomposition: 4x4 x 4x9
    # Split B into [4x4 | 4x4 | 4x1]
    # Two 4x4x4 Strassen blocks (rank 49 each) + one 4x4x1 schoolbook block (rank 16)
    # Total rank = 49 + 49 + 16 = 114
    
    try:
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded before Strassen")
        
        U4, V4, W4 = _strassen_4x4()
        
        rank_total = 49 + 49 + 16
        U = np.zeros((nm, rank_total))
        V = np.zeros((mp, rank_total))
        W = np.zeros((pn, rank_total))
        
        # Block 1: B columns 0-3 (k=0,1,2,3)
        U[:, 0:49] = U4
        for j in range(4):
            for k in range(4):
                V[j*9+k, 0:49] = V4[j*4+k, :]
        for k in range(4):
            for i in range(4):
                W[k*4+i, 0:49] = W4[k*4+i, :]
        
        # Block 2: B columns 4-7 (k=4,5,6,7)
        U[:, 49:98] = U4
        for j in range(4):
            for k in range(4):
                V[j*9+(k+4), 49:98] = V4[j*4+k, :]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i, 49:98] = W4[k*4+i, :]
        
        # Block 3: B column 8 (k=8), schoolbook 4x4x1: rank 16
        for i in range(4):
            for j in range(4):
                r = 98 + i*4 + j
                U[i*4+j, r] = 1
                V[j*9+8, r] = 1
                W[8*4+i, r] = 1
        
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded before return")
        
        best_rank = rank_total
        best_U = U
        best_V = V
        best_W = W
    
    except Exception as e:
        pass
    
    # Convert to list format
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
# EVOLVE-BLOCK-END
