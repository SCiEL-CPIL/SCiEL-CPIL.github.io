"""Matrix multiplication 2,4,8 - Strassen-based rank reduction."""
DIMENSIONS = [2, 4, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _make_strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    U has shape (2*2, 7) = (4, 7), indexed by (i,j)
    V has shape (2*2, 7) = (4, 7), indexed by (j,k)
    W has shape (2*2, 7) = (4, 7), indexed by (k,i)
    
    W index convention: W[k*n + i, c] for n=2, so W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    #
    # U indexed by (i,j): U[i*2+j]
    # V indexed by (j,k): V[j*2+k]
    # W indexed by (k,i): W[k*2+i]
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # U: A00=1, A11=1
    U[0, 0] = 1  # i=0,j=0
    U[3, 0] = 1  # i=1,j=1
    # V: B00=1, B11=1
    V[0, 0] = 1  # j=0,k=0
    V[3, 0] = 1  # j=1,k=1
    # W: C00 gets +1, C11 gets +1
    W[0, 0] = 1  # k=0,i=0
    W[3, 0] = 1  # k=1,i=1
    
    # M2: (A10+A11)B00 -> C10 += M2
    U[2, 1] = 1  # i=1,j=0
    U[3, 1] = 1  # i=1,j=1
    V[0, 1] = 1  # j=0,k=0
    W[0, 1] = 1  # k=0,i=1
    
    # M3: A00(B01-B11) -> C01 += M3, C11 -= M3
    U[0, 2] = 1  # i=0,j=0
    V[1, 2] = 1  # j=0,k=1
    V[3, 2] = -1 # j=1,k=1
    W[1, 0] = 1  # k=1,i=0
    W[3, 2] = -1 # k=1,i=1 -- wait, W[k*2+i], so W[1*2+1]=W[3]
    # C01: k=1,i=0 -> W[2+0]=W[2]
    # C11: k=1,i=1 -> W[2+1]=W[3]
    W[2, 2] = 1
    W[3, 2] = -1
    
    # M4: A11(B10-B00) -> C00 -= M4, C10 += M4
    U[3, 3] = 1  # i=1,j=1
    V[2, 3] = 1  # j=1,k=0
    V[0, 3] = -1 # j=0,k=0
    W[0, 0] = -1 # k=0,i=0
    W[0, 1] = 1  # k=0,i=1
    
    # M5: (A00+A01)B11 -> C01 += M5
    U[0, 4] = 1  # i=0,j=0
    U[1, 4] = 1  # i=0,j=1
    V[3, 4] = 1  # j=1,k=1
    W[2, 4] = 1  # k=1,i=0
    
    # M6: (A10-A00)(B00+B01) -> C00 += M6
    U[2, 5] = 1  # i=1,j=0
    U[0, 5] = -1 # i=0,j=0
    V[0, 5] = 1  # j=0,k=0
    V[1, 5] = 1  # j=0,k=1
    W[0, 5] = 1  # k=0,i=0
    
    # M7: (A01-A11)(B10+B11) -> C11 += M7
    U[1, 6] = 1  # i=0,j=1
    U[3, 6] = -1 # i=1,j=1
    V[2, 6] = 1  # j=1,k=0
    V[3, 6] = 1  # j=1,k=1
    W[3, 6] = 1  # k=1,i=1
    
    return U, V, W


def _make_decomposition(n, m, p):
    """Build full decomposition for n x m x p using 2x2 Strassen blocks.
    
    We tile:
    - n=2 as 1 block of size 2
    - m=4 as 2 blocks of size 2
    - p=8 as 4 blocks of size 2
    
    Each (i_block, j_block, k_block) triple uses one Strassen rank-7 term.
    Total rank = 1 * 2 * 4 * 7 = 56.
    
    U shape: (n*m, rank) = (8, 56), row index = i*m + j
    V shape: (m*p, rank) = (32, 56), row index = j*p + k
    W shape: (p*n, rank) = (16, 56), row index = k*n + i
    """
    n_blocks = n // 2  # 1
    m_blocks = m // 2  # 2
    p_blocks = p // 2  # 4
    
    rank = n_blocks * m_blocks * p_blocks * 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    su, sv, sw = _make_strassen_2x2x2()
    
    col = 0
    for ib in range(n_blocks):
        for jb in range(m_blocks):
            for kb in range(p_blocks):
                for c in range(7):
                    # Strassen pattern c contributes to specific (i,j,k) offsets
                    # su[i*2+j, c] for i,j in {0,1}
                    # sv[j*2+k, c] for j,k in {0,1}
                    # sw[k*2+i, c] for k,i in {0,1}
                    for i_off in range(2):
                        for j_off in range(2):
                            u_val = su[i_off * 2 + j_off, c]
                            if u_val != 0:
                                i_global = ib * 2 + i_off
                                j_global = jb * 2 + j_off
                                U[i_global * m + j_global, col] += u_val
                    for j_off in range(2):
                        for k_off in range(2):
                            v_val = sv[j_off * 2 + k_off, c]
                            if v_val != 0:
                                j_global = jb * 2 + j_off
                                k_global = kb * 2 + k_off
                                V[j_global * p + k_global, col] += v_val
                    for k_off in range(2):
                        for i_off in range(2):
                            w_val = sw[k_off * 2 + i_off, c]
                            if w_val != 0:
                                k_global = kb * 2 + k_off
                                i_global = ib * 2 + i_off
                                W[k_global * n + i_global, col] += w_val
                    col += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Try Strassen-based decomposition (rank 56)
    try:
        U, V, W = _make_decomposition(n, m, p)
        rank = len(U[0])
        
        # Verify correctness
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid and time.time() < deadline:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 64)
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
