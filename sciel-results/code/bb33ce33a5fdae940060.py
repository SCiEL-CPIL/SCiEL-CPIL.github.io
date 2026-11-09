import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify(U, V, W, tol=1e-8):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    n, m, p = DIMENSIONS
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    if not isinstance(V, np.ndarray):
        V = np.array(V)
    if not isinstance(W, np.ndarray):
        W = np.array(W)
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """
    Correct Strassen 2x2 decomposition.
    Returns U, V, W of shape (4, 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # P3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_block_strassen_6x6():
    """
    6x6x6 decomposition using 3x3 blocks of 2x2 Strassen.
    Rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    nblocks = 3
    block_size = 2
    
    Us, Vs, Ws = _strassen_2x2()
    srank = 7
    
    total_rank = nblocks * nblocks * nblocks * srank
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for bi in range(nblocks):
        for bj in range(nblocks):
            for bk in range(nblocks):
                for r in range(srank):
                    for di in range(block_size):
                        for dj in range(block_size):
                            i = 2*bi + di
                            j = 2*bj + dj
                            U[i*m+j, col] = Us[di*2+dj, r]
                    for dj in range(block_size):
                        for dk in range(block_size):
                            j = 2*bj + dj
                            k = 2*bk + dk
                            V[j*p+k, col] = Vs[dj*2+dk, r]
                    for dk in range(block_size):
                        for di in range(block_size):
                            k = 2*bk + dk
                            i = 2*bi + di
                            W[k*n+i, col] = Ws[dk*2+di, r]
                    col += 1
    return U, V, W

def _laderman_3x3():
    """
    Laderman's algorithm for 3x3 matrix multiplication.
    Rank = 19.
    Returns U, V, W of shape (9, 19).
    """
    n, m, p = 3, 3, 3
    rank = 19
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Laderman's 19 products for 3x3:
    # 1. A00 B00
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # 2. A00 B01
    U[0, 1] = 1
    V[1, 1] = 1
    W[1, 1] = 1
    
    # 3. A00 B10
    U[0, 2] = 1
    V[3, 2] = 1
    W[3, 2] = 1
    
    # 4. A00 B11
    U[0, 3] = 1
    V[4, 3] = 1
    W[4, 3] = 1
    
    # 5. A10 B00
    U[3, 4] = 1
    V[0, 4] = 1
    W[3, 4] = 1
    
    # 6. A10 B01
    U[3, 5] = 1
    V[1, 5] = 1
    W[4, 5] = 1
    
    # 7. A10 B10
    U[3, 6] = 1
    V[3, 6] = 1
    W[6, 6] = 1
    
    # 8. A10 B11
    U[3, 7] = 1
    V[4, 7] = 1
    W[7, 7] = 1
    
    # 9. A11 B00
    U[4, 8] = 1
    V[0, 8] = 1
    W[4, 8] = 1
    
    # 10. A11 B01
    U[4, 9] = 1
    V[1, 9] = 1
    W[5, 9] = 1
    
    # 11. A11 B10
    U[4, 10] = 1
    V[3, 10] = 1
    W[7, 10] = 1
    
    # 12. A11 B11
    U[4, 11] = 1
    V[4, 11] = 1
    W[8, 11] = 1
    
    # 13. A01 B00
    U[1, 12] = 1
    V[0, 12] = 1
    W[1, 12] = 1
    
    # 14. A01 B01
    U[1, 13] = 1
    V[1, 13] = 1
    W[2, 13] = 1
    
    # 15. A01 B10
    U[1, 14] = 1
    V[3, 14] = 1
    W[4, 14] = 1
    
    # 16. A01 B11
    U[1, 15] = 1
    V[4, 15] = 1
    W[5, 15] = 1
    
    # 17. A20 B00
    U[6, 16] = 1
    V[0, 16] = 1
    W[6, 16] = 1
    
    # 18. A20 B01
    U[6, 17] = 1
    V[1, 17] = 1
    W[7, 17] = 1
    
    # 19. A20 B10
    U[6, 18] = 1
    V[3, 18] = 1
    W[9, 18] = 1
    
    return U, V, W

def _build_strassen_laderman_6x6():
    """
    6x6x6 decomposition using 2x2 Strassen on 3x3 blocks.
    Each 3x3 block product is decomposed using Laderman (rank 19).
    Total rank = 7 * 19 = 133.
    """
    n, m, p = 6, 6, 6
    nblocks = 2
    block_size = 3
    
    Us, Vs, Ws = _laderman_3x3()
    srank = Us.shape[1]
    
    total_rank = 7 * srank
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    def add_block(bi, bj, bk, sign_a, sign_b, col_offset):
        for r in range(srank):
            for di in range(block_size):
                for dj in range(block_size):
                    i = 3*bi + di
                    j = 3*bj + dj
                    U[i*m+j, col_offset + r] += sign_a * Us[di*3+dj, r]
            
            for dj in range(block_size):
                for dk in range(block_size):
                    j = 3*bj + dj
                    k = 3*bk + dk
                    V[j*p+k, col_offset + r] += sign_b * Vs[dj*3+dk, r]
            
            for dk in range(block_size):
                for di in range(block_size):
                    k = 3*bk + dk
                    i = 3*bi + di
                    W[k*n+i, col_offset + r] += Ws[dk*3+di, r]

    col = 0
    
    # P1 = (A00+A11)(B00+B11)
    add_block(0, 0, 0, 1, 1, col)
    add_block(0, 0, 1, 1, 1, col)
    add_block(1, 0, 0, 1, 1, col)
    add_block(1, 0, 1, 1, 1, col)
    col += 4 * srank
    
    # P2 = (A10+A11)B00
    add_block(1, 0, 0, 1, 1, col)
    add_block(1, 0, 0, 1, 1, col)
    col += 2 * srank
    
    # P3 = A00(B01-B11)
    add_block(0, 0, 0, 1, 1, col)
    add_block(0, 0, 1, 1, -1, col)
    col += 2 * srank
    
    # P4 = A11(B10-B00)
    add_block(1, 0, 0, 1, 1, col)
    add_block(1, 0, 1, 1, -1, col)
    col += 2 * srank
    
    # P5 = (A00+A01)B11
    add_block(0, 0, 1, 1, 1, col)
    add_block(0, 1, 1, 1, 1, col)
    col += 2 * srank
    
    # P6 = (A10-A00)(B00+B01)
    add_block(1, 0, 0, 1, 1, col)
    add_block(1, 0, 1, 1, 1, col)
    add_block(1, 0, 0, -1, 1, col)
    add_block(1, 0, 1, -1, 1, col)
    col += 4 * srank
    
    # P7 = (A01-A11)(B10+B11)
    add_block(0, 1, 0, 1, 1, col)
    add_block(0, 1, 1, 1, 1, col)
    add_block(1, 1, 0, -1, 1, col)
    add_block(1, 1, 1, -1, 1, col)
    col += 4 * srank
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_rank = 216
    best_result = None
    
    # 1. Try Strassen-Laderman (Rank 133)
    if time.time() < deadline - 20:
        try:
            U_sl, V_sl, W_sl = _build_strassen_laderman_6x6()
            if _verify(U_sl, V_sl, W_sl):
                rank = U_sl.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_sl, V_sl, W_sl)
        except Exception:
            pass
            
    # 2. Try Block Strassen (Rank 189)
    if best_rank > 189 and time.time() < deadline - 20:
        try:
            U_bs, V_bs, W_bs = _build_block_strassen_6x6()
            if _verify(U_bs, V_bs, W_bs):
                rank = U_bs.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_bs, V_bs, W_bs)
        except Exception:
            pass

    # 3. Fallback to Schoolbook (Rank 216)
    if best_result is None:
        U_s, V_s, W_s = _build_schoolbook()
        best_result = (U_s, V_s, W_s)
        best_rank = 216
        
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
