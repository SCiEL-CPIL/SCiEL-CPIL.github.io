import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 Strassen multiplication.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing:
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    The tensor identity is sum_r W[k*2+i, r] U[i*2+j, r] V[j*2+k, r] = delta_{k,i}
    for the standard matrix multiplication where C[i, k] = sum_j A[i, j] B[j, k].
    Note: The output index in W is defined as k*n+i. In this 2x2 block, n=2.
    So W[k*2+i] corresponds to the element C[i, k].
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen Products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Combinations for C:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 0: M1
    # A part: A00, A11
    U[0][0] = 1
    U[3][0] = 1
    # B part: B00, B11
    V[0][0] = 1
    V[3][0] = 1
    # C part: C00, C11
    # C00: k=0, i=0 -> idx 0
    # C11: k=1, i=1 -> idx 1*2+1 = 3
    W[0][0] += 1
    W[3][0] += 1

    # Term 1: M2
    # A part: A10, A11
    U[2][1] = 1
    U[3][1] = 1
    # B part: B00
    V[0][1] = 1
    # C part: C10, C11
    # C10 = M2 + ... (+1)
    # C11 = ... - M2 + ... (-1)
    # C10: k=0, i=1 -> idx 0*2+1 = 2
    # C11: k=1, i=1 -> idx 3
    W[2][1] += 1
    W[3][1] += -1

    # Term 2: M3
    # A part: A00
    U[0][2] = 1
    # B part: B01, B11
    V[1][2] = 1
    V[3][2] = -1
    # C part: C01, C11
    # C01 = M3 + ... (+1)
    # C11 = ... + M3 + ... (+1)
    # C01: k=1, i=0 -> idx 1*2+0 = 2
    # C11: k=1, i=1 -> idx 3
    W[2][2] += 1
    W[3][2] += 1

    # Term 3: M4
    # A part: A11
    U[3][3] = 1
    # B part: B10, B00
    V[2][3] = 1
    V[0][3] = -1
    # C part: C00, C10
    # C00 = ... - M4 + ... (-1)
    # C10 = ... + M4 (+1)
    # C00: k=0, i=0 -> idx 0
    # C10: k=0, i=1 -> idx 2
    W[0][3] += -1
    W[2][3] += 1

    # Term 4: M5
    # A part: A00, A01
    U[0][4] = 1
    U[1][4] = 1
    # B part: B11
    V[3][4] = 1
    # C part: C00, C01
    # C00 = ... + M5 ... (+1)
    # C01 = ... + M5 (+1)
    # C00: k=0, i=0 -> idx 0
    # C01: k=1, i=0 -> idx 2
    W[0][4] += 1
    W[2][4] += 1

    # Term 5: M6
    # A part: A10, A00
    U[2][5] = 1
    U[0][5] = -1
    # B part: B00, B01
    V[0][5] = 1
    V[1][5] = 1
    # C part: C11
    # C11 = ... + M6 (+1)
    # C11: k=1, i=1 -> idx 3
    W[3][5] += 1

    # Term 6: M7
    # A part: A01, A11
    U[1][6] = 1
    U[3][6] = -1
    # B part: B10, B11
    V[2][6] = 1
    V[3][6] = 1
    # C part: C00
    # C00 = ... - M7 (-1)
    # C00: k=0, i=0 -> idx 0
    W[0][6] += -1

    return U, V, W

def _build_rank74():
    n, m, p = 3, 5, 5
    
    su, sv, sw = _get_strassen_222()
    
    # Partitions
    # i blocks: 2, 1
    # j blocks: 2, 2, 1
    # k blocks: 2, 2, 1
    
    i_blocks = [(0, 1), (2,)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    total_rank = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b)==2 and len(j_b)==2 and len(k_b)==2:
                    total_rank += 7
                else:
                    total_rank += len(i_b)*len(j_b)*len(k_b)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b)==2 and len(j_b)==2 and len(k_b)==2:
                    # Strassen
                    # i_b has 2 indices, j_b has 2, k_b has 2
                    for c in range(7):
                        # Map local (li, lj) to global
                        for li in range(2):
                            gi = i_b[li]
                            for lj in range(2):
                                gj = j_b[lj]
                                U[gi*m + gj][col+c] = su[li*2+lj][c]
                        for lj in range(2):
                            gj = j_b[lj]
                            for lk in range(2):
                                gk = k_b[lk]
                                V[gj*p + gk][col+c] = sv[lj*2+lk][c]
                        for lk in range(2):
                            gk = k_b[lk]
                            for li in range(2):
                                gi = i_b[li]
                                W[gk*n + gi][col+c] = sw[lk*2+li][c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_b:
                        for gj in j_b:
                            for gk in k_b:
                                U[gi*m + gj][col] = 1
                                V[gj*p + gk][col] = 1
                                W[gk*n + gi][col] = 1
                                col += 1
                                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m + j]
                v_val = V_np[j*p + k]
                w_val = W_np[k*n + i]
                s = np.sum(u_val * v_val * w_val)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
    rank = n * m * p
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Attempt rank 74
    try:
        U, V, W = _build_rank74()
        rank = len(W[0])
        if _verify_decomposition(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
        
    # Fallback to schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
