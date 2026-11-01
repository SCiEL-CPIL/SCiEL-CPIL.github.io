import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's 7-multiplication algorithm.
    n = m = p = 2.
    U: (4, 7), V: (4, 7), W: (4, 7).
    Indexing:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    n = 2
    rank = 7
    U = np.zeros((n*n, rank), dtype=int)
    V = np.zeros((n*n, rank), dtype=int)
    W = np.zeros((n*n, rank), dtype=int)
    
    # Strassen's 7 products:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # W indexing: W[k*n + i]
    # (i=0, k=0) -> W[0], (i=0, k=1) -> W[2]
    # (i=1, k=0) -> W[1], (i=1, k=1) -> W[3]
    
    # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00 -> C10 += 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3: A00(B01-B11) -> C01 += 1, C11 -= 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = -1
    
    # M4: A11(B10-B00) -> C00 += 1, C10 += 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11 += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def _construct_strassen_4x4():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=int)
    V = np.zeros((n*n, rank), dtype=int)
    W = np.zeros((n*n, rank), dtype=int)
    
    U2, V2, W2 = _get_strassen_2x2()
    
    # The 7 outer Strassen terms for 4x4 multiplication
    # P and Q are 2x2 matrices of 2x2 blocks.
    # P is formed from A blocks, Q from B blocks.
    # C is formed from the resulting 2x2 blocks.
    
    terms = [
        # P_blocks, Q_blocks, C_blocks
        {'P': {(0,0): 1, (1,1): 1}, 'Q': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'P': {(1,0): 1, (1,1): 1}, 'Q': {(0,0): 1}, 'C': {(1,0): 1}},
        {'P': {(0,0): 1}, 'Q': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): -1}},
        {'P': {(1,1): 1}, 'Q': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'P': {(0,0): 1, (0,1): 1}, 'Q': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'P': {(1,0): 1, (0,0): -1}, 'Q': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'P': {(0,1): 1, (1,1): -1}, 'Q': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): -1}},
    ]
    
    for rb, term in enumerate(terms):
        P_coeffs = term['P']
        Q_coeffs = term['Q']
        C_coeffs = term['C']
        
        for rs in range(7):
            r = rb * 7 + rs
            
            # Fill U
            for (bi, bj), pc in P_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        u2_val = U2[a * 2 + b][rs]
                        if u2_val != 0:
                            i = 2 * bi + a
                            j = 2 * bj + b
                            U[i * 4 + j][r] += pc * u2_val
            
            # Fill V
            for (bi, bj), qc in Q_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        v2_val = V2[a * 2 + b][rs]
                        if v2_val != 0:
                            j = 2 * bi + a
                            k = 2 * bj + b
                            V[j * 4 + k][r] += qc * v2_val
            
            # Fill W
            for (ci, cj), cc in C_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        w2_val = W2[b * 2 + a][rs]
                        if w2_val != 0:
                            i = 2 * ci + a
                            k = 2 * cj + b
                            W[k * 4 + i][r] += cc * w2_val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # This should be 1 for all i, j, k
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    if not np.all(T == 1):
        return False
    return True

def _schoolbook_decomposition(n, m, p):
    """
    Fallback: Schoolbook decomposition (rank n*m*p).
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
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
    deadline = time.time() + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Primary: Recursive Strassen (Rank 49)
    try:
        U, V, W = _construct_strassen_4x4()
        if time.time() < deadline:
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Fallback: Schoolbook (Rank 64)
    U, V, W = _schoolbook_decomposition(n, m, p)
    if time.time() < deadline:
        _verify_decomposition(U, V, W, n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
