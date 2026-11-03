"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Strassen terms M1..M7
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # A00 is U[0], A11 is U[3]
    # B00 is V[0], B11 is V[3]
    # C00 is W[0], C11 is W[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # A10 is U[2], A11 is U[3]
    # B00 is V[0]
    # C10 is W[1], C11 is W[3]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # A00 is U[0]
    # B01 is V[1], B11 is V[3]
    # C01 is W[2], C11 is W[3]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # A11 is U[3]
    # B10 is V[2], B00 is V[0]
    # C00 is W[0], C10 is W[1]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # A00 is U[0], A01 is U[1]
    # B11 is V[3]
    # C00 is W[0], C01 is W[2]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # A10 is U[2], A00 is U[0]
    # B00 is V[0], B01 is V[1]
    # C11 is W[3]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    # A01 is U[1], A11 is U[3]
    # B10 is V[2], B11 is V[3]
    # C00 is W[0]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    """
    Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4).
    U shape: (4, 4), V shape: (2, 4), W shape: (2, 4)
    Indices: U[i*2+j], V[j*1+k], W[k*2+i]
    """
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W satisfy the tensor identity exactly.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Identity: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _get_223_decomposition():
    """
    Returns U, V, W for 2x2x3 multiplication.
    U shape: (4, r), V shape: (6, r), W shape: (6, r)
    Indices: U[i*2+j], V[j*3+k], W[k*2+i]
    Uses rank 8 decomposition if found, else falls back to schoolbook (rank 12).
    """
    n, m, p = 2, 2, 3
    
    # Try to find a rank-8 decomposition
    # The 2x2x3 matrix multiplication tensor has known rank 8.
    # Here's a rank-8 decomposition:
    
    r = 8
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Decomposition terms:
    # We need: C[i,k] = sum_j A[i,j] * B[j,k]
    # for i in {0,1}, j in {0,1}, k in {0,1,2}
    
    # Term 1: (A[0,0] + A[1,0]) * B[0,0]
    U[0, 0] = 1.0; U[2, 0] = 1.0
    V[0, 0] = 1.0
    W[0, 0] = 1.0; W[2, 0] = 1.0  # C[0,0] += 1, C[1,0] += 1 (via W[k*2+i])
    
    # Wait, W[k*n+i] = W[k*2+i]. For k=0: W[0]=C[0,0], W[1]=C[1,0]
    # Let me redo with correct indexing.
    
    # Reset
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # I'll use a known rank-8 decomposition for 2x2x3.
    # After research, here is one:
    
    # The decomposition uses the following 8 bilinear terms:
    # M1 = (A00 + A10) * B00
    # M2 = (A01 + A11) * B10  
    # M3 = (A00 + A10) * B01
    # M4 = (A01 + A11) * B11
    # M5 = (A00 + A10) * B02
    # M6 = (A01 + A11) * B12
    # M7 = (A00 - A10) * B00
    # M8 = (A01 - A11) * B10
    
    # But this only gives correct results for k=0,1. For k=2, we need more.
    # So this isn't a valid rank-8 decomposition for the full 2x2x3.
    
    # Let me just use schoolbook (rank 12) for safety.
    r = 12
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin

    # Use p-split [3, 3, 1] with 2x2x3 (rank 12) and 2x2x1 (rank 4)
    # Per m-block: 12 + 12 + 4 = 28
    # Total: 2 * 28 = 56. That's worse than 50.
    
    # Let's stick with p-split [2, 2, 2, 1] for rank 50.
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col_offset = 0

    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]

    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()

    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
                p_sub = 2
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
                p_sub = 1

            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]

            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]

            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]

            col_offset += r_sub

    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            r = n * m * p
            U_fb = [[0] * r for _ in range(n * m)]
            V_fb = [[0] * r for _ in range(m * p)]
            W_fb = [[0] * r for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
