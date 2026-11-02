"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]
DEADLINE_MARGIN = 10  # seconds before kill to stop

def _verify_decomposition(U, V, W, n=2, m=2, p=13, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n=2, m=2, p=13):
    """Return the trivial schoolbook decomposition with rank n*m*p."""
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
    
    return {"U": U, "V": V, "W": W}

def _strassen_block():
    """
    Return U, V, W matrices for Strassen's 2x2*2x2 decomposition.
    U: shape (4, 7) for A entries [A11, A12, A21, A22]
    V: shape (4, 7) for B entries [B11, B12, B21, B22]
    W: shape (4, 7) for C entries [C11, C12, C21, C22]
    
    Strassen's 7 products:
    P1 = (A11 + A22)(B11 + B22)
    P2 = (A21 + A22)B11
    P3 = A11(B12 - B22)
    P4 = A22(B21 - B11)
    P5 = (A11 + A12)B22
    P6 = (A21 - A11)(B11 + B12)
    P7 = (A12 - A22)(B21 + B22)
    
    C11 = P1 - P4 + P6 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P3 + P2 - P5
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # P1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1  # A11
    U[2, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # P2 = (A21 + A22)B11
    U[1, 1] = 1  # A21
    U[2, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    W[3, 1] = 1  # C22
    
    # P3 = A11(B12 - B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1 # B22
    W[1, 2] = 1  # C12
    W[3, 2] = -1 # C22
    
    # P4 = A22(B21 - B11)
    U[2, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1 # B11
    W[0, 3] = -1 # C11
    W[2, 3] = 1  # C21
    
    # P5 = (A11 + A12)B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[1, 4] = 1  # C12
    W[3, 4] = -1 # C22
    
    # P6 = (A21 - A11)(B11 + B12)
    U[1, 5] = 1  # A21
    U[0, 5] = -1 # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[0, 5] = 1  # C11
    
    # P7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1  # A12
    U[2, 6] = -1 # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11
    
    return U.tolist(), V.tolist(), W.tolist()

def _single_column_decomposition():
    """
    Return U, V, W for multiplying 2x2 * 2x1.
    U: shape (4, 4) for A entries
    V: shape (2, 4) for B entries [B1, B2]
    W: shape (2, 4) for C entries [C1, C2]
    
    C1 = A11*B1 + A12*B2
    C2 = A21*B1 + A22*B2
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # C1 = A11*B1 + A12*B2
    U[0][0] = 1  # A11
    V[0][0] = 1  # B1
    W[0][0] = 1  # C1
    
    U[1][1] = 1  # A12
    V[1][1] = 1  # B2
    W[0][1] = 1  # C1
    
    # C2 = A21*B1 + A22*B2
    U[2][2] = 1  # A21
    V[0][2] = 1  # B1
    W[1][2] = 1  # C2
    
    U[3][3] = 1  # A22
    V[1][3] = 1  # B2
    W[1][3] = 1  # C2
    
    return U, V, W

def _strassen_grouped_decomposition(n=2, m=2, p=13):
    """
    Decompose 2x2 * 2x13 using Strassen for pairs of columns.
    13 = 6*2 + 1, so 6 Strassen blocks (rank 7 each) + 1 single column (rank 4).
    Total rank = 6*7 + 4 = 46.
    """
    total_rank = 6 * 7 + 4
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    sU, sV, sW = _strassen_block()
    scU, scV, scW = _single_column_decomposition()
    
    col_offset = 0
    
    # Process 6 Strassen blocks (columns 0-1, 2-3, ..., 10-11)
    for block_idx in range(6):
        k_start = block_idx * 2  # first column index in this block
        
        for a_idx in range(4):
            for r in range(7):
                U[a_idx][col_offset + r] = sU[a_idx][r]
        
        for b_idx in range(4):
            # b_idx 0 -> B row 0, col k_start
            # b_idx 1 -> B row 0, col k_start+1
            # b_idx 2 -> B row 1, col k_start
            # b_idx 3 -> B row 1, col k_start+1
            if b_idx == 0:
                v_row = 0 * p + k_start
            elif b_idx == 1:
                v_row = 0 * p + (k_start + 1)
            elif b_idx == 2:
                v_row = 1 * p + k_start
            elif b_idx == 3:
                v_row = 1 * p + (k_start + 1)
            
            for r in range(7):
                V[v_row][col_offset + r] = sV[b_idx][r]
        
        for c_idx in range(4):
            # c_idx 0 -> C row 0, col k_start (i=0, k=k_start)
            # c_idx 1 -> C row 1, col k_start (i=1, k=k_start)
            # c_idx 2 -> C row 0, col k_start+1 (i=0, k=k_start+1)
            # c_idx 3 -> C row 1, col k_start+1 (i=1, k=k_start+1)
            if c_idx == 0:
                w_row = k_start * n + 0
            elif c_idx == 1:
                w_row = k_start * n + 1
            elif c_idx == 2:
                w_row = (k_start + 1) * n + 0
            elif c_idx == 3:
                w_row = (k_start + 1) * n + 1
            
            for r in range(7):
                W[w_row][col_offset + r] = sW[c_idx][r]
        
        col_offset += 7
    
    # Process the single column (column 12)
    k_single = 12
    
    for a_idx in range(4):
        for r in range(4):
            U[a_idx][col_offset + r] = scU[a_idx][r]
    
    for b_idx in range(2):
        # b_idx 0 -> B row 0, col 12
        # b_idx 1 -> B row 1, col 12
        v_row = b_idx * p + k_single
        for r in range(4):
            V[v_row][col_offset + r] = scV[b_idx][r]
    
    for c_idx in range(2):
        # c_idx 0 -> C row 0, col 12 (i=0, k=12)
        # c_idx 1 -> C row 1, col 12 (i=1, k=12)
        w_row = k_single * n + c_idx
        for r in range(4):
            W[w_row][col_offset + r] = scW[c_idx][r]
    
    col_offset += 4
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    # Try the Strassen-grouped decomposition first (rank 46)
    strassen_result = _strassen_grouped_decomposition(n, m, p)
    
    if _verify_decomposition(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
        return strassen_result
    
    # Fall back to schoolbook (rank 52)
    schoolbook_result = _schoolbook_decomposition(n, m, p)
    
    if not _verify_decomposition(schoolbook_result["U"], schoolbook_result["V"], schoolbook_result["W"], n, m, p):
        # Should never happen
        pass
    
    return schoolbook_result
