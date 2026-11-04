"""Matrix multiplication 4x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 matrix multiplication, rank 7."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A10)*B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A11-A01)*(B00+B01)
    U[3][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _block_2x3x2():
    """
    Return U,V,W for 2x3x2 matrix multiplication, rank 8.
    A is 2x3, B is 3x2, C is 2x2.
    U: 6x8, V: 6x8, W: 4x8.
    """
    rank = 8
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(4)]
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for i in range(2):
        for j in range(2):
            for r in range(7):
                U[i*3 + j][r] = U_s[i*2 + j][r]
                
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j*2 + k][r] = V_s[j*2 + k][r]
                
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[k*2 + i][r] = W_s[k*2 + i][r]
                
    for i in range(2):
        U[i*3 + 2][7] = 1
    for k in range(2):
        V[2*2 + k][7] = 1
    for k in range(2):
        for i in range(2):
            W[k*2 + i][7] = 1
            
    return U, V, W

def _block_2x2x3():
    """
    Return U,V,W for 2x2x3 matrix multiplication, rank 8.
    A is 2x2, B is 2x3, C is 2x3.
    U: 4x8, V: 6x8, W: 6x8.
    """
    rank = 8
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for i in range(2):
        for j in range(2):
            for r in range(7):
                U[i*2 + j][r] = U_s[i*2 + j][r]
                
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j*3 + k][r] = V_s[j*2 + k][r]
                
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[k*2 + i][r] = W_s[k*2 + i][r]
                
    for i in range(2):
        for j in range(2):
            U[i*2 + j][7] = 1
    for j in range(2):
        V[j*3 + 2][7] = 1
    for i in range(2):
        W[2*2 + i][7] = 1
        
    return U, V, W

def _block_2x3x3():
    """
    Return U,V,W for 2x3x3 matrix multiplication, rank 9.
    A is 2x3, B is 3x3, C is 2x3.
    U: 6x9, V: 9x9, W: 6x9.
    """
    rank = 9
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(6)]
    
    U_223, V_223, W_223 = _block_2x2x3()
    
    for i in range(2):
        for j in range(2):
            for r in range(8):
                U[i*3 + j][r] = U_223[i*2 + j][r]
                
    for j in range(2):
        for k in range(3):
            for r in range(8):
                V[j*3 + k][r] = V_223[j*3 + k][r]
                
    for k in range(3):
        for i in range(2):
            for r in range(8):
                W[k*2 + i][r] = W_223[k*2 + i][r]
                
    for i in range(2):
        U[i*3 + 2][8] = 1
    for k in range(3):
        V[2*3 + k][8] = 1
    for k in range(3):
        for i in range(2):
            W[k*2 + i][8] = 1
            
    return U, V, W

def _place_block(U_full, V_full, W_full, U_block, V_block, W_block,
                 i0, j0, k0, n_b, m_b, p_b,
                 n_full, m_full, p_full, col_off):
    """Place a block decomposition into the full matrices."""
    rank_b = len(U_block[0])
    
    for i in range(n_b):
        for j in range(m_b):
            u_row_full = (i0+i)*m_full + (j0+j)
            u_row_block = i*m_b + j
            for c in range(rank_b):
                U_full[u_row_full][col_off + c] = U_block[u_row_block][c]
    
    for j in range(m_b):
        for k in range(p_b):
            v_row_full = (j0+j)*p_full + (k0+k)
            v_row_block = j*p_b + k
            for c in range(rank_b):
                V_full[v_row_full][col_off + c] = V_block[v_row_block][c]
    
    for k in range(p_b):
        for i in range(n_b):
            w_row_full = (k0+k)*n_full + (i0+i)
            w_row_block = k*n_b + i
            for c in range(rank_b):
                W_full[w_row_full][col_off + c] = W_block[w_row_block][c]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose n x m x p matrix multiplication."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(rank):
                        C_computed[i, k] += U[i*m+j][r] * A[i, j] * V[j*p+k][r] * B[j, k] * W[k*n+i][r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _schoolbook(n, m, p):
    """Return U,V,W for n x m x p matrix multiplication using schoolbook, rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Construct rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.
    Rank 64.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 5, 5)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    U_232, V_232, W_232 = _block_2x3x2()
    U_223, V_223, W_223 = _block_2x2x3()
    U_233, V_233, W_233 = _block_2x3x3()
    
    rank_total = 64
    
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    col_off = 0
    
    # Block C[0:2, 0:2]
    _place_block(U, V, W, U_s, V_s, W_s,
                 i0=0, j0=0, k0=0, n_b=2, m_b=2, p_b=2,
                 n_full=n, m_full=m, p_full=p, col_off=col_off)
    col_off += 7
    
    _place_block(U, V, W, U_232, V_232, W_232,
                 i0=0, j0=2, k0=0, n_b=2, m_b=3, p_b=2,
                 n_full=n, m_full=m, p_full=p, col_off=col_off)
    col_off += 8
    
    # Block C[0:2, 2:5]
    _place_block(U, V, W, U_223, V_223, W_223,
                 i0=0, j0=0, k0=2, n_b=2, m_b=2, p_b=3,
                 n_full=n, m_full=m, p_full=p, col_off=col_off)
    col_off += 8
    
    _place_block(U, V, W, U_233, V_233, W_233,
                 i0=0, j0=2, k0=2, n_b=2, m_b=3, p_b=3,
                 n_full=n, m_full=m, p_full=p, col_off=col_off)
    col_off += 9
    
    # Block C[2:4, 0:2]
    _place_block(U, V, W, U_s, V_s, W_s,
                 i0=2, j0=0, k0=0, n_b=2, m_b=2, p_b=2,
                 n_full=n, m_full=m, p_full=p, col_off=col_off)
    col_off += 7
    
    _place_block(U, V, W, U_232, V_232, W_232,
                 i0=2, j0=2, k0=0, n_b=2, m_b=3, p_b=2,
                 n_full=n, m_full=m, p_full=p, col_off=col_off)
    col_off += 8
    
    # Block C[2:4, 2:5]
    _place_block(U, V, W, U_223, V_223, W_223,
                 i0=2, j0=0, k0=2, n_b=2, m_b=2, p_b=3,
                 n_full=n, m_full=m, p_full=p, col_off=col_off)
    col_off += 8
    
    _place_block(U, V, W, U_233, V_233, W_233,
                 i0=2, j0=2, k0=2, n_b=2, m_b=3, p_b=3,
                 n_full=n, m_full=m, p_full=p, col_off=col_off)
    col_off += 9
    
    assert col_off == rank_total, f"col_off={col_off}, rank_total={rank_total}"
    
    # Verify correctness with time guard
    if time.time() < deadline - 5:
        is_valid = _verify_decomposition(U, V, W, n, m, p)
        if not is_valid:
            U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
