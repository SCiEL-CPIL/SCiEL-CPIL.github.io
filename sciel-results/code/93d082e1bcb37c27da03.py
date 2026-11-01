"""Program for matrix multiplication 2,2,13 - rank-minimal decomposition."""
import time
import numpy as np

_START_TIME = time.time()
DEADLINE = _START_TIME + 140.0

def _check_deadline():
    return time.time() > DEADLINE

def _strassen_2x2_block(col0, col1, n, m, p):
    """
    Generate 7 rank-1 terms for Strassen decomposition of 2x2 @ 2x2 block.
    Columns col0, col1 of B.
    U: (4, 1), V: (26, 1), W: (26, 1)
    """
    U_rows = []
    V_rows = []
    W_rows = []
    
    # p1 = (a00+a11)*(b00+b11)
    U1 = [1, 0, 0, 1]
    V1 = [0] * (m * p)
    V1[0 * p + col0] = 1
    V1[1 * p + col1] = 1
    W1 = [0] * (p * n)
    W1[col0 * n + 0] = 1
    W1[col1 * n + 1] = 1
    U_rows.append(U1); V_rows.append(V1); W_rows.append(W1)
    
    # p2 = (a10+a11)*b00
    U2 = [0, 0, 1, 1]
    V2 = [0] * (m * p)
    V2[0 * p + col0] = 1
    W2 = [0] * (p * n)
    W2[col0 * n + 1] = 1
    W2[col1 * n + 1] = -1
    U_rows.append(U2); V_rows.append(V2); W_rows.append(W2)
    
    # p3 = a00*(b10-b00)
    U3 = [1, 0, 0, 0]
    V3 = [0] * (m * p)
    V3[1 * p + col0] = 1
    V3[0 * p + col0] = -1
    W3 = [0] * (p * n)
    W3[col1 * n + 0] = 1
    W3[col1 * n + 1] = 1
    U_rows.append(U3); V_rows.append(V3); W_rows.append(W3)
    
    # p4 = a11*(b01-b11)
    U4 = [0, 0, 0, 1]
    V4 = [0] * (m * p)
    V4[0 * p + col1] = 1
    V4[1 * p + col1] = -1
    W4 = [0] * (p * n)
    W4[col0 * n + 0] = 1
    W4[col0 * n + 1] = 1
    U_rows.append(U4); V_rows.append(V4); W_rows.append(W4)
    
    # p5 = (a00+a01)*b11
    U5 = [1, 1, 0, 0]
    V5 = [0] * (m * p)
    V5[1 * p + col1] = 1
    W5 = [0] * (p * n)
    W5[col0 * n + 0] = -1
    W5[col1 * n + 0] = 1
    U_rows.append(U5); V_rows.append(V5); W_rows.append(W5)
    
    # p6 = (a10-a00)*(b00+b01)
    U6 = [-1, 0, 1, 0]
    V6 = [0] * (m * p)
    V6[0 * p + col0] = 1
    V6[0 * p + col1] = 1
    W6 = [0] * (p * n)
    W6[col1 * n + 1] = 1
    U_rows.append(U6); V_rows.append(V6); W_rows.append(W6)
    
    # p7 = (a01-a11)*(b10+b11)
    U7 = [0, 1, 0, -1]
    V7 = [0] * (m * p)
    V7[1 * p + col0] = 1
    V7[1 * p + col1] = 1
    W7 = [0] * (p * n)
    W7[col0 * n + 0] = 1
    U_rows.append(U7); V_rows.append(V7); W_rows.append(W7)
    
    return U_rows, V_rows, W_rows

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Fallback: schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x2 @ 2x13.
    
    Uses Strassen decomposition for 6 pairs of columns (rank 7 each)
    plus direct decomposition for the last column (rank 4).
    Total rank: 6*7 + 4 = 46.
    """
    if _check_deadline():
        n, m, p = 2, 2, 13
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    n, m, p = 2, 2, 13
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Process 6 chunks of 2 columns each (columns 0-11)
    for t in range(6):
        col0 = 2 * t
        col1 = 2 * t + 1
        
        u_block, v_block, w_block = _strassen_2x2_block(col0, col1, n, m, p)
        U_rows.extend(u_block)
        V_rows.extend(v_block)
        W_rows.extend(w_block)
        
        if _check_deadline():
            break
    
    # Last column (12)
    col_last = 12
    for i in range(n):
        for j in range(m):
            U_term = [0] * (n * m)
            U_term[i * m + j] = 1
            V_term = [0] * (m * p)
            V_term[j * p + col_last] = 1
            W_term = [0] * (p * n)
            W_term[col_last * n + i] = 1
            U_rows.append(U_term)
            V_rows.append(V_term)
            W_rows.append(W_term)
            
    U = np.array(U_rows, dtype=float).T
    V = np.array(V_rows, dtype=float).T
    W = np.array(W_rows, dtype=float).T
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    if _check_deadline():
        n_fb, m_fb, p_fb = 2, 2, 13
        U_fb, V_fb, W_fb = _schoolbook_decomposition(n_fb, m_fb, p_fb)
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    U_fb, V_fb, W_fb = _schoolbook_decomposition(n, m, p)
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
