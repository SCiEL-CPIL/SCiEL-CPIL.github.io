"""Rank-minimal exact tensor decomposition search for 3x5x6 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Strassen's 7-multiplication algorithm for 2x2x2 matrix multiplication.
    Returns U, V, W of shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7), dtype=object)
    V = np.zeros((4, 7), dtype=object)
    W = np.zeros((4, 7), dtype=object)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10 += M2
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = -1 # C00 += M3, C01 -= M3
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = 1 # C01 += M4, C11 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1 # C00 += M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1 # C10 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = -1 # C01 += M7, C11 -= M7
    
    return U, V, W

def _schoolbook_block(n_size, m_size, p_size):
    """Generic schoolbook for any block size. Rank = n*m*p."""
    rank = n_size * m_size * p_size
    U = np.zeros((n_size * m_size, rank), dtype=object)
    V = np.zeros((m_size * p_size, rank), dtype=object)
    W = np.zeros((p_size * n_size, rank), dtype=object)
    c = 0
    for i in range(n_size):
        for j in range(m_size):
            for k in range(p_size):
                U[i * m_size + j, c] = 1
                V[j * p_size + k, c] = 1
                W[k * n_size + i, c] = 1
                c += 1
    return U, V, W, rank

def _get_block_algo(n_size, m_size, p_size):
    """Get the best known block algorithm."""
    if n_size == 2 and m_size == 2 and p_size == 2:
        Ub, Vb, Wb = _strassen_2x2x2()
        return Ub, Vb, Wb, 7
    else:
        return _schoolbook_block(n_size, m_size, p_size)

def _build_decomposition(n_parts, m_parts, p_parts):
    n, m, p = 3, 5, 6
    
    blocks = []
    
    for ni, (n_start, n_end) in enumerate(n_parts):
        for mi, (m_start, m_end) in enumerate(m_parts):
            for pi, (p_start, p_end) in enumerate(p_parts):
                n_size = n_end - n_start
                m_size = m_end - m_start
                p_size = p_end - p_start
                
                Ub, Vb, Wb, rank = _get_block_algo(n_size, m_size, p_size)
                
                blocks.append({
                    'n_start': n_start, 'n_end': n_end,
                    'm_start': m_start, 'm_end': m_end,
                    'p_start': p_start, 'p_end': p_end,
                    'U': Ub, 'V': Vb, 'W': Wb,
                    'rank': rank
                })
    
    total_rank = sum(b['rank'] for b in blocks)
    
    U_full = np.zeros((n * m, total_rank), dtype=object)
    V_full = np.zeros((m * p, total_rank), dtype=object)
    W_full = np.zeros((p * n, total_rank), dtype=object)
    
    col_offset = 0
    for b in blocks:
        n_start, n_end = b['n_start'], b['n_end']
        m_start, m_end = b['m_start'], b['m_end']
        p_start, p_end = b['p_start'], b['p_end']
        n_size = n_end - n_start
        m_size = m_end - m_start
        p_size = p_end - p_start
        
        Ub, Vb, Wb = b['U'], b['V'], b['W']
        rank = b['rank']
        
        for c in range(rank):
            # U: i*m + j
            for i in range(n_start, n_end):
                for j in range(m_start, m_end):
                    u_idx = i * m + j
                    ub_idx = (i - n_start) * m_size + (j - m_start)
                    U_full[u_idx, col_offset + c] = Ub[ub_idx, c]
            
            # V: j*p + k
            for j in range(m_start, m_end):
                for k in range(p_start, p_end):
                    v_idx = j * p + k
                    vb_idx = (j - m_start) * p_size + (k - p_start)
                    V_full[v_idx, col_offset + c] = Vb[vb_idx, c]
            
            # W: k*n + i
            for k in range(p_start, p_end):
                for i in range(n_start, n_end):
                    w_idx = k * n + i
                    wb_idx = (k - p_start) * n_size + (i - n_start)
                    W_full[w_idx, col_offset + c] = Wb[wb_idx, c]
        
        col_offset += rank
    
    return U_full, V_full, W_full, total_rank

def _check_exact_numpy(U, V, W, n, m, p):
    """Fast exact check using numpy with integer arithmetic."""
    r = U.shape[1]
    # Convert to float64 for speed (values are small integers)
    Uf = np.array(U, dtype=np.float64)
    Vf = np.array(V, dtype=np.float64)
    Wf = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            u_row = Uf[i * m + j]
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                s = np.dot(u_row, Vf[j * p + k] * Wf[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _check_exact_slow(U, V, W, n, m, p):
    """Slow but exact check with object arithmetic."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0
                for c in range(r):
                    s = s + u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True

def _to_list(mat):
    return mat.tolist()

def construct():
    start = time.time()
    deadline = start + 140.0
    
    # Try multiple partition schemes, keep the best
    best_result = None
    best_rank = float('inf')
    
    # Partition schemes to try
    # Each is (n_parts, m_parts, p_parts)
    schemes = [
        # Scheme 1: n=[2,1], m=[2,2,1], p=[2,2,2]
        ([(0,2),(2,3)], [(0,2),(2,4),(4,5)], [(0,2),(2,4),(4,6)]),
        # Scheme 2: n=[1,2], m=[2,2,1], p=[2,2,2]
        ([(0,1),(1,3)], [(0,2),(2,4),(4,5)], [(0,2),(2,4),(4,6)]),
        # Scheme 3: n=[2,1], m=[1,2,2], p=[2,2,2]
        ([(0,2),(2,3)], [(0,1),(1,3),(3,5)], [(0,2),(2,4),(4,6)]),
        # Scheme 4: n=[1,2], m=[1,2,2], p=[2,2,2]
        ([(0,1),(1,3)], [(0,1),(1,3),(3,5)], [(0,2),(2,4),(4,6)]),
    ]
    
    for scheme in schemes:
        if time.time() > deadline - 10.0:
            break
        try:
            U, V, W, rank = _build_decomposition(*scheme)
            if rank >= best_rank:
                continue
            if _check_exact_numpy(U, V, W, 3, 5, 6):
                # Double-check with exact arithmetic
                if _check_exact_slow(U, V, W, 3, 5, 6):
                    best_rank = rank
                    best_result = {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    n, m, p = 3, 5, 6
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
