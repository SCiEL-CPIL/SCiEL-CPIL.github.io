"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
TOL = 1e-9

# ----------------- helpers -----------------

def _schoolbook(n, m, p):
    """Generates schoolbook decomposition for n x m * m x p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W

def _strassen222():
    """Generates rank-7 Strassen decomposition for 2x2 * 2x2."""
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen terms
    # M0 = (A00+A11)*(B00+B11) -> C00+1, C11-1
    # M1 = (A10+A11)*B00       -> C10+1, C11-1
    # M2 = A00*(B01-B11)       -> C01+1, C11+1
    # M3 = A11*(B10-B00)       -> C00+1, C10+1
    # M4 = (A00+A01)*B11       -> C01+1, C11+1
    # M5 = (A10-A00)*(B00+B01) -> C00-1
    # M6 = (A01-A11)*(B10+B11) -> C00+1
    
    terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:-1}),   # M0
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),        # M1
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # M2
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # M3
        ({0:1, 1:1}, {3:1}, {1:1, 3:1}),         # M4
        ({2:1, 0:-1}, {0:1, 1:1}, {0:-1}),       # M5
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})         # M6
    ]
    
    for r, (ac, bc, cc) in enumerate(terms):
        for ai, val in ac.items():
            i, j = divmod(ai, 2)
            U[i * m + j][r] += val
        for bi, val in bc.items():
            j, k = divmod(bi, 2)
            V[j * p + k][r] += val
        for ci, val in cc.items():
            i, k = divmod(ci, 2)
            W[k * n + i][r] += val
            
    return U, V, W

def _make_2x2x4():
    """Generates rank-11 decomposition for 2x2 * 2x4.
    Uses Strassen for first 2 cols (k=0,1) and Schoolbook for next 2 cols (k=2,3).
    """
    U_s, V_s, W_s = _strassen222()
    U_sb, V_sb, W_sb = _schoolbook(2, 2, 2)
    
    rank = 7 + 4
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(8)] # 2*4
    W = [[0.0] * rank for _ in range(8)] # 4*2
    
    # Fill Strassen part (cols 0,1)
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*2+j][r] = U_s[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*2+k][r] = V_s[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*2+i][r] = W_s[k*2+i][r]
                
    # Fill Schoolbook part (cols 2,3)
    # Local k for schoolbook is 0,1. Global k is 2,3.
    # So V index j*4 + k_global.
    # W index k_global*2 + i.
    offset_r = 7
    for r_local in range(4):
        r = offset_r + r_local
        for i in range(2):
            for j in range(2):
                # U is same for all k, but schoolbook U depends on i,j,k?
                # Schoolbook U[i*m+j, c] = 1 if c corresponds to (i,j,k).
                # Here c is the rank index.
                # In _schoolbook(2,2,2), U[i*2+j, c] is 1 for c corresponding to (i,j,k).
                # We just copy U_sb.
                U[i*2+j][r] = U_sb[i*2+j][r_local]
        for j in range(2):
            for k_local in range(2):
                k_global = 2 + k_local
                V[j*4 + k_global][r] = V_sb[j*2 + k_local][r_local]
        for k_local in range(2):
            k_global = 2 + k_local
            for i in range(2):
                W[k_global*2 + i][r] = W_sb[k_local*2 + i][r_local]
                
    return U, V, W

def _add_block(U_glob, V_glob, W_glob, U_b, V_b, W_b, i_off, j_off, k_off, n_b, m_b, p_b, n, m, p):
    """Adds a block decomposition to the global factors."""
    rank_b = len(U_b[0]) if U_b else 0
    if rank_b == 0: return
    
    # We need to know the current rank of global to know where to start writing.
    # But U_glob is passed by reference? No, we should build it or pass a rank offset.
    # Let's assume U_glob is a list of lists and we append columns?
    # Easier: Pre-allocate or just return the new rank.
    # I'll modify the main construct to accumulate.
    
    pass # This helper is tricky with mutable lists. I'll do it in construct.

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    
    # Define blocks
    # (i_offsets, j_offsets, k_offsets, n_b, m_b, p_b, generator_func)
    blocks = [
        # I1={0,1}, J1={0,1}, K1={0..3}
        ([0,1], [0,1], [0,1,2,3], 2, 2, 4, _make_2x2x4),
        # I1={0,1}, J1={0,1}, K2={4..7}
        ([0,1], [0,1], [4,5,6,7], 2, 2, 4, _make_2x2x4),
        # I1={0,1}, J2={2}, K1={0..3}
        ([0,1], [2], [0,1,2,3], 2, 1, 4, lambda: _schoolbook(2, 1, 4)),
        # I1={0,1}, J2={2}, K2={4..7}
        ([0,1], [2], [4,5,6,7], 2, 1, 4, lambda: _schoolbook(2, 1, 4)),
        # I2={2}, J1={0,1}, K1={0..3}
        ([2], [0,1], [0,1,2,3], 1, 2, 4, lambda: _schoolbook(1, 2, 4)),
        # I2={2}, J1={0,1}, K2={4..7}
        ([2], [0,1], [4,5,6,7], 1, 2, 4, lambda: _schoolbook(1, 2, 4)),
        # I2={2}, J2={2}, K1={0..3}
        ([2], [2], [0,1,2,3], 1, 1, 4, lambda: _schoolbook(1, 1, 4)),
        # I2={2}, J2={2}, K2={4..7}
        ([2], [2], [4,5,6,7], 1, 1, 4, lambda: _schoolbook(1, 1, 4)),
    ]
    
    # Calculate total rank
    total_rank = 0
    block_data = []
    for i_offs, j_offs, k_offs, n_b, m_b, p_b, gen in blocks:
        U_b, V_b, W_b = gen()
        r_b = len(U_b[0]) if U_b else 0
        total_rank += r_b
        block_data.append((i_offs, j_offs, k_offs, n_b, m_b, p_b, U_b, V_b, W_b, r_b))
        
    # Initialize global factors
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Fill global factors
    rank_offset = 0
    for i_offs, j_offs, k_offs, n_b, m_b, p_b, U_b, V_b, W_b, r_b in block_data:
        i_base = i_offs[0]
        j_base = j_offs[0]
        k_base = k_offs[0]
        
        for r_local in range(r_b):
            r_glob = rank_offset + r_local
            
            # Fill U
            for i_local in range(n_b):
                for j_local in range(m_b):
                    i_glob = i_base + i_local
                    j_glob = j_base + j_local
                    U[i_glob * m + j_glob][r_glob] = U_b[i_local * m_b + j_local][r_local]
                    
            # Fill V
            for j_local in range(m_b):
                for k_local in range(p_b):
                    j_glob = j_base + j_local
                    k_glob = k_base + k_local
                    V[j_glob * p + k_glob][r_glob] = V_b[j_local * p_b + k_local][r_local]
                    
            # Fill W
            for k_local in range(p_b):
                for i_local in range(n_b):
                    k_glob = k_base + k_local
                    i_glob = i_base + i_local
                    W[k_glob * n + i_glob][r_glob] = W_b[k_local * n_b + i_local][r_local]
                    
        rank_offset += r_b
        
    return {"U": U, "V": V, "W": W}
