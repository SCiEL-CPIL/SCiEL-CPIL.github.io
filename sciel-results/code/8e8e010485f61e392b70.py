import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p, tol=1e-9):
    """
    Check whether the factor matrices exactly decompose the matmul tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 for all i,j,k.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    expected = np.ones((n, m, p), dtype=np.float64)
    if not np.allclose(T, expected, atol=tol, rtol=tol):
        return False
    return True

def _build_strassen_4x4x4(U, V, W, start_col, b_col_offset, c_row_offset):
    """
    Adds 28 rank-1 terms to U, V, W for a 4x4 x 4x4 multiplication block.
    U rows: 0..15 (fixed)
    V rows: 0..35 (offset by b_col_offset)
    W rows: 0..35 (offset by c_row_offset)
    """
    # Multipliers definition:
    # Each multiplier is a list of (a_row_block, a_col_block, b_row_block, b_col_block, coeff)
    # Blocks are 0 or 1 (indexing the 2x2 grid of 2x2 sub-blocks)
    
    multipliers = [
        # M1 = A00 * (B01 - B11)
        [ (0,0, 0,1, 1), (0,0, 1,1, -1) ],
        # M2 = (A00 + A10) * B11
        [ (0,0, 1,1, 1), (1,0, 1,1, 1) ],
        # M3 = (A01 - A11) * (B10 + B11)
        [ (0,1, 1,0, 1), (0,1, 1,1, 1), (1,1, 1,0, -1), (1,1, 1,1, -1) ],
        # M4 = A11 * (B00 - B01)
        [ (1,1, 0,0, 1), (1,1, 0,1, -1) ],
        # M5 = (A00 + A01) * B00
        [ (0,0, 0,0, 1), (0,1, 0,0, 1) ],
        # M6 = (A10 - A00) * (B00 + B01)
        [ (1,0, 0,0, 1), (1,0, 0,1, 1), (0,0, 0,0, -1), (0,0, 0,1, -1) ],
        # M7 = (A11 - A10) * (B10 - B11)
        [ (1,1, 1,0, 1), (1,1, 1,1, -1), (1,0, 1,0, -1), (1,0, 1,1, 1) ],
    ]
    
    # Output contributions:
    # C00: M1(+1), M4(+1), M5(-1), M7(+1)
    # C01: M3(+1), M5(+1)
    # C10: M2(+1), M4(+1)
    # C11: M1(+1), M2(-1), M3(+1), M6(+1)
    
    contributions = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0], # M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0], # M7 -> C00
    ]
    # Note: The list above is indexed by multiplier index (0-6)
    # and gives coefficients for C00, C01, C10, C11.
    # Let's re-verify M7.
    # C00 = M1 + M4 - M5 + M7. So M7 coeff is 1.
    # C11 = M1 - M2 + M3 + M6. M7 not in C11.
    # So M7 -> C00 only. Correct.
    
    for m_idx in range(7):
        terms = multipliers[m_idx]
        coeffs_out = contributions[m_idx]
        
        # For each u, v, w in {0, 1}
        for u in range(2):
            for v in range(2):
                for w in range(2):
                    col = start_col + m_idx * 4 + u * 2 + v * 2 + w
                    # Wait, we need 28 columns. 7 * 4 = 28.
                    # The index (u,v,w) maps to a specific column.
                    # col = start_col + m_idx * 4 + u * 2 + v + ... ?
                    # Let's just use a counter.
                    pass
                    
    # Let's use a simple counter for the 28 terms
    term_idx = 0
    for m_idx in range(7):
        terms = multipliers[m_idx]
        coeffs_out = contributions[m_idx]
        
        for u in range(2):
            for v in range(2):
                for w in range(2):
                    col = start_col + term_idx
                    term_idx += 1
                    
                    # Initialize vectors for this term
                    U_vec = np.zeros(16)
                    V_vec = np.zeros(36)
                    W_vec = np.zeros(36)
                    
                    # Accumulate U, V, W coefficients from the terms
                    for (ar, ac, br, bc, coeff) in terms:
                        # A index
                        a_row = 2*ar + u
                        a_col = 2*ac + v
                        U_vec[a_row * 4 + a_col] += coeff
                        
                        # B index
                        b_row = 2*br + v
                        b_col = 2*bc + w
                        # Global B col
                        gb_col = b_col + b_col_offset
                        V_vec[b_row * 9 + gb_col] += coeff
                        
                    # W coefficients
                    # C indices depend on which output blocks this multiplier contributes to
                    # C row = 2*cr + u, C col = 2*cc + w
                    # Blocks:
                    # 0: C00 (rows 0-1, cols 0-1)
                    # 1: C01 (rows 0-1, cols 2-3)
                    # 2: C10 (rows 2-3, cols 0-1)
                    # 3: C11 (rows 2-3, cols 2-3)
                    
                    for c_block_idx in range(4):
                        c_coeff = coeffs_out[c_block_idx]
                        if c_coeff != 0:
                            # Determine cr, cc from c_block_idx
                            cr = c_block_idx // 2
                            cc = c_block_idx % 2
                            
                            c_row = 2*cr + u
                            c_col = 2*cc + w
                            
                            # Global C row
                            gc_row = c_row + c_row_offset
                            
                            # W index: W[k*n + i] -> W[c_col * 4 + c_row] ?
                            # Convention: W[k*n+i, r]. k is col of C, i is row of C.
                            # So W index is c_col * 4 + c_row.
                            W_vec[gc_row * 4 + c_col] += c_coeff
                            
                    # Add to global matrices
                    U[:, col] = U_vec
                    V[:, col] = V_vec
                    W[:, col] = W_vec

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Deadline guard
    deadline = start_time + 115.0
    
    rank = 72
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((36, rank), dtype=np.float64)
    W = np.zeros((36, rank), dtype=np.float64)
    
    # Block 1: 4x4 x 4x4 (B cols 0-3, C rows 0-3)
    _build_strassen_4x4x4(U, V, W, 0, 0, 0)
    
    # Block 2: 4x4 x 4x4 (B cols 4-7, C rows 4-7)
    _build_strassen_4x4x4(U, V, W, 28, 4, 4)
    
    # Block 3: 4x4 x 4x1 (B col 8, C row 8)
    # Schoolbook for 4x4 x 4x1
    col = 56
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1.0
            V[j * 9 + 8, col] = 1.0
            W[8 * 4 + i, col] = 1.0
            col += 1
            
    # Validate
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        rank = 144
        U = np.zeros((16, rank), dtype=np.float64)
        V = np.zeros((36, rank), dtype=np.float64)
        W = np.zeros((36, rank), dtype=np.float64)
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U[i * 4 + j, c] = 1.0
                    V[j * 9 + k, c] = 1.0
                    W[k * 4 + i, c] = 1.0
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
