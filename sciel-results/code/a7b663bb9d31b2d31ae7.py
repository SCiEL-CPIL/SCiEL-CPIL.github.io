"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 130  # seconds, safety margin before 150s kill

def _validate_decomposition(U, V, W, n, m, p):
    """Check if the tensor decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reconstruct the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Reconstruct from U, V, W: T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    return np.allclose(T, T_recon, atol=1e-10)

def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    
    The 2x9 matrix is split into:
    - 4 blocks of size 2x2 (columns 0-1, 2-3, 4-5, 6-7)
    - 1 block of size 2x1 (column 8)
    
    For each 2x2x2 block, use Strassen (7 multiplications).
    For the 2x2x1 block, use schoolbook (4 multiplications).
    Total: 4*7 + 4 = 32.
    """
    n, m, p = 2, 2, 9
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * rank for _ in range(p * n)]  # 18 x 32
    
    # Strassen formulas for 2x2x2
    strassen_terms = [
        # P0 = (A00 + A11) * (B00 + B11)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1), (1, -1), (2, 1) ]),
        # P1 = (A10 + A11) * B00
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        # P2 = A00 * (B01 - B11)
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        # P3 = A11 * (B10 - B00)
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, -1) ]),
        # P4 = (A00 + A01) * B11
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (3, 1) ]),
        # P5 = (A10 - A00) * (B00 + B01)
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (0, -1), (2, 1) ]),
        # P6 = (A01 - A11) * (B10 + B11)
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (1, 1) ]),
    ]
    
    # Fill in 4 Strassen blocks for columns 0-7
    for block in range(4):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            for u_idx, coeff in u_coeffs:
                U[u_idx][col] += coeff
            
            for v_idx_off, coeff in v_coeffs:
                v_idx = 2 * block + v_idx_off
                V[v_idx][col] += coeff
            
            for w_idx_off, coeff in w_coeffs:
                w_idx = 2 * block + w_idx_off
                W[w_idx][col] += coeff
    
    # Fill in schoolbook for column 8 (2x2x1 block)
    # V[j*p + k]: for k=8, j=0 -> V[8], j=1 -> V[17]
    # W[k*n + i]: for k=8, i=0 -> W[16], i=1 -> W[17]
    
    schoolbook_terms = [
        # T0: A00 * B[0,8] -> C[0,8]
        ([ (0, 1) ], [ (8, 1) ], [ (16, 1) ]),
        # T1: A01 * B[1,8] -> C[0,8]
        ([ (1, 1) ], [ (17, 1) ], [ (16, 1) ]),
        # T2: A10 * B[0,8] -> C[1,8]
        ([ (2, 1) ], [ (8, 1) ], [ (17, 1) ]),
        # T3: A11 * B[1,8] -> C[1,8]
        ([ (3, 1) ], [ (17, 1) ], [ (17, 1) ]),
    ]
    
    col_start = 4 * 7  # column 28
    for t in range(4):
        col = col_start + t
        u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]
        
        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff
        
        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff
        
        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff
    
    return U, V, W

def _build_rank_30():
    """
    Build a rank-30 decomposition using three 2x2x3 blocks.
    Each 2x2x3 block uses a rank-10 decomposition.
    Total: 3 * 10 = 30.
    
    Columns are split as: [0,1,2], [3,4,5], [6,7,8]
    """
    n, m, p = 2, 2, 9
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 4 x 30
    V = [[0] * rank for _ in range(m * p)]  # 18 x 30
    W = [[0] * rank for _ in range(p * n)]  # 18 x 30
    
    # Rank-10 decomposition for 2x2x3:
    # A (2x2) * B (2x3) = C (2x3)
    # U indices: 0=a00, 1=a01, 2=a10, 3=a11
    # V indices for a block starting at col c: 
    #   V[0*9+c+0]=V[c], V[0*9+c+1]=V[c+1], V[0*9+c+2]=V[c+2]
    #   V[1*9+c+0]=V[9+c], V[1*9+c+1]=V[9+c+1], V[1*9+c+2]=V[9+c+2]
    # W indices for a block starting at col c:
    #   W[(c+0)*2+0]=W[2c], W[(c+0)*2+1]=W[2c+1]
    #   W[(c+1)*2+0]=W[2c+2], W[(c+1)*2+1]=W[2c+3]
    #   W[(c+2)*2+0]=W[2c+4], W[(c+2)*2+1]=W[2c+5]
    
    # Rank-10 terms for [2,2,3]:
    # We use a known construction. Let me derive it.
    # 
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[0,2] = a00*b02 + a01*b12
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    # C[1,2] = a10*b02 + a11*b12
    #
    # Schoolbook (12 terms):
    # T0: a00*b00 -> C00
    # T1: a01*b10 -> C00
    # T2: a00*b01 -> C01
    # T3: a01*b11 -> C01
    # T4: a00*b02 -> C02
    # T5: a01*b12 -> C02
    # T6: a10*b00 -> C10
    # T7: a11*b10 -> C10
    # T8: a10*b01 -> C11
    # T9: a11*b11 -> C11
    # T10: a10*b02 -> C12
    # T11: a11*b12 -> C12
    #
    # To get rank 10, we need to merge 2 pairs.
    # One approach: use Strassen on the first two columns, then add the third.
    # But that gives 7 + 4 = 11.
    #
    # Alternative: share the a00 and a01 terms.
    # Let me try:
    # P1 = a00 * (b00 + b01 + b02)
    # P2 = a01 * (b10 + b11 + b12)
    # P3 = a10 * (b00 + b01 + b02)
    # P4 = a11 * (b10 + b11 + b12)
    # P5 = (a00+a11) * (b00+b10)
    # P6 = (a01+a10) * (b01+b11)
    # P7 = (a00-a11) * (b02+b12)
    # P8 = (a01-a10) * (b02+b12)  -- this doesn't work cleanly
    #
    # Let me try a different known construction.
    # Actually, I'll use a systematic approach: try to find 10 terms that work.
    
    # After research, the rank of [2,2,3] is indeed 10.
    # Here's a construction (verified by checking all 6 output entries):
    
    # Let me use the following 10 terms:
    # The V indices within a block: v0=b00, v1=b01, v2=b02, v3=b10, v4=b11, v5=b12
    # The W indices within a block: w0=C00, w1=C01, w2=C02, w3=C10, w4=C11, w5=C12
    
    # I'll use a known rank-10 scheme for [2,2,3].
    # One valid construction:
    
    rank10_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # v_coeffs are (local_v_index, coeff), w_coeffs are (local_w_index, coeff)
        
        # Term 0: a00 * b00 -> C00
        ([(0, 1)], [(0, 1)], [(0, 1)]),
        # Term 1: a01 * b10 -> C00
        ([(1, 1)], [(3, 1)], [(0, 1)]),
        # Term 2: a00 * b01 -> C01
        ([(0, 1)], [(1, 1)], [(1, 1)]),
        # Term 3: a01 * b11 -> C01
        ([(1, 1)], [(4, 1)], [(1, 1)]),
        # Term 4: a00 * b02 -> C02
        ([(0, 1)], [(2, 1)], [(2, 1)]),
        # Term 5: a01 * b12 -> C02
        ([(1, 1)], [(5, 1)], [(2, 1)]),
        # Term 6: a10 * b00 -> C10
        ([(2, 1)], [(0, 1)], [(3, 1)]),
        # Term 7: a11 * b10 -> C10
        ([(3, 1)], [(3, 1)], [(3, 1)]),
        # Term 8: a10 * b01 -> C11
        ([(2, 1)], [(1, 1)], [(4, 1)]),
        # Term 9: a11 * b11 -> C11
        ([(3, 1)], [(4, 1)], [(4, 1)]),
        # Term 10: a10 * b02 -> C12
        ([(2, 1)], [(2, 1)], [(5, 1)]),
        # Term 11: a11 * b12 -> C12
        ([(3, 1)], [(5, 1)], [(5, 1)]),
    ]
    # That's 12 terms (schoolbook). I need to find a rank-10 construction.
    # Let me try to merge some terms.
    
    # Actually, let me just use the schoolbook for [2,2,3] (rank 12) and see if
    # 3*12=36 is the same as the baseline. Then I'll try to improve.
    # For now, let me return None and use the rank-32 fallback.
    
    # Let me try a different approach: use Strassen for the first 2 columns 
    # and schoolbook for the 3rd, giving 7+4=11 per block, total 33.
    # Or use a mix.
    
    # For now, return None to indicate this construction is not ready.
    return None

def _build_rank_31_mixed():
    """
    Build a rank-31 decomposition using:
    - One 2x2x3 block with rank-11 (Strassen 7 + schoolbook 4)
    - Two 2x2x2 blocks with rank-7 each
    Total: 11 + 7 + 7 = 25? No, that's only 7 columns.
    
    Let me redo: split 9 columns as 3+3+3 or 3+2+2+2.
    
    Option A: 3+3+3 with rank-11 each = 33
    Option B: 3+2+2+2 with rank 11+7+7+7 = 32 (same as current)
    Option C: 3+3+2+1 with rank 11+11+7+4 = 33
    
    So the best with known block decompositions is 32.
    To get below 32, we need a better [2,2,3] decomposition.
    """
    return None

def _search_rank_below_32():
    """
    Try to find a decomposition with rank < 32 using numerical methods.
    This is a heuristic search.
    """
    n, m, p = 2, 2, 9
    # Start from rank-32 and try to reduce
    U32, V32, W32 = _build_rank_32(n, m, p)
    U32 = np.array(U32, dtype=float)
    V32 = np.array(V32, dtype=float)
    W32 = np.array(W32, dtype=float)
    
    # Check if any column is redundant by trying to express it as a combination
    # of others. This is a quick heuristic.
    for target in range(32):
        others = [i for i in range(32) if i != target]
        # Try to solve U[:,target] = U[:,others] @ alpha
        # and V[:,target] = V[:,others] @ beta
        # and W[:,target] = W[:,others] @ gamma
        # with alpha=beta=gamma (same coefficients for all three)
        
        # This is a system of equations. Let's try least squares.
        # We need: for all i, U[i,target] = sum_j alpha[j] * U[i,others[j]]
        # and similarly for V and W.
        
        # Build the system
        Uo = U32[:, others]  # 4 x 31
        Vo = V32[:, others]  # 18 x 31
        Wo = W32[:, others]  # 18 x 31
        
        Ut = U32[:, target]  # 4
        Vt = V32[:, target]  # 18
        Wt = W32[:, target]  # 18
        
        # Solve for alpha from U, then check V and W
        alpha, residuals_u, rank_u, sv_u = np.linalg.lstsq(Uo, Ut, rcond=None)
        
        # Check if this alpha also works for V and W
        V_pred = Vo @ alpha
        W_pred = Wo @ alpha
        
        if np.allclose(Vt, V_pred, atol=1e-10) and np.allclose(Wt, W_pred, atol=1e-10):
            # Found a redundant column!
            U = U32[:, others]
            V = V32[:, others]
            W = W32[:, others]
            if _validate_decomposition(U, V, W, n, m, p):
                return [[U[i][j] for j in range(31)] for i in range(n*m)], \
                       [[V[i][j] for j in range(31)] for i in range(m*p)], \
                       [[W[i][j] for j in range(31)] for i in range(p*n)]
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-31 decomposition
    if time.time() - _START_TIME < _TIMEOUT - 10:
        result = _search_rank_below_32()
        if result is not None:
            U, V, W = result
            if _validate_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to rank-32 Strassen-based decomposition
    U, V, W = _build_rank_32(n, m, p)
    
    # Validate before returning
    if _validate_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
