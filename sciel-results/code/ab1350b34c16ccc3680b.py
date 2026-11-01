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

def _try_rank_31():
    """
    Attempt to find a rank-31 decomposition.
    We use the rank-32 Strassen-based decomposition as a starting point
    and try to merge or eliminate one term.
    For 2x2x9, it's possible that rank 31 is achievable.
    We do a quick search over simple perturbations of the rank-32 solution.
    """
    n, m, p = 2, 2, 9
    rank = 31
    
    # Start from the rank-32 solution and try to drop one column by checking
    # if the remaining 31 still span the tensor. This is a quick heuristic.
    U32, V32, W32 = _build_rank_32(n, m, p)
    U32 = np.array(U32, dtype=float)
    V32 = np.array(V32, dtype=float)
    W32 = np.array(W32, dtype=float)
    
    # Try dropping each column and checking validity
    for drop in range(32):
        mask = [i for i in range(32) if i != drop]
        U = U32[:, mask]
        V = V32[:, mask]
        W = W32[:, mask]
        if _validate_decomposition(U, V, W, n, m, p):
            return [[U[i][j] for j in range(rank)] for i in range(n*m)], \
                   [[V[i][j] for j in range(rank)] for i in range(m*p)], \
                   [[W[i][j] for j in range(rank)] for i in range(p*n)]
    
    return None

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
    
    # Strassen formulas for 2x2x2:
    # P0 = (A00 + A11) * (B00 + B11)
    # P1 = (A10 + A11) * B00
    # P2 = A00 * (B01 - B11)
    # P3 = A11 * (B10 - B00)
    # P4 = (A00 + A01) * B11
    # P5 = (A10 - A00) * (B00 + B01)
    # P6 = (A01 - A11) * (B10 + B11)
    
    # C00 = P0 + P3 - P5 + P6
    # C01 = P2 + P4
    # C10 = P1 + P5
    # C11 = P0 - P1 + P2 + P4
    
    # U indices: i*m + j where i in {0,1}, j in {0,1}
    #   U[0] = A00, U[1] = A01, U[2] = A10, U[3] = A11
    # V indices: j*p + k where j in {0,1}, k in {0,1} for a 2x2 block
    #   For block b, V[2*b] = B00, V[2*b+1] = B01, V[2*b+2] = B10, V[2*b+3] = B11
    # W indices: k*n + i where k in {0,1}, i in {0,1}
    #   For block b, W[2*b] = C00, W[2*b+1] = C01, W[2*b+2] = C10, W[2*b+3] = C11
    
    # Strassen terms: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (U_index, coeff)
    # v_coeffs: list of (V_index_offset, coeff) - offset from block start
    # w_coeffs: list of (W_index_offset, coeff) - offset from block start
    
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
                v_idx = 2 * block + v_idx_off  # V index for this block
                V[v_idx][col] += coeff
            
            for w_idx_off, coeff in w_coeffs:
                w_idx = 2 * block + w_idx_off  # W index for this block
                W[w_idx][col] += coeff
    
    # Fill in schoolbook for column 8 (2x2x1 block)
    # C00 = A00*B0 + A01*B1 -> W[16] = C00 for k=8, i=0
    # C10 = A10*B0 + A11*B1 -> W[17] = C10 for k=8, i=1
    # V indices: 2*8+0 = 16 (B0 for row 0), 2*8+1 = 17 (B1 for row 0), 2*8+2 = 18 (B0 for row 1), 2*8+3 = 19 (B1 for row 1)
    # Wait, V has m*p = 2*9 = 18 rows, indices 0..17
    # V[j*p + k]: j in {0,1}, k in {0..8}
    # For k=8: V[0*9+8]=V[8], V[1*9+8]=V[17]
    # W[k*n + i]: k in {0..8}, i in {0,1}
    # For k=8: W[8*2+0]=W[16], W[8*2+1]=W[17]
    
    # Schoolbook terms for 2x2x1:
    # T0 = A00 * B0 -> C00
    # T1 = A01 * B1 -> C00
    # T2 = A10 * B0 -> C10
    # T3 = A11 * B1 -> C10
    
    schoolbook_terms = [
        # T0: A00 * B0 -> C00
        ([ (0, 1) ], [ (8, 1) ], [ (16, 1) ]),
        # T1: A01 * B1 -> C00
        ([ (1, 1) ], [ (17, 1) ], [ (16, 1) ]),
        # T2: A10 * B0 -> C10
        ([ (2, 1) ], [ (8, 1) ], [ (17, 1) ]),
        # T3: A11 * B1 -> C10
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

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-31 decomposition first
    if time.time() - _START_TIME < _TIMEOUT - 10:
        result = _try_rank_31()
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
