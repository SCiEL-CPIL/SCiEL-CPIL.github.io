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
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * rank for _ in range(p * n)]  # 18 x 32
    
    # Strassen formulas for 2x2x2
    # U indices: 0=a00, 1=a01, 2=a10, 3=a11
    # V local indices: 0=b00, 1=b01, 2=b10, 3=b11
    # W local indices: 0=c00, 1=c01, 2=c10, 3=c11
    
    strassen_terms = [
        # P0 = (A00 + A11) * (B00 + B11) -> C00 + C11 + (C10 - C01)... 
        # Actually: C00 = P0 - P1 - P4 + P6, C01 = P4 + P6, C10 = P1 + P5, C11 = P0 - P2 + P3 - P5
        # Let me use the standard Strassen:
        # P0 = (A00+A11)(B00+B11)
        # P1 = (A10+A11)B00
        # P2 = A00(B01-B11)
        # P3 = A11(B10-B00)
        # P4 = (A00+A01)B11
        # P5 = (A10-A00)(B00+B01)
        # P6 = (A01-A11)(B10+B11)
        # C00 = P0 - P1 - P4 + P6
        # C01 = P4 + P6
        # C10 = P1 + P5
        # C11 = P0 - P2 + P3 - P5
        
        # P0
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1), (1, 1), (2, 1)]),
        # P1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # P2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, -1), (3, 1)]),
        # P3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, -1)]),
        # P4
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
        # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, -1), (2, 1)]),
        # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)]),
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
                # v_idx_off: 0=b00, 1=b01, 2=b10, 3=b11
                # Global V index: V[j*p + k]
                # b00: j=0, k=2*block -> V[0*9 + 2*block]
                # b01: j=0, k=2*block+1 -> V[0*9 + 2*block + 1]
                # b10: j=1, k=2*block -> V[1*9 + 2*block]
                # b11: j=1, k=2*block+1 -> V[1*9 + 2*block + 1]
                row = v_idx_off // 2
                col_local = v_idx_off % 2
                v_idx = row * 9 + 2 * block + col_local
                V[v_idx][col] += coeff
            
            for w_idx_off, coeff in w_coeffs:
                # w_idx_off: 0=c00, 1=c01, 2=c10, 3=c11
                # Global W index: W[k*n + i]
                # c00: k=2*block, i=0 -> W[2*block*2 + 0]
                # c01: k=2*block, i=1 -> W[2*block*2 + 1]
                # c10: k=2*block+1, i=0 -> W[(2*block+1)*2 + 0]
                # c11: k=2*block+1, i=1 -> W[(2*block+1)*2 + 1]
                k_local = w_idx_off // 2
                i_local = w_idx_off % 2
                k_global = 2 * block + k_local
                w_idx = k_global * 2 + i_local
                W[w_idx][col] += coeff
    
    # Fill in schoolbook for column 8 (2x2x1 block)
    # V indices: V[0*9+8]=V[8] (b08), V[1*9+8]=V[17] (b18)
    # W indices: W[8*2+0]=W[16] (c08), W[8*2+1]=W[17] (c18)
    
    schoolbook_terms = [
        # T0: A00 * B[0,8] -> C[0,8]
        ([(0, 1)], [(8, 1)], [(16, 1)]),
        # T1: A01 * B[1,8] -> C[0,8]
        ([(1, 1)], [(17, 1)], [(16, 1)]),
        # T2: A10 * B[0,8] -> C[1,8]
        ([(2, 1)], [(8, 1)], [(17, 1)]),
        # T3: A11 * B[1,8] -> C[1,8]
        ([(3, 1)], [(17, 1)], [(17, 1)]),
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
    
    # Try to build rank-32 Strassen-based decomposition
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
