import numpy as np
from itertools import product
import time

# EVOLVE-BLOCK-START
DIMENSIONS = [2, 4, 5]
DEADLINE_SECONDS = 145

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    D = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
    return np.allclose(D, T, atol=tol)

def _build_block_strassen_2x4x5():
    """
    Build a rank-36 decomposition using Block-Strassen structure.
    
    The 2x4x5 multiplication tensor is decomposed as:
    - 4 Strassen 2x2x2 sub-problems (j-blocks {0,1},{2,3} x k-blocks {0,1},{2,3})
    - 2 schoolbook 2x2x1 sub-problems (j-blocks {0,1},{2,3} x k={4})
    
    Total rank: 4*7 + 2*4 = 36
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    def strassen_2x2x2(j_base, k_base, U, V, W, col_offset):
        """Add 7 Strassen terms for the 2x2x2 sub-problem at j_base, k_base."""
        terms = []
        
        # M1 = (A11 - A00) * (B00 + B11)
        # A11 = A[1, j_base+1], A00 = A[0, j_base+0]
        # B00 = B[j_base+0, k_base+0], B11 = B[j_base+1, k_base+1]
        # C00 += M1, C11 += M1
        terms.append({
            'U': {1*n + (j_base+1): 1, 0*n + (j_base+0): -1},
            'V': {(j_base+0)*p + (k_base+0): 1, (j_base+1)*p + (k_base+1): 1},
            'W': {0*p + 0: 1, 1*p + 1: 1},
        })
        
        # M2 = (A00 + A11) * B10
        # A00 = A[0, j_base+0], A11 = A[1, j_base+1]
        # B10 = B[j_base+1, k_base+0]
        # C10 += M2
        terms.append({
            'U': {0*n + (j_base+0): 1, 1*n + (j_base+1): 1},
            'V': {(j_base+1)*p + (k_base+0): 1},
            'W': {0*p + 1: 1},
        })
        
        # M3 = A10 * (B01 - B11)
        # A10 = A[1, j_base+0]
        # B01 = B[j_base+0, k_base+1], B11 = B[j_base+1, k_base+1]
        # C01 += M3
        terms.append({
            'U': {1*n + (j_base+0): 1},
            'V': {(j_base+0)*p + (k_base+1): 1, (j_base+1)*p + (k_base+1): -1},
            'W': {1*p + 0: 1},
        })
        
        # M4 = (A11 - A10) * (B00 + B10)
        # A11 = A[1, j_base+1], A10 = A[1, j_base+0]
        # B00 = B[j_base+0, k_base+0], B10 = B[j_base+1, k_base+0]
        # C00 += M4
        terms.append({
            'U': {1*n + (j_base+1): 1, 1*n + (j_base+0): -1},
            'V': {(j_base+0)*p + (k_base+0): 1, (j_base+1)*p + (k_base+0): 1},
            'W': {0*p + 0: 1},
        })
        
        # M5 = A00 * (B11 - B00)
        # A00 = A[0, j_base+0]
        # B11 = B[j_base+1, k_base+1], B00 = B[j_base+0, k_base+0]
        # C10 += M5
        terms.append({
            'U': {0*n + (j_base+0): 1},
            'V': {(j_base+1)*p + (k_base+1): 1, (j_base+0)*p + (k_base+0): -1},
            'W': {0*p + 1: 1},
        })
        
        # M6 = (A01 - A11) * (B10 + B11)
        # A01 = A[0, j_base+1], A11 = A[1, j_base+1]
        # B10 = B[j_base+1, k_base+0], B11 = B[j_base+1, k_base+1]
        # C11 += M6
        terms.append({
            'U': {0*n + (j_base+1): 1, 1*n + (j_base+1): -1},
            'V': {(j_base+1)*p + (k_base+0): 1, (j_base+1)*p + (k_base+1): 1},
            'W': {1*p + 1: 1},
        })
        
        # M7 = (A00 + A01) * B01
        # A00 = A[0, j_base+0], A01 = A[0, j_base+1]
        # B01 = B[j_base+0, k_base+1]
        # C01 += M7
        terms.append({
            'U': {0*n + (j_base+0): 1, 0*n + (j_base+1): 1},
            'V': {(j_base+0)*p + (k_base+1): 1},
            'W': {1*p + 0: 1},
        })
        
        for term in terms:
            for a_idx, coeff in term['U'].items():
                U[a_idx][col_offset] += coeff
            for b_idx, coeff in term['V'].items():
                V[b_idx][col_offset] += coeff
            for c_idx, coeff in term['W'].items():
                W[c_idx][col_offset] += coeff
            col_offset += 1
        
        return col_offset
    
    def schoolbook_2x2x1(j_base, k_global, U, V, W, col_offset):
        """Add 4 schoolbook terms for the 2x2x1 sub-problem at j_base, k_global."""
        for i in range(2):
            for jl in range(2):
                j_global = j_base + jl
                U[i*m + j_global][col_offset] = 1
                V[j_global*p + k_global][col_offset] = 1
                W[k_global*n + i][col_offset] = 1
                col_offset += 1
        
        return col_offset
    
    # Build the decomposition
    total_rank = 4*7 + 2*4  # 36
    U = [[0] * total_rank for _ in range(n*m)]
    V = [[0] * total_rank for _ in range(m*p)]
    W = [[0] * total_rank for _ in range(p*n)]
    
    col = 0
    # Four 2x2x2 Strassen blocks
    for j_block in range(2):
        for k_block in range(2):
            col = strassen_2x2x2(j_block*2, k_block*2, U, V, W, col)
    
    # Two 2x2x1 schoolbook blocks for k=4
    for j_block in range(2):
        col = schoolbook_2x2x1(j_block*2, 4, U, V, W, col)
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    
    # Try the Block-Strassen construction (rank 36)
    try:
        result = _build_block_strassen_2x4x5()
        if _verify(result['U'], result['V'], result['W']):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 40)
    result = _build_schoolbook()
    if _verify(result['U'], result['V'], result['W']):
        return result
    
    raise ValueError("No valid decomposition found")
# EVOLVE-BLOCK-END
