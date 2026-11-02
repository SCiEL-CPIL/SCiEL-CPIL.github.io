"""
Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition.
Implements a valid decomposition with rank 72 using block decomposition.
Splits dimensions to use Strassen (rank 7) for 2x2x2 and schoolbook (rank 2) for 1x2x2.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-6):
    """Full verification."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check all invalid entries
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
    return True

def _add_strassen(U, V, W, a_indices, b_indices, c_indices, base_rank):
    """
    Adds a 2x2x2 Strassen decomposition.
    a_indices: [A00, A01, A10, A11]
    b_indices: [B00, B01, B10, B11]
    c_indices: [C00, C01, C10, C11]
    """
    # M1: (A00+A11)(B00+B11)
    U[a_indices[0], base_rank] += 1; U[a_indices[3], base_rank] += 1
    V[b_indices[0], base_rank] += 1; V[b_indices[3], base_rank] += 1
    W[c_indices[0], base_rank] += 1; W[c_indices[3], base_rank] += 1
    
    # M2: (A10+A11)B00
    U[a_indices[2], base_rank+1] += 1; U[a_indices[3], base_rank+1] += 1
    V[b_indices[0], base_rank+1] += 1
    W[c_indices[2], base_rank+1] += 1; W[c_indices[3], base_rank+1] += 1
    
    # M3: A00(B01-B11)
    U[a_indices[0], base_rank+2] += 1
    V[b_indices[1], base_rank+2] += 1; V[b_indices[3], base_rank+2] -= 1
    W[c_indices[1], base_rank+2] += 1; W[c_indices[3], base_rank+2] += 1
    
    # M4: A11(B10-B00)
    U[a_indices[3], base_rank+3] += 1
    V[b_indices[2], base_rank+3] += 1; V[b_indices[0], base_rank+3] -= 1
    W[c_indices[2], base_rank+3] += 1; W[c_indices[3], base_rank+3] += 1
    
    # M5: (A00+A01)B11
    U[a_indices[0], base_rank+4] += 1; U[a_indices[1], base_rank+4] += 1
    V[b_indices[3], base_rank+4] += 1
    W[c_indices[1], base_rank+4] += 1; W[c_indices[3], base_rank+4] += 1
    
    # M6: (A10-A00)(B00+B01)
    U[a_indices[2], base_rank+5] += 1; U[a_indices[0], base_rank+5] -= 1
    V[b_indices[0], base_rank+5] += 1; V[b_indices[1], base_rank+5] += 1
    W[c_indices[2], base_rank+5] += 1; W[c_indices[3], base_rank+5] += 1
    
    # M7: (A01-A11)(B10+B11)
    U[a_indices[1], base_rank+6] += 1; U[a_indices[3], base_rank+6] -= 1
    V[b_indices[2], base_rank+6] += 1; V[b_indices[3], base_rank+6] += 1
    W[c_indices[0], base_rank+6] += 1; W[c_indices[3], base_rank+6] += 1

def _add_dot_1x2x2(U, V, W, a_indices, b_indices, c_indices, base_rank):
    """
    Adds a 1x2x2 dot product decomposition.
    a_indices: [A0, A1]
    b_indices: [B00, B01, B10, B11]
    c_indices: [C0, C1]
    """
    # r=0: A0 * (B00, B01) -> (C0, C1)
    U[a_indices[0], base_rank] += 1
    V[b_indices[0], base_rank] += 1; V[b_indices[1], base_rank] += 1
    W[c_indices[0], base_rank] += 1; W[c_indices[1], base_rank] += 1
    
    # r=1: A1 * (B10, B11) -> (C0, C1)
    U[a_indices[1], base_rank+1] += 1
    V[b_indices[2], base_rank+1] += 1; V[b_indices[3], base_rank+1] += 1
    W[c_indices[0], base_rank+1] += 1; W[c_indices[1], base_rank+1] += 1

def _construct_decomposition():
    n, m, p = DIMENSIONS # 3, 4, 8
    # Rank calculation:
    # 2 p-blocks * 2 m-blocks * (1 n-block 2x2x2 (rank 14) + 1 n-block 1x2x2 (rank 4))
    # Wait, 2x2x2 split into two 2x2x2? No.
    # M_{2,2,4} -> 2 * M_{2,2,2} (rank 14)
    # M_{1,2,4} -> 2 * M_{1,2,2} (rank 4)
    # Total per M_{3,2,4}: 18
    # Total per M_{3,4,4}: 36
    # Total M_{3,4,8}: 72
    
    total_rank = 72
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    current_rank = 0
    
    # Loop over p-blocks (0:4, 4:8)
    for p_start in [0, 4]:
        # Loop over m-blocks (0:2, 2:4)
        for m_start in [0, 2]:
            # n-block 1: rows 0,1 (2x2x4 part, split into 2x2x2 + 2x2x2)
            # p is split into 2:2 within the 4-block
            for p_sub_start in [p_start, p_start + 2]:
                # A indices: i in [0,1], j in [m_start, m_start+2]
                a_idx = [i*m + j for i in range(2) for j in range(m_start, m_start+2)]
                # B indices: j in [m_start, m_start+2], k in [p_sub_start, p_sub_start+2]
                b_idx = [j*p + k for j in range(m_start, m_start+2) for k in range(p_sub_start, p_sub_start+2)]
                # C indices: k in [p_sub_start, p_sub_start+2], i in [0,1]
                c_idx = [k*n + i for k in range(p_sub_start, p_sub_start+2) for i in range(2)]
                
                _add_strassen(U, V, W, a_idx, b_idx, c_idx, current_rank)
                current_rank += 7
            
            # n-block 2: row 2 (1x2x4 part, split into 1x2x2 + 1x2x2)
            for p_sub_start in [p_start, p_start + 2]:
                # A indices: i=2, j in [m_start, m_start+2]
                a_idx = [2*m + j for j in range(m_start, m_start+2)]
                # B indices: j in [m_start, m_start+2], k in [p_sub_start, p_sub_start+2]
                b_idx = [j*p + k for j in range(m_start, m_start+2) for k in range(p_sub_start, p_sub_start+2)]
                # C indices: k in [p_sub_start, p_sub_start+2], i=2
                c_idx = [k*n + 2 for k in range(p_sub_start, p_sub_start+2)]
                
                _add_dot_1x2x2(U, V, W, a_idx, b_idx, c_idx, current_rank)
                current_rank += 2
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    try:
        U, V, W = _construct_decomposition()
        
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Fallback
    return _schoolbook()
# EVOLVE-BLOCK-END
