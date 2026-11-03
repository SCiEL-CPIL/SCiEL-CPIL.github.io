import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    # Split m=4 into m1=2, m2=2
    # For each 2×2×7 sub-problem, split p=7 into 2+2+2+1
    # 2×2×2 uses Strassen (rank 7), 2×2×1 uses schoolbook (rank 4)
    # Total per 2×2×7: 7+7+7+4 = 25
    # Total for 2×4×7: 25+25 = 50
    
    total_rank = 50
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    # Helper: add Strassen 2×2×2 decomposition
    # A_sub is 2×2 (indices i in {0,1}, j in {0,1} of local block)
    # B_sub is 2×2 (indices j in {0,1} of local block, k in {0,1} of local block)
    # Global indices: A row i, col (m_offset + j); B row (m_offset + j), col (p_offset + k)
    # C row i, col (p_offset + k)
    def add_strassen_222(m_offset, p_offset):
        nonlocal col
        # Strassen for 2×2×2:
        # M1 = A00*B00
        # M2 = A00*B01
        # M3 = A10*B10
        # M4 = A11*B11
        # M5 = (A00+A11)*(B00+B11)
        # M6 = (A10+A11)*(B10+B11)
        # M7 = (A00+A10)*(B00+B10)
        # C00 = M1 + M5 - M6
        # C01 = M2 + M6 - M7
        # C10 = M3 + M7 - M5
        # C11 = M4 + M5 - M1
        
        # Term 1: M1 = A00*B00
        U[0 * m + (m_offset + 0), col] = 1.0
        V[(m_offset + 0) * p + (p_offset + 0), col] = 1.0
        W[(p_offset + 0) * n + 0, col] = 1.0
        col += 1
        
        # Term 2: M2 = A00*B01
        U[0 * m + (m_offset + 0), col] = 1.0
        V[(m_offset + 0) * p + (p_offset + 1), col] = 1.0
        W[(p_offset + 1) * n + 0, col] = 1.0
        col += 1
        
        # Term 3: M3 = A10*B10
        U[1 * m + (m_offset + 0), col] = 1.0
        V[(m_offset + 0) * p + (p_offset + 0), col] = 1.0
        W[(p_offset + 0) * n + 1, col] = 1.0
        col += 1
        
        # Term 4: M4 = A11*B11
        U[1 * m + (m_offset + 1), col] = 1.0
        V[(m_offset + 1) * p + (p_offset + 1), col] = 1.0
        W[(p_offset + 1) * n + 1, col] = 1.0
        col += 1
        
        # Term 5: M5 = (A00+A11)*(B00+B11)
        U[0 * m + (m_offset + 0), col] = 1.0
        U[1 * m + (m_offset + 1), col] = 1.0
        V[(m_offset + 0) * p + (p_offset + 0), col] = 1.0
        V[(m_offset + 1) * p + (p_offset + 1), col] = 1.0
        W[(p_offset + 0) * n + 0, col] = 1.0
        W[(p_offset + 1) * n + 1, col] = 1.0
        col += 1
        
        # Term 6: M6 = (A10+A11)*(B10+B11)
        U[1 * m + (m_offset + 0), col] = 1.0
        U[1 * m + (m_offset + 1), col] = 1.0
        V[(m_offset + 0) * p + (p_offset + 0), col] = 1.0
        V[(m_offset + 1) * p + (p_offset + 1), col] = 1.0
        W[(p_offset + 0) * n + 1, col] = -1.0
        W[(p_offset + 1) * n + 1, col] = -1.0
        col += 1
        
        # Term 7: M7 = (A00+A10)*(B00+B10)
        U[0 * m + (m_offset + 0), col] = 1.0
        U[1 * m + (m_offset + 0), col] = 1.0
        V[(m_offset + 0) * p + (p_offset + 0), col] = 1.0
        V[(m_offset + 0) * p + (p_offset + 1), col] = 1.0
        W[(p_offset + 0) * n + 0, col] = -1.0
        W[(p_offset + 1) * n + 0, col] = -1.0
        col += 1
    
    # Helper: add schoolbook 2×2×1 decomposition
    def add_schoolbook_221(m_offset, p_offset):
        nonlocal col
        # C00 = A00*B00 + A10*B10  (but wait, for 2×2×1, B is 2×1)
        # Actually: C[i, k] = sum_j A[i,j] B[j,k]
        # For 2×2×1: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        #            C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # Schoolbook: 4 terms
        # Term 1: A00*B00
        U[0 * m + (m_offset + 0), col] = 1.0
        V[(m_offset + 0) * p + (p_offset + 0), col] = 1.0
        W[(p_offset + 0) * n + 0, col] = 1.0
        col += 1
        
        # Term 2: A01*B10
        U[0 * m + (m_offset + 1), col] = 1.0
        V[(m_offset + 1) * p + (p_offset + 0), col] = 1.0
        W[(p_offset + 0) * n + 0, col] = 1.0
        col += 1
        
        # Term 3: A10*B00
        U[1 * m + (m_offset + 0), col] = 1.0
        V[(m_offset + 0) * p + (p_offset + 0), col] = 1.0
        W[(p_offset + 0) * n + 1, col] = 1.0
        col += 1
        
        # Term 4: A11*B10
        U[1 * m + (m_offset + 1), col] = 1.0
        V[(m_offset + 1) * p + (p_offset + 0), col] = 1.0
        W[(p_offset + 0) * n + 1, col] = 1.0
        col += 1
    
    # First 2×2×7 block: m_offset=0
    # Split p=7 into 2+2+2+1
    add_strassen_222(0, 0)   # p_offset=0, covers k=0,1
    add_strassen_222(0, 2)   # p_offset=2, covers k=2,3
    add_strassen_222(0, 4)   # p_offset=4, covers k=4,5
    add_schoolbook_221(0, 6) # p_offset=6, covers k=6
    
    # Second 2×2×7 block: m_offset=2
    add_strassen_222(2, 0)   # p_offset=0, covers k=0,1
    add_strassen_222(2, 2)   # p_offset=2, covers k=2,3
    add_strassen_222(2, 4)   # p_offset=4, covers k=4,5
    add_schoolbook_221(2, 6) # p_offset=6, covers k=6
    
    assert col == total_rank, f"col={col}, total_rank={total_rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
