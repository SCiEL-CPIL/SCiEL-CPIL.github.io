import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Rank-66 exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    Strategy:
    - Partition A (3x4) rows as 1+2, cols as 2+2
    - Partition B (4x6) rows as 2+2, cols as 2+2+2
    - Use Strassen (rank 7) for 2x2*2x2 blocks
    - Use schoolbook (rank 4) for 1x2*2x2 blocks
    - Total: 6*(1x2*2x2) + 6*(2x2*2x2) = 6*4 + 6*7 = 24 + 42 = 66
    
    The output factor W follows the trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = 3, 4, 6
    rank = 66
    
    U = np.zeros((n * m, rank), dtype=int)   # (12, 66)
    V = np.zeros((m * p, rank), dtype=int)   # (24, 66)
    W = np.zeros((p * n, rank), dtype=int)   # (18, 66)
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add Strassen 2x2*2x2 block (rank 7) at given offsets."""
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
        
        if time.time() > deadline:
            return False
        return True

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 1x2*2x2 block (rank 4) at given offsets."""
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        # M1: A0*B00
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        # M2: A1*B10
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        # M3: A0*B01
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        # M4: A1*B11
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1
        
        if time.time() > deadline:
            return False
        return True

    # Build decomposition:
    # Row 0 of C (1x6) = A[0,0:2]*B[0:2,:] + A[0,2:4]*B[2:4,:]
    # C[0,0:2]
    if not add_1x2_2x2(0, 0, 0, 0, 0, 0): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    if not add_1x2_2x2(0, 2, 2, 0, 0, 0): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # C[0,2:4]
    if not add_1x2_2x2(0, 0, 0, 2, 0, 2): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    if not add_1x2_2x2(0, 2, 2, 2, 0, 2): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # C[0,4:6]
    if not add_1x2_2x2(0, 0, 0, 4, 0, 4): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    if not add_1x2_2x2(0, 2, 2, 4, 0, 4): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Rows 1:3 of C (2x6) = A[1:3,0:2]*B[0:2,:] + A[1:3,2:4]*B[2:4,:]
    # C[1:3,0:2]
    if not add_strassen(1, 0, 0, 0, 1, 0): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    if not add_strassen(1, 2, 2, 0, 1, 0): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # C[1:3,2:4]
    if not add_strassen(1, 0, 0, 2, 1, 2): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    if not add_strassen(1, 2, 2, 2, 1, 2): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # C[1:3,4:6]
    if not add_strassen(1, 0, 0, 4, 1, 4): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    if not add_strassen(1, 2, 2, 4, 1, 4): return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    assert col == 66, f"Expected 66 columns, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
