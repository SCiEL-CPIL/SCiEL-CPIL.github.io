import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Rank-68 decomposition for 3x3 * 3x8 matrix multiplication.
    
    Strategy: Split the 8 columns into 4 blocks of 2. For each 3x3 * 3x2 block,
    split the 3x3 matrix into 2x2, 2x1, 1x2, 1x1 sub-blocks:
      - 2x2 * 2x2: use Strassen (rank 7)
      - 2x1 * 1x2: schoolbook (rank 4)
      - 1x2 * 2x2: schoolbook (rank 4)
      - 1x1 * 1x2: schoolbook (rank 2)
    Total per 2-column block: 17. Four blocks: 68.
    
    Indexing:
      U: row i*m+j (i=0..2, j=0..2) -> index i*3+j
      V: row j*p+k (j=0..2, k=0..7) -> index j*8+k
      W: row k*n+i (k=0..7, i=0..2) -> index k*3+i
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = 68
    
    U = np.zeros((n * m, rank), dtype=float)  # 9 x 68
    V = np.zeros((m * p, rank), dtype=float)  # 24 x 68
    W = np.zeros((p * n, rank), dtype=float)  # 24 x 68
    
    col = 0  # current column index in U, V, W
    
    # Strassen 2x2*2x2: 7 products
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B11+B01)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen U coefficients (for A matrix 2x2):
    # Rows: A00, A01, A10, A11 (local indices 0,1,2,3)
    strassen_U = [
        [1, 0, 1, 0],  # A00: M1=+1, M3=+1
        [0, 1, 0, 1],  # A01: M5=+1, M7=+1
        [0, 0, 1, -1], # A10: M2=+1, M6=-1  wait let me redo
    ]
    # Let me carefully list:
    # M1: A00(+1), A11(+1)
    # M2: A10(+1), A11(+1)
    # M3: A00(+1)
    # M4: A11(+1)
    # M5: A00(+1), A01(+1)
    # M6: A10(+1), A00(-1)
    # M7: A01(+1), A11(-1)
    
    # U for A (2x2 block, rows 0-3):
    # Row 0 (A00): M1=+1, M3=+1, M5=+1, M6=-1
    # Row 1 (A01): M5=+1, M7=+1
    # Row 2 (A10): M2=+1, M6=+1
    # Row 3 (A11): M1=+1, M2=+1, M4=+1, M7=-1
    
    # V for B (2x2 block, rows 0-3):
    # Row 0 (B00): M1=+1, M2=+1, M4=-1, M6=+1
    # Row 1 (B01): M4=+1, M7=+1
    # Row 2 (B10): M3=-1, M6=+1
    # Row 3 (B11): M1=+1, M3=+1, M5=+1, M7=+1
    
    # W for C (2x2 block, rows 0-3):
    # Row 0 (C00): M1=+1, M4=+1, M5=-1, M7=+1
    # Row 1 (C01): M3=+1, M5=+1
    # Row 2 (C10): M2=+1, M4=+1
    # Row 3 (C11): M1=+1, M2=-1, M3=+1, M6=+1
    
    for block in range(4):  # 4 blocks of 2 columns each
        k_base = block * 2  # starting k index for this block
        
        # --- Strassen 2x2*2x2 part (rank 7) ---
        # A sub-block: rows i=0,1; cols j=0,1
        # B sub-block: rows j=0,1; cols k=k_base, k_base+1
        # C sub-block: rows i=0,1; cols k=k_base, k_base+1
        
        # A indices (global): i*3+j
        # A00 = (0,0) -> 0, A01 = (0,1) -> 1, A10 = (1,0) -> 3, A11 = (1,1) -> 4
        # B indices (global): j*8+k
        # B00 = (0,k_base) -> k_base, B01 = (0,k_base+1) -> k_base+1
        # B10 = (1,k_base) -> 8+k_base, B11 = (1,k_base+1) -> 8+k_base+1
        # C indices (global): k*3+i
        # C00 = (k_base,0) -> 3*k_base, C01 = (k_base+1,0) -> 3*k_base+1
        # C10 = (k_base,1) -> 3*k_base+3, C11 = (k_base+1,1) -> 3*k_base+4
        
        a_rows = [0, 1, 3, 4]  # A00, A01, A10, A11
        b_rows = [k_base, k_base + 1, 8 + k_base, 8 + k_base + 1]  # B00, B01, B10, B11
        c_rows = [3 * k_base, 3 * k_base + 1, 3 * k_base + 3, 3 * k_base + 4]  # C00, C01, C10, C11
        
        # Strassen 7 terms
        # M1: (A00+A11)*(B00+B11)
        U[a_rows[0], col] += 1; U[a_rows[3], col] += 1
        V[b_rows[0], col] += 1; V[b_rows[3], col] += 1
        W[c_rows[0], col] += 1; W[c_rows[3], col] += 1
        col += 1
        
        # M2: (A10+A11)*B00
        U[a_rows[2], col] += 1; U[a_rows[3], col] += 1
        V[b_rows[0], col] += 1
        W[c_rows[2], col] += 1
        col += 1
        
        # M3: A00*(B10-B11)
        U[a_rows[0], col] += 1
        V[b_rows[2], col] += 1; V[b_rows[3], col] += -1
        W[c_rows[1], col] += 1
        col += 1
        
        # M4: A11*(B01-B00)
        U[a_rows[3], col] += 1
        V[b_rows[1], col] += 1; V[b_rows[0], col] += -1
        W[c_rows[0], col] += 1; W[c_rows[2], col] += 1
        col += 1
        
        # M5: (A00+A01)*B11
        U[a_rows[0], col] += 1; U[a_rows[1], col] += 1
        V[b_rows[3], col] += 1
        W[c_rows[0], col] += -1; W[c_rows[1], col] += 1
        col += 1
        
        # M6: (A10-A00)*(B00+B10)
        U[a_rows[2], col] += 1; U[a_rows[0], col] += -1
        V[b_rows[0], col] += 1; V[b_rows[2], col] += 1
        W[c_rows[3], col] += 1
        col += 1
        
        # M7: (A01-A11)*(B11+B01)
        U[a_rows[1], col] += 1; U[a_rows[3], col] += -1
        V[b_rows[3], col] += 1; V[b_rows[1], col] += 1
        W[c_rows[0], col] += 1
        col += 1
        
        # --- 2x1 * 1x2 part (rank 4) ---
        # A: rows i=0,1; col j=2
        # B: row j=2; cols k=k_base, k_base+1
        # C: rows i=0,1; cols k=k_base, k_base+1
        # C[i,k] = A[i,2] * B[2,k]
        
        # Term 1: A02 * B20 -> C00
        U[2, col] += 1  # A(0,2)
        V[16 + k_base, col] += 1  # B(2, k_base)
        W[3 * k_base, col] += 1  # C(k_base, 0)
        col += 1
        
        # Term 2: A02 * B21 -> C01
        U[2, col] += 1  # A(0,2)
        V[16 + k_base + 1, col] += 1  # B(2, k_base+1)
        W[3 * k_base + 1, col] += 1  # C(k_base+1, 0)
        col += 1
        
        # Term 3: A12 * B20 -> C10
        U[5, col] += 1  # A(1,2)
        V[16 + k_base, col] += 1  # B(2, k_base)
        W[3 * k_base + 3, col] += 1  # C(k_base, 1)
        col += 1
        
        # Term 4: A12 * B21 -> C11
        U[5, col] += 1  # A(1,2)
        V[16 + k_base + 1, col] += 1  # B(2, k_base+1)
        W[3 * k_base + 4, col] += 1  # C(k_base+1, 1)
        col += 1
        
        # --- 1x2 * 2x2 part (rank 4) ---
        # A: row i=2; cols j=0,1
        # B: rows j=0,1; cols k=k_base, k_base+1
        # C: row i=2; cols k=k_base, k_base+1
        # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
        
        # Term 1: A20 * B00 -> C20
        U[6, col] += 1  # A(2,0)
        V[k_base, col] += 1  # B(0, k_base)
        W[3 * k_base + 6, col] += 1  # C(k_base, 2)
        col += 1
        
        # Term 2: A20 * B01 -> C21
        U[6, col] += 1  # A(2,0)
        V[k_base + 1, col] += 1  # B(0, k_base+1)
        W[3 * k_base + 7, col] += 1  # C(k_base+1, 2)
        col += 1
        
        # Term 3: A21 * B10 -> C20
        U[7, col] += 1  # A(2,1)
        V[8 + k_base, col] += 1  # B(1, k_base)
        W[3 * k_base + 6, col] += 1  # C(k_base, 2)
        col += 1
        
        # Term 4: A21 * B11 -> C21
        U[7, col] += 1  # A(2,1)
        V[8 + k_base + 1, col] += 1  # B(1, k_base+1)
        W[3 * k_base + 7, col] += 1  # C(k_base+1, 2)
        col += 1
        
        # --- 1x1 * 1x2 part (rank 2) ---
        # A: row i=2; col j=2
        # B: row j=2; cols k=k_base, k_base+1
        # C: row i=2; cols k=k_base, k_base+1
        # C[2,k] += A[2,2] * B[2,k]
        
        # Term 1: A22 * B20 -> C20
        U[8, col] += 1  # A(2,2)
        V[16 + k_base, col] += 1  # B(2, k_base)
        W[3 * k_base + 6, col] += 1  # C(k_base, 2)
        col += 1
        
        # Term 2: A22 * B21 -> C21
        U[8, col] += 1  # A(2,2)
        V[16 + k_base + 1, col] += 1  # B(2, k_base+1)
        W[3 * k_base + 7, col] += 1  # C(k_base+1, 2)
        col += 1
    
    assert col == 68, f"Expected col=68, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
