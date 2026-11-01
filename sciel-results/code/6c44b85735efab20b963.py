# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition for 2x3x4
    # Split B into two 3x2 blocks
    # Each 2x3x2 block has rank 11 (7 from Strassen + 4 from outer product)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val

    # We need to map the logical indices to the flattened U, V, W indices
    # U is n*m = 6. Index: i*m + j
    # V is m*p = 12. Index: j*p + k
    # W is p*n = 8. Index: k*n + i
    
    # Block 1: B cols 0,1
    # Strassen part (rank 7)
    # A cols 0,1. B rows 0,1, cols 0,1.
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 1, 0, 1, 0, 1, 0) # A00, B00, W00 (k=0,i=0)
    add_term(1, 1, 0, 1, 0, 1, 0) # A10? No, A is 2x3.
    # Let's be precise.
    # U index: i*m + j. i in 0..1, j in 0..2.
    # V index: j*p + k. j in 0..2, k in 0..3.
    # W index: k*n + i. k in 0..3, i in 0..1.
    
    # Strassen for 2x2x2 (A00, A01, A10, A11) x (B00, B01, B10, B11)
    # A00 is U[0], A01 is U[1], A10 is U[3], A11 is U[4]
    # B00 is V[0], B01 is V[1], B10 is V[4], B11 is V[5]
    # W00 is W[0], W01 is W[1], W10 is W[2], W11 is W[3]
    
    # M1: (A00+A11)(B00+B11)
    add_term(0, 1, 0, 1, 0, 1, 0)
    add_term(4, 1, 0, 1, 0, 1, 0)
    add_term(0, 1, 5, 1, 0, 1, 0)
    add_term(4, 1, 5, 1, 0, 1, 0)
    # Wait, add_term overwrites. I need to accumulate.
    # Let's use a different approach. Build lists of (idx, val) for each col.
    
    # Re-initialize
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Block 1: B cols 0,1
    # Strassen (cols 0,1 of B)
    # M1
    add(0, 1, 0, 1, 0, 1, 0)
    add(4, 1, 0, 1, 0, 1, 0)
    add(0, 1, 5, 1, 0, 1, 0)
    add(4, 1, 5, 1, 0, 1, 0)
    # M2: (A10+A11)B00
    add(3, 1, 0, 1, 0, 1, 1)
    add(4, 1, 0, 1, 0, 1, 1)
    # M3: A00(B10-B11)
    add(0, 1, 4, 1, 0, 1, 2)
    add(0, 1, 5, -1, 0, 1, 2)
    # M4: A11(B01-B00)
    add(4, 1, 1, 1, 0, 1, 3)
    add(4, 1, 0, -1, 0, 1, 3)
    # M5: (A00+A01)B11
    add(0, 1, 5, 1, 0, 1, 4)
    add(1, 1, 5, 1, 0, 1, 4)
    # M6: (A10-A00)(B00+B10)
    add(3, 1, 0, 1, 0, 1, 5)
    add(0, -1, 0, 1, 0, 1, 5)
    add(3, 1, 4, 1, 0, 1, 5)
    add(0, -1, 4, 1, 0, 1, 5)
    # M7: (A01-A11)(B11+B01)
    add(1, 1, 5, 1, 0, 1, 6)
    add(4, -1, 5, 1, 0, 1, 6)
    add(1, 1, 1, 1, 0, 1, 6)
    add(4, -1, 1, 1, 0, 1, 6)
    
    # W updates for Block 1 Strassen
    # C00 = M1 + M4 - M5 + M7
    add(0, 0, 0, 0, 0, 1, 0) # M1
    add(0, 0, 0, 0, 0, 1, 3) # M4
    add(0, 0, 0, 0, 0, -1, 4) # -M5
    add(0, 0, 0, 0, 0, 1, 6) # M7
    
    # C01 = M3 + M5
    add(0, 0, 0, 0, 1, 1, 2) # M3
    add(0, 0, 0, 0, 1, 1, 4) # M5
    
    # C10 = M2 + M4
    add(0, 0, 0, 0, 2, 1, 1) # M2
    add(0, 0, 0, 0, 2, 1, 3) # M4
    
    # C11 = M1 - M2 + M3 + M6
    add(0, 0, 0, 0, 3, 1, 0) # M1
    add(0, 0, 0, 0, 3, -1, 1) # -M2
    add(0, 0, 0, 0, 3, 1, 2) # M3
    add(0, 0, 0, 0, 3, 1, 5) # M6

    # Outer product part for Block 1 (A col 2, B row 2)
    # A col 2 is U[2] (i=0,j=2) and U[5] (i=1,j=2)
    # B row 2 is V[8] (j=2,k=0) and V[9] (j=2,k=1)
    # M8: A02 B20
    add(2, 1, 8, 1, 0, 1, 7)
    # M9: A02 B21
    add(2, 1, 9, 1, 1, 1, 8)
    # M10: A12 B20
    add(5, 1, 8, 1, 2, 1, 9)
    # M11: A12 B21
    add(5, 1, 9, 1, 3, 1, 10)
    
    # Block 2: B cols 2,3
    # Strassen part (A cols 0,1, B rows 0,1, cols 2,3)
    # B02 is V[2], B03 is V[3], B12 is V[6], B13 is V[7]
    # W indices: k=2,3. W[4], W[5], W[6], W[7]
    
    # M12 (M1 of block 2)
    add(0, 1, 2, 1, 4, 1, 11)
    add(4, 1, 2, 1, 4, 1, 11)
    add(0, 1, 7, 1, 4, 1, 11)
    add(4, 1, 7, 1, 4, 1, 11)
    
    # M13 (M2)
    add(3, 1, 2, 1, 4, 1, 12)
    add(4, 1, 2, 1, 4, 1, 12)
    
    # M14 (M3)
    add(0, 1, 6, 1, 4, 1, 13)
    add(0, 1, 7, -1, 4, 1, 13)
    
    # M15 (M4)
    add(4, 1, 3, 1, 4, 1, 14)
    add(4, 1, 2, -1, 4, 1, 14)
    
    # M16 (M5)
    add(0, 1, 7, 1, 4, 1, 15)
    add(1, 1, 7, 1, 4, 1, 15)
    
    # M17 (M6)
    add(3, 1, 2, 1, 4, 1, 16)
    add(0, -1, 2, 1, 4, 1, 16)
    add(3, 1, 6, 1, 4, 1, 16)
    add(0, -1, 6, 1, 4, 1, 16)
    
    # M18 (M7)
    add(1, 1, 7, 1, 4, 1, 17)
    add(4, -1, 7, 1, 4, 1, 17)
    add(1, 1, 3, 1, 4, 1, 17)
    add(4, -1, 3, 1, 4, 1, 17)
    
    # W updates for Block 2 Strassen
    # C02 = M12 + M15 - M16 + M18
    add(0, 0, 0, 0, 4, 1, 11)
    add(0, 0, 0, 0, 4, 1, 14)
    add(0, 0, 0, 0, 4, -1, 15)
    add(0, 0, 0, 0, 4, 1, 17)
    
    # C03 = M14 + M16
    add(0, 0, 0, 0, 5, 1, 13)
    add(0, 0, 0, 0, 5, 1, 15)
    
    # C12 = M13 + M15
    add(0, 0, 0, 0, 6, 1, 12)
    add(0, 0, 0, 0, 6, 1, 14)
    
    # C13 = M12 - M13 + M14 + M17
    add(0, 0, 0, 0, 7, 1, 11)
    add(0, 0, 0, 0, 7, -1, 12)
    add(0, 0, 0, 0, 7, 1, 13)
    add(0, 0, 0, 0, 7, 1, 16)
    
    # Outer product part for Block 2 (A col 2, B row 2)
    # B row 2 cols 2,3 are V[10] (j=2,k=2) and V[11] (j=2,k=3)
    # M19: A02 B22
    add(2, 1, 10, 1, 4, 1, 18)
    # M20: A02 B23
    add(2, 1, 11, 1, 5, 1, 19)
    # M21: A12 B22
    add(5, 1, 10, 1, 6, 1, 20)
    # M22: A12 B23
    add(5, 1, 11, 1, 7, 1, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
