import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Block L: j in {0,1}, k in {0,1}
    # Columns 0-6
    # Block R: j in {0,1}, k in {2,3}
    # Columns 7-13
    
    def add_strassen_block(base_col, j_offset, k_offset):
        # j_offset is 0 for both, but k_offset is 0 or 2
        # i is always 0,1
        
        # M1 (col base_col + 0)
        c = base_col
        U[0][c] += 1
        U[1][c] += 1
        V[0 * p + 0 + j_offset][c] += 1
        V[1 * p + 0 + j_offset][c] += 1
        # W indices: k*n + i
        # C00 (i=0, k=k_offset): W[k_offset*n + 0]
        # C11 (i=1, k=k_offset+1): W[(k_offset+1)*n + 1]
        W[k_offset * n + 0][c] += 1
        W[(k_offset + 1) * n + 1][c] += 1
        
        # M2 (col base_col + 1)
        c = base_col + 1
        U[1][c] += 1
        U[1][c] += 1 # A10+A11 -> U[1] gets 1, U[1] gets 1? No.
        # A10 is U[1][*], A11 is U[1][*]. So U[1] gets 1+1=2?
        # Wait, U[i, c] is the coefficient for A[i, j].
        # M2 = (A10+A11) * B00.
        # So U[1, c] should be 1 for A10 and 1 for A11?
        # No, U is a vector. U[i, c] is the coefficient of A[i, j_c].
        # Here j_c is fixed for the term.
        # In Strassen, we form sums of A entries.
        # M2 uses A10 and A11.
        # So U[1, c] = 1 (for A10) and U[1, c] = 1 (for A11)?
        # No, U[i, c] is a single value.
        # The term is $\sum_i U[i, c] A[i, j]$.
        # If the term involves $A_{10}$ and $A_{11}$, and they are both in row 1,
        # then $U[1, c]$ must be the sum of their coefficients.
        # Coeff of A10 is 1, Coeff of A11 is 1.
        # So $U[1, c] = 2$?
        # But $A_{10}$ and $A_{11}$ are different columns of A.
        # $U$ has size $n \times m$. $U[i, j]$? No, $U$ has size $n \times m$?
        # Task: U shape (n*m, rank).
        # So U[i*m + j, c].
        # A10 is index 1*2 + 0 = 2.
        # A11 is index 1*2 + 1 = 3.
        # So U[2, c] = 1, U[3, c] = 1.
        
        U[1 * m + 0][c] += 1
        U[1 * m + 1][c] += 1
        V[0 * p + j_offset][c] += 1 # B00
        # C10 (i=1, k=k_offset): W[k_offset*n + 1]
        W[k_offset * n + 1][c] += 1
        
        # M3 (col base_col + 2)
        c = base_col + 2
        U[0 * m + 0][c] += 1 # A00
        V[1 * p + j_offset][c] += 1 # B10
        V[1 * p + 1 + j_offset][c] += -1 # B11
        # C01 (i=0, k=k_offset+1): W[(k_offset+1)*n + 0]
        W[(k_offset + 1) * n + 0][c] += 1
        
        # M4 (col base_col + 3)
        c = base_col + 3
        U[1 * m + 1][c] += 1 # A11
        V[0 * p + j_offset][c] += 1 # B00
        V[0 * p + 1 + j_offset][c] += -1 # B01
        # C00 (i=0, k=k_offset): W[k_offset*n + 0]
        W[k_offset * n + 0][c] += 1
        # C10 (i=1, k=k_offset): W[k_offset*n + 1]
        W[k_offset * n + 1][c] += 1
        
        # M5 (col base_col + 4)
        c = base_col + 4
        U[0 * m + 0][c] += 1 # A00
        U[0 * m + 1][c] += 1 # A01
        V[1 * p + 1 + j_offset][c] += 1 # B11
        # C00 (i=0, k=k_offset): W[k_offset*n + 0]
        W[k_offset * n + 0][c] += -1
        # C01 (i=0, k=k_offset+1): W[(k_offset+1)*n + 0]
        W[(k_offset + 1) * n + 0][c] += 1
        
        # M6 (col base_col + 5)
        c = base_col + 5
        U[1 * m + 0][c] += 1 # A10
        U[0 * m + 0][c] += -1 # A00
        V[0 * p + j_offset][c] += 1 # B00
        V[0 * p + 1 + j_offset][c] += 1 # B01
        # C11 (i=1, k=k_offset+1): W[(k_offset+1)*n + 1]
        W[(k_offset + 1) * n + 1][c] += 1
        
        # M7 (col base_col + 6)
        c = base_col + 6
        U[0 * m + 1][c] += 1 # A01
        U[1 * m + 1][c] += -1 # A11
        V[1 * p + j_offset][c] += 1 # B10
        V[1 * p + 1 + j_offset][c] += 1 # B11
        # C00 (i=0, k=k_offset): W[k_offset*n + 0]
        W[k_offset * n + 0][c] += 1
        # C11 (i=1, k=k_offset+1): W[(k_offset+1)*n + 1]
        W[(k_offset + 1) * n + 1][c] += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(7, 0, 2)
    
    # Outer product part (j=2, k=0..3)
    c = 14
    for i in range(n):
        U[i * m + 2][c] = 1
    V[2 * p + 0][c] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
