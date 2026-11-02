import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # We decompose C = A * B
    # A is 2x4, B is 4x5
    # Split B into B_left (4x2) and B_right (4x3)
    # C = A * B_left + A * B_right
    
    # Part 1: A * B_left (2x4 * 4x2)
    # Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # Split B_left into B_L1 (2x2, rows 0-1) and B_L2 (2x2, rows 2-3)
    # C_left = A1 * B_L1 + A2 * B_L2
    # Use Strassen for A1*B_L1 (Rank 7) and A2*B_L2 (Rank 7)
    
    # Part 2: A * B_right (2x4 * 4x3)
    # Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # Split B_right into B_R1 (2x3, rows 0-1) and B_R2 (2x3, rows 2-3)
    # C_right = A1 * B_R1 + A2 * B_R2
    # Use Schoolbook for A1*B_R1 (Rank 12) and A2*B_R2 (Rank 12)
    
    # Total Rank = 7 + 7 + 12 + 12 = 38
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Strassen for 2x2 * 2x2
    # Inputs: A rows 0-1, cols a0-a1; B rows b0-b1, cols 0-1
    # Output: C rows 0-1, cols c0-c1
    def strassen_2x2(a_col_offset, b_row_offset, c_col_offset):
        # A indices: i in 0..1, j in a0..a0+1
        # B indices: j in b0..b0+1, k in 0..1
        # C indices: i in 0..1, k in c0..c0+1
        
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Note: Indices in U are i*m + j. m=4.
        # A11 is i=0, j=a0. A12 is i=0, j=a0+1.
        # A21 is i=1, j=a0. A22 is i=1, j=a0+1.
        
        # M1
        # U: A00, A11. V: B00, B11. W: C00, C01, C10, C11
        # C00 += M1, C11 += M1
        # C01 += M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add 7 terms.
        # Term 1: (A00+A11)(B00+B11)
        # U: A00(1), A11(1). V: B00(1), B11(1).
        # This is not a rank 1 term. It's a product of sums.
        # We need to expand this into rank 1 terms?
        # No, Strassen's algorithm works by defining intermediate matrices.
        # The tensor decomposition rank is 7 because we have 7 multiplications.
        # Each multiplication is a rank 1 tensor?
        # No, the multiplication $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$ is a bilinear form.
        # In the tensor decomposition, we represent the whole thing as a sum of 7 rank 1 tensors.
        
        # The 7 rank 1 tensors correspond to the 7 multiplications.
        # But the inputs to the multiplications are linear combinations of A and B.
        # So U and V will have entries that are sums/differences.
        
        # Let's define the 7 terms explicitly.
        
        # T1: (A00+A11) * (B00+B11)
        # U: A00=1, A11=1. V: B00=1, B11=1.
        # W: C00=1, C11=1. (From C00=M1..., C11=M1...)
        # Wait, C00 uses M1 with +1. C11 uses M1 with +1.
        # C01, C10 don't use M1.
        
        # We need to be careful. The W matrix accumulates the contributions.
        # For term 1:
        # U[0*m+a0] += 1
        # U[1*m+a0+1] += 1  # A11
        # V[b0*5+0] += 1     # B00
        # V[(b0+1)*5+1] += 1 # B11
        # W[0*n+0] += 1      # C00
        # W[1*n+1] += 1      # C11
        
        # But we can't just "add" to the column if we are constructing the column.
        # We are constructing a specific column for this term.
        # So for this term (column index c):
        # U[0*m+a0][c] = 1
        # U[1*m+a0+1][c] = 1
        # V[b0*5+0][c] = 1
        # V[(b0+1)*5+1][c] = 1
        # W[0*n+0][c] = 1
        # W[1*n+1][c] = 1
        
        # This works.
        
        a0 = a_col_offset
        b0 = b_row_offset
        c0 = c_col_offset
        
        # T1: (A00+A11)(B00+B11) -> C00, C11
        U[0*m + a0][col] = 1
        U[1*m + a0 + 1][col] = 1
        V[b0*5 + 0][col] = 1
        V[(b0+1)*5 + 1][col] = 1
        W[0*n + c0][col] = 1
        W[1*n + c0 + 1][col] = 1
        col += 1
        
        # T2: (A10+A11)B00 -> C10
        # A10 is i=1, j=a0. A11 is i=1, j=a0+1.
        U[1*m + a0][col] = 1
        U[1*m + a0 + 1][col] = 1
        V[b0*5 + 0][col] = 1
        W[1*n + c0][col] = 1
        col += 1
        
        # T3: A00(B01-B11) -> C01
        # B01 is j=b0, k=1. B11 is j=b0+1, k=1.
        U[0*m + a0][col] = 1
        V[b0*5 + 1][col] = 1
        V[(b0+1)*5 + 1][col] = -1
        W[0*n + c0 + 1][col] = 1
        col += 1
        
        # T4: A11(B10-B00) -> C00, C10
        # B10 is j=b0+1, k=0. B00 is j=b0, k=0.
        U[1*m + a0 + 1][col] = 1
        V[(b0+1)*5 + 0][col] = 1
        V[b0*5 + 0][col] = -1
        W[0*n + c0][col] = 1
        W[1*n + c0][col] = 1
        col += 1
        
        # T5: (A00+A01)B11 -> C00, C01
        # A00 is i=0, j=a0. A01 is i=0, j=a0+1.
        U[0*m + a0][col] = 1
        U[0*m + a0 + 1][col] = 1
        V[(b0+1)*5 + 1][col] = 1
        W[0*n + c0][col] = -1 # C00 -= M5
        W[0*n + c0 + 1][col] = 1 # C01 += M5
        col += 1
        
        # T6: (A10-A00)(B00+B01) -> C11
        # A10 is i=1, j=a0. A00 is i=0, j=a0.
        U[1*m + a0][col] = 1
        U[0*m + a0][col] = -1
        V[b0*5 + 0][col] = 1
        V[b0*5 + 1][col] = 1
        W[1*n + c0 + 1][col] = 1
        col += 1
        
        # T7: (A01-A11)(B10+B11) -> C00, C11
        # A01 is i=0, j=a0+1. A11 is i=1, j=a0+1.
        U[0*m + a0 + 1][col] = 1
        U[1*m + a0 + 1][col] = -1
        V[(b0+1)*5 + 0][col] = 1
        V[(b0+1)*5 + 1][col] = 1
        W[0*n + c0][col] = 1
        W[1*n + c0 + 1][col] = 1
        col += 1

    # Schoolbook for 2x2 * 2x3
    def schoolbook_2x2_3(a_col_offset, b_row_offset, c_col_offset):
        a0 = a_col_offset
        b0 = b_row_offset
        c0 = c_col_offset
        
        # C00 = A00B00 + A01B10
        U[0*m + a0][col] = 1
        V[b0*5 + 0][col] = 1
        W[0*n + c0][col] = 1
        col += 1
        
        U[0*m + a0 + 1][col] = 1
        V[(b0+1)*5 + 0][col] = 1
        W[0*n + c0][col] = 1
        col += 1
        
        # C01 = A00B01 + A01B11
        U[0*m + a0][col] = 1
        V[b0*5 + 1][col] = 1
        W[0*n + c0 + 1][col] = 1
        col += 1
        
        U[0*m + a0 + 1][col] = 1
        V[(b0+1)*5 + 1][col] = 1
        W[0*n + c0 + 1][col] = 1
        col += 1
        
        # C02 = A00B02 + A01B12
        U[0*m + a0][col] = 1
        V[b0*5 + 2][col] = 1
        W[0*n + c0 + 2][col] = 1
        col += 1
        
        U[0*m + a0 + 1][col] = 1
        V[(b0+1)*5 + 2][col] = 1
        W[0*n + c0 + 2][col] = 1
        col += 1
        
        # C10 = A10B00 + A11B10
        U[1*m + a0][col] = 1
        V[b0*5 + 0][col] = 1
        W[1*n + c0][col] = 1
        col += 1
        
        U[1*m + a0 + 1][col] = 1
        V[(b0+1)*5 + 0][col] = 1
        W[1*n + c0][col] = 1
        col += 1
        
        # C11 = A10B01 + A11B11
        U[1*m + a0][col] = 1
        V[b0*5 + 1][col] = 1
        W[1*n + c0 + 1][col] = 1
        col += 1
        
        U[1*m + a0 + 1][col] = 1
        V[(b0+1)*5 + 1][col] = 1
        W[1*n + c0 + 1][col] = 1
        col += 1
        
        # C12 = A10B02 + A11B12
        U[1*m + a0][col] = 1
        V[b0*5 + 2][col] = 1
        W[1*n + c0 + 2][col] = 1
        col += 1
        
        U[1*m + a0 + 1][col] = 1
        V[(b0+1)*5 + 2][col] = 1
        W[1*n + c0 + 2][col] = 1
        col += 1

    # Part 1: A1 * B_L1 (Strassen)
    # A1 cols 0-1, B_L1 rows 0-1, C_L1 cols 0-1
    strassen_2x2(0, 0, 0)
    
    # Part 1: A2 * B_L2 (Strassen)
    # A2 cols 2-3, B_L2 rows 2-3, C_L2 cols 0-1 (Wait, C_left is cols 0-1)
    # B_L2 is rows 2-3 of B.
    # A2 is cols 2-3 of A.
    # Product is A2 * B_L2.
    # This contributes to C_left.
    strassen_2x2(2, 2, 0)
    
    # Part 2: A1 * B_R1 (Schoolbook)
    # A1 cols 0-1, B_R1 rows 0-1, C_R1 cols 2-4
    schoolbook_2x2_3(0, 0, 2)
    
    # Part 2: A2 * B_R2 (Schoolbook)
    # A2 cols 2-3, B_R2 rows 2-3, C_R2 cols 2-4
    schoolbook_2x2_3(2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
