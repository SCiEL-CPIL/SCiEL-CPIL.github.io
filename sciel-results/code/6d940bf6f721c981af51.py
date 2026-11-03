import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (12, 44)
    # V shape: (m*p, rank) = (16, 44)
    # W shape: (p*n, rank) = (12, 44)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices in A (n x m) -> U index: i*m + j
    # Indices in B (m x p) -> V index: j*p + k
    # Indices in C (p x n) -> W index: k*n + i
    # Note: C is p x n in W index?
    # Task: W[k*n+i, r]. k is row of C (0..p-1), i is col of C (0..n-1).
    # C = A B. A is n x m. B is m x p. C is n x p.
    # Wait, standard matrix multiplication C[i,k] = sum A[i,j] B[j,k].
    # C is n x p.
    # But W index is k*n + i. This implies C is stored as p x n?
    # "W[k*n+i,r]". k is 0..p-1, i is 0..n-1.
    # So W corresponds to C^T? Or just a specific ordering.
    # Let's check the schoolbook code.
    # W[k*n+i][column] = 1.
    # Loop: i in range(n), j in range(m), k in range(p).
    # U[i*m+j] = 1. (A[i,j])
    # V[j*p+k] = 1. (B[j,k])
    # W[k*n+i] = 1. (C[k,i]?)
    # If C[i,k] = sum A[i,j]B[j,k], then the term is A[i,j]B[j,k].
    # The output index should correspond to C[i,k].
    # If W index is k*n+i, then W[k*n+i] corresponds to C[i,k] if we consider C transposed?
    # Or maybe the task defines C as p x n?
    # "multiplying a 3 by 4 matrix by a 4 by 4 matrix".
    # A (3x4) * B (4x4) = C (3x4).
    # So C is 3x4. n=3, p=4.
    # W index k*n+i. k in 0..3, i in 0..2.
    # This maps to a 4x3 matrix.
    # So W stores C^T?
    # Let's assume the evaluator checks consistency with this indexing.
    # I will follow the indexing of the seed program.
    
    col = 0
    
    # Define the 4 blocks
    # Block (a, b) where a is col block of A (0 or 1), b is row block of B (0 or 1)
    # A cols: 0-1 (block 0), 2-3 (block 1)
    # B rows: 0-1 (block 0), 2-3 (block 1)
    # B cols: 0-1 (block 0), 2-3 (block 1)
    
    # We need to compute C[:, 0:2] and C[:, 2:4].
    # C[:, 0:2] = A[:, 0:2] B[0:2, 0:2] + A[:, 2:4] B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] B[0:2, 2:4] + A[:, 2:4] B[2:4, 2:4]
    
    # Let's define the 11 multiplications for a generic 3x2x2 block.
    # Inputs: A_sub (3x2), B_sub (2x2).
    # A_sub rows: 0,1,2. Cols: 0,1.
    # B_sub rows: 0,1. Cols: 0,1.
    # Output C_sub (3x2). Rows: 0,1,2. Cols: 0,1.
    
    # Multiplications:
    # 1. (A00+A11)(B00+B11)
    # 2. (A10+A11)B00
    # 3. A00(B10-B00)
    # 4. A11(B01-B11)
    # 5. (A00+A01)B11
    # 6. (A10-A00)(B00+B01)
    # 7. (A01-A11)(B10+B11)
    # 8. A20 B00
    # 9. A21 B10
    # 10. A20 B01
    # 11. A21 B11
    
    # Note: A indices in 3x2:
    # Row 0: A00, A01
    # Row 1: A10, A11
    # Row 2: A20, A21
    # B indices in 2x2:
    # Row 0: B00, B01
    # Row 1: B10, B11
    
    # Output C indices in 3x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # C20 = M8 + M9
    # C21 = M10 + M11
    
    def add_block(A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset):
        # A_row_offset: 0 (since we use all 3 rows of A)
        # A_col_offset: 0 or 2
        # B_row_offset: 0 or 2
        # B_col_offset: 0 or 2
        # C_row_offset: 0 (since we use all 3 rows of C)
        # C_col_offset: 0 or 2
        
        nonlocal col
        
        # M1
        U[A_row_offset + 0][col] = 1 # A00
        U[A_row_offset + 1][col] = 1 # A11
        V[B_row_offset + 0][col] = 1 # B00
        V[B_row_offset + 1][col] = 1 # B11
        W[C_row_offset + 0][col] = 1 # C00
        W[C_row_offset + 1][col] = 1 # C11
        col += 1
        
        # M2
        U[A_row_offset + 1][col] = 1 # A10 + A11
        U[A_row_offset + 1][col] = 1 # Wait, A10 is row 1 col 0. A11 is row 1 col 1.
        # My A indexing: A[i, j].
        # A00: i=0, j=0.
        # A10: i=1, j=0.
        # A11: i=1, j=1.
        # So M2 = (A10 + A11) B00.
        U[A_row_offset + 1][col] = 1 # A10? No, A[1,0]
        # Wait, U index is i*m + j.
        # For the sub-block, local i, j.
        # Global i = A_row_offset + local_i.
        # Global j = A_col_offset + local_j.
        
        # Let's restart the mapping for one block.
        
        # Local A (3x2):
        # (0,0), (0,1)
        # (1,0), (1,1)
        # (2,0), (2,1)
        
        # Local B (2x2):
        # (0,0), (0,1)
        # (1,0), (1,1)
        
        # Local C (3x2):
        # (0,0), (0,1)
        # (1,0), (1,1)
        # (2,0), (2,1)
        
        # M1: (A00 + A11) (B00 + B11)
        # U: A00 (0,0), A11 (1,1)
        # V: B00 (0,0), B11 (1,1)
        # W: C00 (0,0), C11 (1,1)
        
        # M2: (A10 + A11) B00
        # U: A10 (1,0), A11 (1,1)
        # V: B00 (0,0)
        # W: C10 (1,0)
        
        # M3: A00 (B10 - B00)
        # U: A00 (0,0)
        # V: B10 (1,0), -B00 (0,0)
        # W: C01 (0,1)
        
        # M4: A11 (B01 - B11)
        # U: A11 (1,1)
        # V: B01 (0,1), -B11 (1,1)
        # W: C00 (0,0), C10 (1,0)
        
        # M5: (A00 + A01) B11
        # U: A00 (0,0), A01 (0,1)
        # V: B11 (1,1)
        # W: -C00 (0,0), C01 (0,1)
        
        # M6: (A10 - A00) (B00 + B01)
        # U: A10 (1,0), -A00 (0,0)
        # V: B00 (0,0), B01 (0,1)
        # W: C11 (1,1)
        
        # M7: (A01 - A11) (B10 + B11)
        # U: A01 (0,1), -A11 (1,1)
        # V: B10 (1,0), B11 (1,1)
        # W: C00 (0,0)
        
        # M8: A20 B00
        # U: A20 (2,0)
        # V: B00 (0,0)
        # W: C20 (2,0)
        
        # M9: A21 B10
        # U: A21 (2,1)
        # V: B10 (1,0)
        # W: C20 (2,0)
        
        # M10: A20 B01
        # U: A20 (2,0)
        # V: B01 (0,1)
        # W: C21 (2,1)
        
        # M11: A21 B11
        # U: A21 (2,1)
        # V: B11 (1,1)
        # W: C21 (2,1)
        
        # Helper to add a term
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for ui in u_indices:
                U[ui][col] += 1
            for vi in v_indices:
                V[vi][col] += 1
            for wi in w_indices:
                W[wi][col] += 1
            col += 1
        
        # M1
        add_term([A_row_offset*4 + 0*2 + 0, A_row_offset*4 + 1*2 + 1], 
                 [B_row_offset*4 + 0*2 + 0, B_row_offset*4 + 1*2 + 1],
                 [C_row_offset*3 + 0*2 + 0, C_row_offset*3 + 1*2 + 1])
        # Wait, U index is i*m + j. m=4.
        # So A(i,j) -> i*4 + j.
        # V index is j*p + k. p=4.
        # So B(j,k) -> j*4 + k.
        # W index is k*n + i. n=3.
        # So C(i,k) -> k*3 + i.
        
        # Let's redefine indices with global offsets.
        # A local (li, lj) -> global (li + A_row_offset, lj + A_col_offset).
        # U index: (li + A_row_offset)*4 + (lj + A_col_offset).
        
        # B local (lj, lk) -> global (lj + B_row_offset, lk + B_col_offset).
        # V index: (lj + B_row_offset)*4 + (lk + B_col_offset).
        
        # C local (li, lk) -> global (li + C_row_offset, lk + C_col_offset).
        # W index: (lk + C_col_offset)*3 + (li + C_row_offset).
        
        def get_u(li, lj): return (li + A_row_offset)*4 + (lj + A_col_offset)
        def get_v(lj, lk): return (lj + B_row_offset)*4 + (lk + B_col_offset)
        def get_w(li, lk): return (lk + C_col_offset)*3 + (li + C_row_offset)
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([get_u(0,0), get_u(1,1)], [get_v(0,0), get_v(1,1)], [get_w(0,0), get_w(1,1)])
        
        # M2: (A10+A11)B00 -> C10
        add_term([get_u(1,0), get_u(1,1)], [get_v(0,0)], [get_w(1,0)])
        
        # M3: A00(B10-B00) -> C01
        add_term([get_u(0,0)], [get_v(1,0), -get_v(0,0)], [get_w(0,1)])
        # Note: add_term adds 1. We need -1 for some.
        # Let's modify add_term to take values.
        
    # Re-implement with values
    def add_block(A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset):
        nonlocal col
        
        def get_u(li, lj): return (li + A_row_offset)*4 + (lj + A_col_offset)
        def get_v(lj, lk): return (lj + B_row_offset)*4 + (lk + B_col_offset)
        def get_w(li, lk): return (lk + C_col_offset)*3 + (li + C_row_offset)
        
        def add_term(u_vals, v_vals, w_vals):
            nonlocal col
            for idx, val in u_vals:
                U[idx][col] += val
            for idx, val in v_vals:
                V[idx][col] += val
            for idx, val in w_vals:
                W[idx][col] += val
            col += 1
            
        # M1
        add_term([(get_u(0,0), 1), (get_u(1,1), 1)], 
                 [(get_v(0,0), 1), (get_v(1,1), 1)], 
                 [(get_w(0,0), 1), (get_w(1,1), 1)])
        
        # M2
        add_term([(get_u(1,0), 1), (get_u(1,1), 1)], 
                 [(get_v(0,0), 1)], 
                 [(get_w(1,0), 1)])
        
        # M3
        add_term([(get_u(0,0), 1)], 
                 [(get_v(1,0), 1), (get_v(0,0), -1)], 
                 [(get_w(0,1), 1)])
        
        # M4
        add_term([(get_u(1,1), 1)], 
                 [(get_v(0,1), 1), (get_v(1,1), -1)], 
                 [(get_w(0,0), 1), (get_w(1,0), 1)])
        
        # M5
        add_term([(get_u(0,0), 1), (get_u(0,1), 1)], 
                 [(get_v(1,1), 1)], 
                 [(get_w(0,0), -1), (get_w(0,1), 1)])
        
        # M6
        add_term([(get_u(1,0), 1), (get_u(0,0), -1)], 
                 [(get_v(0,0), 1), (get_v(0,1), 1)], 
                 [(get_w(1,1), 1)])
        
        # M7
        add_term([(get_u(0,1), 1), (get_u(1,1), -1)], 
                 [(get_v(1,0), 1), (get_v(1,1), 1)], 
                 [(get_w(0,0), 1)])
        
        # M8
        add_term([(get_u(2,0), 1)], 
                 [(get_v(0,0), 1)], 
                 [(get_w(2,0), 1)])
        
        # M9
        add_term([(get_u(2,1), 1)], 
                 [(get_v(1,0), 1)], 
                 [(get_w(2,0), 1)])
        
        # M10
        add_term([(get_u(2,0), 1)], 
                 [(get_v(0,1), 1)], 
                 [(get_w(2,1), 1)])
        
        # M11
        add_term([(get_u(2,1), 1)], 
                 [(get_v(1,1), 1)], 
                 [(get_w(2,1), 1)])

    # Block 1: A[:, 0:2] B[0:2, 0:2] -> C[:, 0:2] part 1
    add_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A[:, 2:4] B[2:4, 0:2] -> C[:, 0:2] part 2
    add_block(0, 2, 2, 0, 0, 0)
    
    # Block 3: A[:, 0:2] B[0:2, 2:4] -> C[:, 2:4] part 1
    add_block(0, 0, 0, 2, 0, 2)
    
    # Block 4: A[:, 2:4] B[2:4, 2:4] -> C[:, 2:4] part 2
    add_block(0, 2, 2, 2, 0, 2)
    
    return {"U": U, "V": V, "W": W}
