import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen(i_offset, j_offset, k_offset):
        # i_offset: row offset in A (0 or 2)
        # j_offset: col offset in A (0 or 2) - wait, A is 2x4.
        # A indices: i in 0..1, j in 0..3.
        # B indices: j in 0..3, k in 0..4.
        # C indices: i in 0..1, k in 0..4.
        
        # We are doing A_block (2x2) * B_block (2x2)
        # A_block rows: i_offset, i_offset+1
        # A_block cols: j_offset, j_offset+1
        # B_block rows: j_offset, j_offset+1
        # B_block cols: k_offset, k_offset+1
        
        # Strassen products P1..P7
        # P1 = (a+d)(e+h)
        # a = A[i_offset, j_offset]
        # d = A[i_offset+1, j_offset+1]
        # e = B[j_offset, k_offset]
        # h = B[j_offset+1, k_offset+1]
        
        # We need to express each P in terms of U, V, W.
        # P = (sum u_a A_a) (sum v_b B_b)
        # This corresponds to one term in the tensor decomposition?
        # No, Strassen computes Q = M N.
        # Q_{ij} = sum P_l * coeff_{l,ij}.
        # The tensor decomposition is for the multiplication itself.
        # Each Strassen product P_l corresponds to a rank-1 tensor?
        # No, P_l is a scalar product of linear combinations.
        # $P_l = (\sum A_{i,j} u_{i,j,l}) (\sum B_{j,k} v_{j,k,l})$.
        # Then $Q_{i,k} = \sum_l P_l w_{i,k,l}$.
        # So for each Strassen product l, we add one column to U, V, W.
        
        nonlocal col
        if col >= rank:
            return
            
        # Strassen coefficients for 2x2x2
        # P1: (A00+A11)(B00+B11)
        # P2: (A10+A11)B00
        # P3: A00(B01-B11)
        # P4: A11(B10-B00)
        # P5: (A00+A01)B11
        # P6: (A10-A00)(B00+B01)
        # P7: (A01-A11)(B10+B11)
        
        # Q00 = P1 + P4 - P5 + P7
        # Q01 = P3 + P5
        # Q10 = P2 + P4
        # Q11 = P1 - P2 + P3 + P6
        
        # U, V, W for each P
        
        # P1
        U[i_offset * m + j_offset][col] = 1
        U[(i_offset+1) * m + (j_offset+1)][col] = 1
        V[j_offset * p + k_offset][col] = 1
        V[(j_offset+1) * p + (k_offset+1)][col] = 1
        W[k_offset * n + i_offset][col] = 1
        W[(k_offset+1) * n + (i_offset+1)][col] = 1
        col += 1
        
        # P2
        U[(i_offset+1) * m + j_offset][col] = 1
        U[(i_offset+1) * m + (j_offset+1)][col] = 1
        V[j_offset * p + k_offset][col] = 1
        W[k_offset * n + (i_offset+1)][col] = 1
        col += 1
        
        # P3
        U[i_offset * m + j_offset][col] = 1
        V[j_offset * p + (k_offset+1)][col] = 1
        V[(j_offset+1) * p + (k_offset+1)][col] = -1
        W[(k_offset+1) * n + i_offset][col] = 1
        col += 1
        
        # P4
        U[(i_offset+1) * m + (j_offset+1)][col] = 1
        V[(j_offset+1) * p + k_offset][col] = 1
        V[j_offset * p + k_offset][col] = -1
        W[k_offset * n + (i_offset+1)][col] = 1
        col += 1
        
        # P5
        U[i_offset * m + j_offset][col] = 1
        U[i_offset * m + (j_offset+1)][col] = 1
        V[(j_offset+1) * p + (k_offset+1)][col] = 1
        W[(k_offset+1) * n + i_offset][col] = -1
        col += 1
        
        # P6
        U[(i_offset+1) * m + j_offset][col] = 1
        U[i_offset * m + j_offset][col] = -1
        V[j_offset * p + k_offset][col] = 1
        V[j_offset * p + (k_offset+1)][col] = 1
        W[(k_offset+1) * n + (i_offset+1)][col] = 1
        col += 1
        
        # P7
        U[i_offset * m + (j_offset+1)][col] = 1
        U[(i_offset+1) * m + (j_offset+1)][col] = -1
        V[(j_offset+1) * p + k_offset][col] = 1
        V[(j_offset+1) * p + (k_offset+1)][col] = 1
        W[k_offset * n + i_offset][col] = 1
        col += 1

    # Helper to add 2x2x1 standard block
    def add_std_2x2x1(i_offset, j_offset, k_offset):
        # A_block (2x2) * v (2x1)
        # Q0 = A00 v0 + A01 v1
        # Q1 = A10 v0 + A11 v1
        nonlocal col
        if col >= rank:
            return
            
        # Term 1: A00 v0
        U[i_offset * m + j_offset][col] = 1
        V[j_offset * p + k_offset][col] = 1
        W[k_offset * n + i_offset][col] = 1
        col += 1
        
        # Term 2: A01 v1
        U[i_offset * m + (j_offset+1)][col] = 1
        V[(j_offset+1) * p + k_offset][col] = 1
        W[k_offset * n + i_offset][col] = 1
        col += 1
        
        # Term 3: A10 v0
        U[(i_offset+1) * m + j_offset][col] = 1
        V[j_offset * p + k_offset][col] = 1
        W[k_offset * n + (i_offset+1)][col] = 1
        col += 1
        
        # Term 4: A11 v1
        U[(i_offset+1) * m + (j_offset+1)][col] = 1
        V[(j_offset+1) * p + k_offset][col] = 1
        W[k_offset * n + (i_offset+1)][col] = 1
        col += 1

    # 1. A(0,1) * B(0,1) [2x2 * 2x4]
    # B is 2x4, so split into two 2x2
    # B cols 0,1
    add_strassen(0, 0, 0)
    # B cols 2,3
    add_strassen(0, 0, 2) # Wait, j_offset is for A cols and B rows.
    # A cols 0,1. B rows 0,1.
    # For B cols 2,3, k_offset is 2.
    
    # 2. A(2,3) * B(0,1) [2x2 * 2x4]
    # A cols 2,3. B rows 0,1.
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # 3. A(0,1) * B(2,3) [2x2 * 2x4]
    # A cols 0,1. B rows 2,3.
    add_strassen(0, 0, 0) # j_offset 0? No.
    # A cols 0,1 -> j_offset 0.
    # B rows 2,3 -> j_offset 2.
    # So add_strassen(0, 2, 0) is already used?
    # No, add_strassen(i_offset, j_offset, k_offset).
    # i_offset is A row offset (always 0 for 2x2 rows).
    # j_offset is A col offset AND B row offset.
    # So for A cols 0,1 and B rows 2,3, j_offset must be both 0 and 2? Impossible.
    
    # My helper assumes A cols and B rows have the same offset.
    # This is true for block diagonal structure.
    # But here we are doing all combinations.
    # A cols 0,1 * B rows 2,3.
    # A indices: i, j (j in 0,1).
    # B indices: j', k (j' in 2,3).
    # The product term is A[i,j] B[j',k] with j != j'.
    # This is NOT a simple 2x2x2 multiplication in the same index space.
    # It is a 2x2x2 multiplication but with shifted indices.
    
    # I need a more general helper.
    pass

# I will rewrite the construction more carefully.
