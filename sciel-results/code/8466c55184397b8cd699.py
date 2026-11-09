import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    
    We use a block decomposition strategy:
    Split A (3x4) into rows [0,1] and [2], cols [0,1] and [2,3].
    Split B (4x6) into rows [0,1] and [2,3], cols [0,1,2] and [3,4,5].
    Split C (3x6) into rows [0,1] and [2], cols [0,1,2] and [3,4,5].
    
    This results in 4 blocks of type 2x2x3 and 4 blocks of type 1x2x3.
    Rank(2x2x3) = 11 (using Strassen for 2x2x2 part + schoolbook for 2x2x1 part).
    Rank(1x2x3) = 6 (schoolbook).
    Total Rank = 4 * 11 + 4 * 6 = 68.
    """
    n, m, p = DIMENSIONS
    # Total rank calculation
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Helper for 2x2x3 block multiplication
    # A: 2x2, B: 2x3, C: 2x3
    # u_indices: list of 4 global U indices for A[0,0], A[0,1], A[1,0], A[1,1]
    # v_indices: list of 6 global V indices for B[0,0], B[0,1], B[0,2], B[1,0], B[1,1], B[1,2]
    # w_indices: list of 6 global W indices for C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2]
    def add_223_block(u_idx, v_idx, w_idx):
        # Split B into B_L (2x2) and B_R (2x1)
        # B_L indices: v_idx[0], v_idx[1], v_idx[3], v_idx[4] (cols 0,1)
        # B_R indices: v_idx[2], v_idx[5] (col 2)
        
        # Part 1: A * B_L (2x2x2) using Strassen (Rank 7)
        # Strassen terms for M = A * B_L
        # M[0,0] = C[0,0], M[0,1] = C[0,1], M[1,0] = C[1,0], M[1,1] = C[1,1]
        # w_idx[0] is C[0,0], w_idx[1] is C[0,1], w_idx[3] is C[1,0], w_idx[4] is C[1,1]
        
        # M1 = A[0,0] * (B[0,0] + B[1,1])
        # We need to create linear combinations in U and V.
        # Since we are building a tensor decomposition, we add terms.
        # Term 1: A[0,0] * B[0,0] -> contributes to M1
        add_term(u_idx[0], v_idx[0], -1, 1, 1, 0) # Placeholder, we need to map to W later
        # Actually, it's easier to just expand the Strassen formulas into rank-1 terms directly.
        
        # Strassen 7 multiplications:
        # P1 = A00 * (B00 + B11)
        # P2 = (A10 + A11) * B00
        # P3 = (A00 + A01) * B11
        # P4 = A11 * (B10 - B00)
        # P5 = (A00 + A11) * (B00 + B11)
        # P6 = (A01 - A11) * (B10 + B11)
        # P7 = (A00 - A10) * (B00 + B10)
        
        # C00 = P5 + P4 - P2 + P6
        # C01 = P3 + P7
        # C10 = P2 + P4
        # C11 = P1 - P3 + P5 + P7
        
        # We will add the 7 terms.
        # Term 1: P1
        # U: A00 (1), V: B00+B11 (1,1), W: C00(1), C11(-1)
        # We can't have multiple W entries in one rank-1 term.
        # So we must expand P1 into two terms? No, P1 is one multiplication.
        # The output C is a linear combination of the P's.
        # So for each P, we add a term with U and V set, and W set to the coefficients in the output.
        # Wait, the tensor decomposition is $\sum_{r} U_r \otimes V_r \otimes W_r$.
        # Each term r corresponds to one scalar multiplication.
        # The value added to C[i,j] is $\sum_r U[i,r] V[j,r] W[i,j,r]$? No.
        # $C_{ij} = \sum_r U_{i,r} V_{j,r} W_{i,j,r}$?
        # Standard form: $T_{ijk} = \sum_r U_{ir} V_{jr} W_{kr}$.
        # Here indices are:
        # i: row of A (0..n-1)
        # j: col of A / row of B (0..m-1)
        # k: col of B (0..p-1)
        # U[i*m+j, r]
        # V[j*p+k, r]
        # W[k*n+i, r]
        
        # So for a term r:
        # It contributes to C[i,k] (which is W[k*n+i]) the value U[i*m+j] * V[j*p+k] * W[k*n+i] ?
        # No, the tensor entry is $T_{i,j,k}$.
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # So the tensor has non-zero entries at $(i,j,k)$ with value 1 if we are doing schoolbook.
        # The decomposition is $\sum_r \sum_{i,j,k} U_{ij,r} V_{jk,r} W_{ki,r} x_{ij} y_{jk} z_{ki}$.
        
        # So for a Strassen term like P1 = A00 * (B00 + B11):
        # This is not a single rank-1 term in the original tensor space because B00+B11 is a sum.
        # We must distribute it.
        # P1 contributes to C00 and C11.
        # Actually, Strassen works by computing linear combinations of A and B, multiplying them, and then combining results.
        # The "rank" counts the number of scalar multiplications of the linear combinations.
        # So yes, each of the 7 P's is one rank-1 term in the decomposed tensor.
        # For P1:
        # U part: A00 (index 0,0). Value 1.
        # V part: B00 (0,0) and B11 (1,1). Values 1, 1.
        # W part: C00 (0,0) coeff 1, C11 (1,1) coeff -1.
        # Wait, W is a vector in the tensor.
        # If we define the term r such that $U_r$ has 1 at (0,0), $V_r$ has 1 at (0,0) and 1 at (1,1), and $W_r$ has 1 at (0,0) and -1 at (1,1)...
        # Then the contribution to $C_{00}$ is $1 \cdot 1 \cdot 1 = 1$.
        # Contribution to $C_{11}$ is $1 \cdot 1 \cdot (-1) = -1$.
        # But $C_{00}$ also gets contributions from other terms.
        # This is correct.
        
        # So we just need to fill U, V, W vectors for each of the 7 terms.
        
        # Term P1: A00 * (B00 + B11)
        add_term(u_idx[0], v_idx[0], w_idx[0], 1, 1, 1) # C00 gets +1
        # But V also has B11. We need a separate term for that part?
        # No, V_r is a vector. It can have non-zero entries at multiple positions.
        # So for term r=0 (P1):
        # U[0,0] = 1
        # V[0,0] = 1, V[1,1] = 1
        # W[0,0] = 1, W[1,1] = -1
        # This single term adds 1 to C00 and -1 to C11.
        
        # Let's implement this logic.
        
        # P1: A00 * (B00 + B11)
        # U: u_idx[0] (A00)
        # V: v_idx[0] (B00), v_idx[4] (B11)
        # W: w_idx[0] (C00) coeff 1, w_idx[4] (C11) coeff -1
        add_term(u_idx[0], v_idx[0], w_idx[0], 1, 1, 1)
        # We need to add the second part of V and W to the SAME column.
        # My add_term function increments col. I need to modify it to allow multiple entries in the same column.
        
        # Revised add_term:
        # It should take a list of (index, value) pairs for U, V, W.
        
        # Let's rewrite the helper.
        
    # Redefine add_term to handle sparse updates to a specific column
    def update_term(col_idx, u_updates, v_updates, w_updates):
        for u_idx, val in u_updates:
            U[u_idx][col_idx] += val
        for v_idx, val in v_updates:
            V[v_idx][col_idx] += val
        for w_idx, val in w_updates:
            W[w_idx][col_idx] += val

    # Reset col
    col = 0
    
    # Re-define add_223_block using update_term
    def add_223_block(u_idx, v_idx, w_idx):
        nonlocal col
        # P1 = A00 * (B00 + B11)
        update_term(col, 
                    [(u_idx[0], 1)], 
                    [(v_idx[0], 1), (v_idx[4], 1)], 
                    [(w_idx[0], 1), (w_idx[4], -1)])
        col += 1
        
        # P2 = (A10 + A11) * B00
        update_term(col, 
                    [(u_idx[2], 1), (u_idx[3], 1)], 
                    [(v_idx[0], 1)], 
                    [(w_idx[3], 1), (w_idx[0], -1)]) # C10 gets +1, C00 gets -1
        col += 1
        
        # P3 = (A00 + A01) * B11
        update_term(col, 
                    [(u_idx[0], 1), (u_idx[1], 1)], 
                    [(v_idx[4], 1)], 
                    [(w_idx[1], 1), (w_idx[4], -1)]) # C01 gets +1, C11 gets -1
        col += 1
        
        # P4 = A11 * (B10 - B00)
        update_term(col, 
                    [(u_idx[3], 1)], 
                    [(v_idx[3], 1), (v_idx[0], -1)], 
                    [(w_idx[3], 1), (w_idx[0], 1)]) # C10 gets +1, C00 gets +1
        col += 1
        
        # P5 = (A00 + A11) * (B00 + B11)
        update_term(col, 
                    [(u_idx[0], 1), (u_idx[3], 1)], 
                    [(v_idx[0], 1), (v_idx[4], 1)], 
                    [(w_idx[0], 1), (w_idx[4], 1)]) # C00 gets +1, C11 gets +1
        col += 1
        
        # P6 = (A01 - A11) * (B10 + B11)
        update_term(col, 
                    [(u_idx[1], 1), (u_idx[3], -1)], 
                    [(v_idx[3], 1), (v_idx[4], 1)], 
                    [(w_idx[0], 1), (w_idx[4], -1)]) # C00 gets +1, C11 gets -1
        col += 1
        
        # P7 = (A00 - A10) * (B00 + B10)
        update_term(col, 
                    [(u_idx[0], 1), (u_idx[2], -1)], 
                    [(v_idx[0], 1), (v_idx[3], 1)], 
                    [(w_idx[1], 1), (w_idx[4], 1)]) # C01 gets +1, C11 gets +1
        col += 1
        
        # Part 2: A * B_R (2x2x1) Schoolbook
        # B_R is column 2 of B. Indices v_idx[2] (B02) and v_idx[5] (B12).
        # C_R is column 2 of C. Indices w_idx[2] (C02) and w_idx[5] (C12).
        
        # T1 = A00 * B02 -> C02
        update_term(col, [(u_idx[0], 1)], [(v_idx[2], 1)], [(w_idx[2], 1)])
        col += 1
        
        # T2 = A01 * B12 -> C02
        update_term(col, [(u_idx[1], 1)], [(v_idx[5], 1)], [(w_idx[2], 1)])
        col += 1
        
        # T3 = A10 * B02 -> C12
        update_term(col, [(u_idx[2], 1)], [(v_idx[2], 1)], [(w_idx[5], 1)])
        col += 1
        
        # T4 = A11 * B12 -> C12
        update_term(col, [(u_idx[3], 1)], [(v_idx[5], 1)], [(w_idx[5], 1)])
        col += 1

    # Helper for 1x2x3 block multiplication
    def add_123_block(u_idx, v_idx, w_idx):
        nonlocal col
        # A: 1x2 (u_idx[0], u_idx[1])
        # B: 2x3 (v_idx[0..5])
        # C: 1x3 (w_idx[0..2])
        
        # C0 = A0*B00 + A1*B10
        update_term(col, [(u_idx[0], 1)], [(v_idx[0], 1)], [(w_idx[0], 1)])
        col += 1
        update_term(col, [(u_idx[1], 1)], [(v_idx[3], 1)], [(w_idx[0], 1)])
        col += 1
        
        # C1 = A0*B01 + A1*B11
        update_term(col, [(u_idx[0], 1)], [(v_idx[1], 1)], [(w_idx[1], 1)])
        col += 1
        update_term(col, [(u_idx[1], 1)], [(v_idx[4], 1)], [(w_idx[1], 1)])
        col += 1
        
        # C2 = A0*B02 + A1*B12
        update_term(col, [(u_idx[0], 1)], [(v_idx[2], 1)], [(w_idx[2], 1)])
        col += 1
        update_term(col, [(u_idx[1], 1)], [(v_idx[5], 1)], [(w_idx[2], 1)])
        col += 1

    # Global Index Mappings
    # U index: i*4 + j
    # V index: j*6 + k
    # W index: k*3 + i
    
    # Block C11: A11 (i0-1, j0-1) * B11 (j0-1, k0-2) + A12 (i0-1, j2-3) * B21 (j2-3, k0-2)
    # C11 maps to C rows 0-1, cols 0-2.
    
    # A11 U indices:
    u_A11 = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # A00, A01, A10, A11
    # B11 V indices:
    v_B11 = [0*6+0, 0*6+1, 0*6+2, 1*6+0, 1*6+1, 1*6+2] # B00, B01, B02, B10, B11, B12
    # C11 W indices (local 2x3 to global):
    # Local C00 -> Global C[0,0] -> W[0*3+0]=0
    # Local C01 -> Global C[0,1] -> W[1*3+0]=3
    # Local C02 -> Global C[0,2] -> W[2*3+0]=6
    # Local C10 -> Global C[1,0] -> W[0*3+1]=1
    # Local C11 -> Global C[1,1] -> W[1*3+1]=4
    # Local C12 -> Global C[1,2] -> W[2*3+1]=7
    w_C11 = [0, 3, 6, 1, 4, 7]
    
    add_223_block(u_A11, v_B11, w_C11)
    
    # A12 U indices:
    u_A12 = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    # B21 V indices:
    v_B21 = [2*6+0, 2*6+1, 2*6+2, 3*6+0, 3*6+1, 3*6+2]
    # C11 W indices same as above
    add_223_block(u_A12, v_B21, w_C11)
    
    # Block C12: A11 * B12 + A12 * B22
    # C12 maps to C rows 0-1, cols 3-5.
    
    # B12 V indices:
    v_B12 = [0*6+3, 0*6+4, 0*6+5, 1*6+3, 1*6+4, 1*6+5]
    # C12 W indices:
    # Local C00 -> Global C[0,3] -> W[3*3+0]=9
    # Local C01 -> Global C[0,4] -> W[4*3+0]=12
    # Local C02 -> Global C[0,5] -> W[5*3+0]=15
    # Local C10 -> Global C[1,3] -> W[3*3+1]=10
    # Local C11 -> Global C[1,4] -> W[4*3+1]=13
    # Local C12 -> Global C[1,5] -> W[5*3+1]=16
    w_C12 = [9, 12, 15, 10, 13, 16]
    
    add_223_block(u_A11, v_B12, w_C12)
    
    # B22 V indices:
    v_B22 = [2*6+3, 2*6+4, 2*6+5, 3*6+3, 3*6+4, 3*6+5]
    add_223_block(u_A12, v_B22, w_C12)
    
    # Block C21: A21 * B11 + A22 * B21
    # C21 maps to C row 2, cols 0-2.
    
    # A21 U indices (1x2):
    u_A21 = [2*4+0, 2*4+1]
    # C21 W indices (1x3):
    # Local C0 -> Global C[2,0] -> W[0*3+2]=2
    # Local C1 -> Global C[2,1] -> W[1*3+2]=5
    # Local C2 -> Global C[2,2] -> W[2*3+2]=8
    w_C21 = [2, 5, 8]
    
    add_123_block(u_A21, v_B11, w_C21)
    
    # A22 U indices:
    u_A22 = [2*4+2, 2*4+3]
    add_123_block(u_A22, v_B21, w_C21)
    
    # Block C22: A21 * B12 + A22 * B22
    # C22 maps to C row 2, cols 3-5.
    
    # C22 W indices:
    # Local C0 -> Global C[2,3] -> W[3*3+2]=11
    # Local C1 -> Global C[2,4] -> W[4*3+2]=14
    # Local C2 -> Global C[2,5] -> W[5*3+2]=17
    w_C22 = [11, 14, 17]
    
    add_123_block(u_A21, v_B12, w_C22)
    add_123_block(u_A22, v_B22, w_C22)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
