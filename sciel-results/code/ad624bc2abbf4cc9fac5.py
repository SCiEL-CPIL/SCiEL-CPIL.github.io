"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Tensor T[i, j, k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()
    
    # Constructive Strassen Block Decomposition
    # Strategy:
    # Split A (2x4) into A1 (2x2), A2 (2x2)
    # Split B (4x6) into B1 (2x2), B2 (2x2), B3 (2x2)
    # 
    # We compute:
    # A1*B1, A1*B2, A1*B3
    # A2*B1, A2*B2, A2*B3
    #
    # Using Strassen for 2x2x2 matrix multiplication requires 7 multiplications.
    # We can group the 6 required products into two sets of 3? No, Strassen computes 2x2 * 2x2.
    # Here we have 2 blocks of A multiplying 3 blocks of B.
    # Total 6 products of 2x2 matrices.
    #
    # Let's use the standard 7-multiplication Strassen for 2x2 matrices.
    # But we have 6 distinct 2x2 products needed.
    # Can we compute 6 2x2 products with fewer than 6*4=24 scalar multiplications?
    # Yes.
    #
    # Let A1 = [[a, b], [c, d]] and A2 = [[e, f], [g, h]]
    # Let B1, B2, B3 be 2x2 matrices.
    #
    # We want to compute:
    # P11 = A1 B1, P12 = A1 B2, P13 = A1 B3
    # P21 = A2 B1, P22 = A2 B2, P23 = A2 B3
    #
    # Standard Strassen for 2x2:
    # M1 = (a+d)(B11+B22)
    # M2 = (c+d)B22
    # M3 = (a-b)(B11+B12)
    # M4 = (e+f)B21 -- wait, indices are mixed.
    #
    # Let's apply Strassen to the pair (A1, B1) -> 7 mults
    # Apply Strassen to the pair (A2, B1) -> 7 mults
    # This shares nothing.
    #
    # Better approach:
    # Note that the 6 products are linear combinations of A1, A2 and B1, B2, B3.
    # The tensor for 2x2x2 matrix multiplication has rank 7 (over most fields).
    # We are computing a "slice" of the tensor? No, it's a sum of 6 such tensors.
    #
    # Actually, we can view this as multiplying a 2x4 matrix by a 4x6 matrix.
    # This is equivalent to multiplying a 2x2 block matrix by a 2x3 block matrix? No, A is 2x4, B is 4x6.
    # A = [A1 A2] (2x4)
    # B = [B1; B2; B3] ? No, B is 4x6.
    # If we split B into 3 blocks of 4x2:
    # B = [B1 B2 B3] where Bj is 4x2.
    # Then A1 (2x2) multiplies top half of Bj?
    # Let's stick to 2x2 blocks.
    # A = [A1 A2] (each 2x2)
    # B = [B1; B2; B3] (each 2x2? No, B is 4x6. If we split rows of B into 2,2,2? No, 4 rows.
    # So B can be split into 2 blocks of 2x6? Or 3 blocks of 4x2?
    #
    # Let's split A into A_top (2x2) and A_bottom (2x2). (Rows 0-1, 2-3 of the 4x4? No A is 2x4).
    # A is 2x4. Split columns: A1 (2x2), A2 (2x2).
    # B is 4x6. Split rows: B1 (2x6), B2 (2x6).
    # Then A1 * B1 + A2 * B2 = C (2x6).
    # This requires 2 matrix multiplications of 2x2 * 2x6.
    #
    # Let's use the 3-block split for B as in inspiration 2.
    # B = [B1 B2 B3] where each Bj is 4x2.
    # A = [A1 A2] where each Ai is 2x2.
    # This doesn't align for multiplication A*B directly as block matrix mult.
    # A (2x4) * B (4x6) = A1 (2x2) * B_top (2x6) + A2 (2x2) * B_bottom (2x6).
    #
    # Let's try:
    # A = [A1 A2] (2x2 blocks)
    # B = [B1; B2] (2x6 blocks)
    # C = A1 B1 + A2 B2
    #
    # We can compute A1 B1 using Strassen (7 mults) if we treat B1 as 2x2? No B1 is 2x6.
    # We can split B1 into 3 2x2 blocks: B11, B12, B13.
    # Then A1 B1 = A1 B11 + A1 B12 + A1 B13.
    # This is 3 products of 2x2 matrices.
    #
    # Total 6 products of 2x2 matrices:
    # A1 B11, A1 B12, A1 B13
    # A2 B21, A2 B22, A2 B23
    #
    # We can compute these 6 products using a total of 7 + 7 = 14 multiplications?
    # No, we can combine them.
    #
    # Consider the tensor T_{2,4,6}.
    # It is known that R(2,4,6) <= 42?
    # Let's check the score of rank 42. 48/42 = 1.142.
    #
    # Construction for rank 42:
    # Use Strassen for 2x2x2 (rank 7).
    # We have 6 such blocks.
    # If we just did 6 * 7 = 42, that would be rank 42.
    # Is it possible to do better?
    # The tensor is a sum of 6 disjoint 2x2x2 tensors (in the sense that the indices don't overlap in a way that allows sharing? No, they share indices in A and B).
    #
    # Actually, A1 and A2 share the row indices of A (0,1).
    # B11, B12, B13 share the row indices of B (0,1).
    #
    # Let's just implement the 6 separate Strassen decompositions.
    # Rank = 6 * 7 = 42.
    #
    # Strassen for 2x2 * 2x2:
    # M1 = (a+d)(B11+B22)
    # M2 = (c+d)B22
    # M3 = (a-b)(B11+B12)
    # M4 = (e+f)B21  -- wait, standard Strassen:
    # M1 = (a+d)(B11+B22)
    # M2 = (c+d)B22
    # M3 = (a-b)(B11+B12)
    # M4 = (e+f)B21  -- no, (A21+A22)B21
    # M5 = A11(B21-B22)
    # M6 = A22(B12-B11)
    # M7 = (A11+A22)(B11+B22) -- wait, M1 is (A11+A22)(B11+B22).
    #
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # We need to map these to the global indices.
    #
    # A = [A1 A2]
    # A1 = [[a00, a01], [a10, a11]] -> Global A indices: (0,0), (0,1), (1,0), (1,1)
    # A2 = [[a02, a03], [a12, a13]] -> Global A indices: (0,2), (0,3), (1,2), (1,3)
    #
    # B = [B1; B2]
    # B1 = [B11 B12 B13] -> B1 is 2x6.
    # B11 = [[b00, b01], [b10, b11]] -> Global B indices: (0,0), (0,1), (1,0), (1,1)
    # B12 = [[b02, b03], [b12, b13]] -> Global B indices: (0,2), (0,3), (1,2), (1,3)
    # B13 = [[b04, b05], [b14, b15]] -> Global B indices: (0,4), (0,5), (1,4), (1,5)
    #
    # B2 = [B21 B22 B23] -> B2 is 2x6.
    # B21 = [[b20, b21], [b30, b31]] -> Global B indices: (2,0), (2,1), (3,0), (3,1)
    # B22 = [[b22, b23], [b32, b33]] -> Global B indices: (2,2), (2,3), (3,2), (3,3)
    # B23 = [[b24, b25], [b34, b35]] -> Global B indices: (2,4), (2,5), (3,4), (3,5)
    #
    # C = A1 B1 + A2 B2
    # C is 2x6.
    # C = [C1 C2 C3] where Ci is 2x2.
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    # C3 = A1 B13 + A2 B23
    #
    # We compute C1, C2, C3 separately?
    # C1 = A1 B11 + A2 B21.
    # We can compute A1 B11 using 7 mults.
    # We can compute A2 B21 using 7 mults.
    # Total 14 mults for C1.
    # Similarly for C2 and C3.
    # Total 42 mults.
    #
    # This is a valid decomposition of rank 42.
    #
    # Let's construct the matrices U, V, W for this.
    #
    # For each of the 6 blocks (A1B11, A1B12, A1B13, A2B21, A2B22, A2B23):
    # We add 7 columns to U, V, W corresponding to the 7 Strassen multiplications.
    #
    # Strassen Mult 1: (A11+A22)(B11+B22)
    # U: A11, A22 get +1.
    # V: B11, B22 get +1.
    # W: C11, C22 get +1 (from M1).
    # Wait, W accumulates into C.
    # C11 gets M1. C22 gets M1.
    #
    # Let's define the Strassen terms for a generic 2x2 * 2x2.
    # Inputs: A (4 vals), B (4 vals).
    # Outputs: C (4 vals).
    #
    # Term 1: (A11+A22) * (B11+B22) -> contributes to C11, C22
    # Term 2: (A21+A22) * B11 -> contributes to C21, C22 (with -1)
    # Term 3: A11 * (B21-B22) -> contributes to C12, C21
    # Term 4: A22 * (B12-B11) -> contributes to C11, C21
    # Term 5: (A11+A12) * B22 -> contributes to C11, C12
    # Term 6: (A21-A11) * (B11+B12) -> contributes to C22
    # Term 7: (A12-A22) * (B21+B22) -> contributes to C11, C22
    
    # Let's verify the contributions for Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's map indices for Block (A_idx, B_idx) -> C_idx
    # A_idx is 0 or 1 (A1 or A2)
    # B_idx is (block_row, block_col) where block_row is 0 or 1 (B1 or B2), block_col is 0,1,2 (Bx1, Bx2, Bx3)
    # C_idx is block_col (0,1,2)
    
    # We will iterate over the 6 combinations.
    # A_block: 0, 1
    # B_block: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    # Note: B_block (r,c) corresponds to rows 2r, 2r+1 and cols 2c, 2c+1 of B.
    # A_block a corresponds to cols 2a, 2a+1 of A.
    
    # U indices: (i, j) for A[i,j]
    # V indices: (j, k) for B[j,k]
    # W indices: (k, i) for C[i,k] -> W[k*n+i]
    
    U = [[0] * 42 for _ in range(n * m)]
    V = [[0] * 42 for _ in range(m * p)]
    W = [[0] * 42 for _ in range(p * n)]
    
    col = 0
    for a_blk in range(2):
        for b_blk_r in range(2):
            for b_blk_c in range(3):
                # A indices
                a_r0 = 0
                a_r1 = 1
                a_c0 = a_blk * 2
                a_c1 = a_blk * 2 + 1
                
                # B indices
                b_r0 = b_blk_r * 2
                b_r1 = b_blk_r * 2 + 1
                b_c0 = b_blk_c * 2
                b_c1 = b_blk_c * 2 + 1
                
                # C indices
                c_r0 = 0
                c_r1 = 1
                c_c0 = b_blk_c * 2
                c_c1 = b_blk_c * 2 + 1
                
                # Helper to add to U, V, W
                def add_u(r, c, val):
                    U[r*m+c][col] += val
                def add_v(r, c, val):
                    V[r*p+c][col] += val
                def add_w(r, c, val):
                    # W index is k*n + i. Here r is k, c is i.
                    W[r*n+c][col] += val
                
                # Term 1: (A11+A22)(B11+B22)
                # A11 is (a_r0, a_c0), A22 is (a_r1, a_c1)
                # B11 is (b_r0, b_c0), B22 is (b_r1, b_c1)
                # C11 is (c_r0, c_c0), C22 is (c_r1, c_c1)
                add_u(a_r0, a_c0, 1)
                add_u(a_r1, a_c1, 1)
                add_v(b_r0, b_c0, 1)
                add_v(b_r1, b_c1, 1)
                add_w(c_c0, c_r0, 1) # C11 -> k=c_c0, i=c_r0
                add_w(c_c1, c_r1, 1) # C22 -> k=c_c1, i=c_r1
                col += 1
                
                # Term 2: (A21+A22)B11
                # A21 is (a_r1, a_c0), A22 is (a_r1, a_c1)
                # B11 is (b_r0, b_c0)
                # C21 is (c_r1, c_c0), C22 is (c_r1, c_c1)
                add_u(a_r1, a_c0, 1)
                add_u(a_r1, a_c1, 1)
                add_v(b_r0, b_c0, 1)
                add_w(c_c0, c_r1, 1) # C21
                add_w(c_c1, c_r1, -1) # C22 (from -M2)
                col += 1
                
                # Term 3: A11(B21-B22)
                # A11 is (a_r0, a_c0)
                # B21 is (b_r1, b_c0), B22 is (b_r1, b_c1)
                # C12 is (c_r0, c_c1), C21 is (c_r1, c_c0)
                add_u(a_r0, a_c0, 1)
                add_v(b_r1, b_c0, 1)
                add_v(b_r1, b_c1, -1)
                add_w(c_c1, c_r0, 1) # C12
                add_w(c_c0, c_r1, 1) # C21
                col += 1
                
                # Term 4: A22(B12-B11)
                # A22 is (a_r1, a_c1)
                # B12 is (b_r0, b_c1), B11 is (b_r0, b_c0)
                # C11 is (c_r0, c_c0), C21 is (c_r1, c_c0)
                add_u(a_r1, a_c1, 1)
                add_v(b_r0, b_c1, 1)
                add_v(b_r0, b_c0, -1)
                add_w(c_c0, c_r0, 1) # C11
                add_w(c_c0, c_r1, 1) # C21
                col += 1
                
                # Term 5: (A11+A12)B22
                # A11 is (a_r0, a_c0), A12 is (a_r0, a_c1)
                # B22 is (b_r1, b_c1)
                # C11 is (c_r0, c_c0), C12 is (c_r0, c_c1)
                add_u(a_r0, a_c0, 1)
                add_u(a_r0, a_c1, 1)
                add_v(b_r1, b_c1, 1)
                add_w(c_c0, c_r0, -1) # C11 (from -M5)
                add_w(c_c1, c_r0, 1) # C12
                col += 1
                
                # Term 6: (A21-A11)(B11+B12)
                # A21 is (a_r1, a_c0), A11 is (a_r0, a_c0)
                # B11 is (b_r0, b_c0), B12 is (b_r0, b_c1)
                # C22 is (c_r1, c_c1)
                add_u(a_r1, a_c0, 1)
                add_u(a_r0, a_c0, -1)
                add_v(b_r0, b_c0, 1)
                add_v(b_r0, b_c1, 1)
                add_w(c_c1, c_r1, 1) # C22
                col += 1
                
                # Term 7: (A12-A22)(B21+B22)
                # A12 is (a_r0, a_c1), A22 is (a_r1, a_c1)
                # B21 is (b_r1, b_c0), B22 is (b_r1, b_c1)
                # C11 is (c_r0, c_c0), C22 is (c_r1, c_c1)
                add_u(a_r0, a_c1, 1)
                add_u(a_r1, a_c1, -1)
                add_v(b_r1, b_c0, 1)
                add_v(b_r1, b_c1, 1)
                add_w(c_c0, c_r0, 1) # C11
                add_w(c_c1, c_r1, 1) # C22
                col += 1
    
    # Verify the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i,j,k] = 1 for all i,j,k.
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check a few entries
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(42):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if construction failed (should not happen)
        return best_result

# EVOLVE-BLOCK-END
