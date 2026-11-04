"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: 150s limit, use 120s for safety
    deadline = time.time() + 120
    
    # Theoretical lower bound is 32. Current seed uses 56 (7*8).
    # We will try to find a rank 32 decomposition using Strassen-like block structures
    # or a direct search for a lower rank.
    
    # Let's first try to see if we can construct a rank 32 solution.
    # A 2x4 * 4x8 multiplication.
    # Can be viewed as 2 (2x2) * (2x2) blocks?
    # A is 2x4, B is 4x8.
    # Let A = [A1 A2] where A1, A2 are 2x2.
    # Let B = [[B11 B12] [B21 B22]] where Bij are 2x4.
    # C = A1*B11 + A2*B21 (top half of C, 2x4)
    #   A1*B12 + A2*B22 (bottom half of C, 2x4)
    
    # If we use Strassen for each 2x2 * 2x2 block, we need 7 multiplications per block.
    # There are 4 blocks (A1B11, A1B12, A2B21, A2B22). Total 28 multiplications.
    # But wait, Strassen combines them.
    # M1 = (A11+A21)(B11+B21)
    # M2 = (A11+A21)(B12-B22)
    # M3 = (A12-A22)(B11+B12)
    # M4 = (A12-A22)(B21-B22)
    # M5 = (A11+A21)(B11+B21) -- wait, standard Strassen is for 2x2 * 2x2.
    
    # Let's check the block structure more carefully.
    # A is 2x4. B is 4x8.
    # Split B into top 2 rows and bottom 2 rows?
    # B_top is 2x8, B_bottom is 2x8.
    # C = A * B_top + ... no, A is 2x4.
    # A * B = [A_col1 A_col2 A_col3 A_col4] * [B_row1; B_row2; B_row3; B_row4]
    
    # Let's try a different block split.
    # A = [A_left A_right] (2x2 each)
    # B = [B_left; B_right] (2x8 each, so 2 rows of 8)
    # C = A_left * B_left + A_right * B_right
    # This is sum of two products: (2x2)*(2x8) and (2x2)*(2x8).
    # Each (2x2)*(2x8) can be done with rank 4 (standard) or lower?
    # Rank of (2x2)*(2x8) tensor is 4? No, standard is 4.
    # Can we do better?
    # The tensor for (2x2)*(2x8) has dimensions 4, 2, 8.
    # It is equivalent to 4 independent 2x1 * 1x8? No.
    # It is 2x2 * 2x8.
    # We can treat it as 4 terms: a11*b11, a12*b21, a21*b12, a22*b22? No.
    # C11 = a11*b11 + a12*b21
    # C12 = a11*b12 + a12*b22
    # C21 = a21*b11 + a22*b21
    # C22 = a21*b12 + a22*b22
    
    # This is just standard multiplication. Rank 4.
    # Can we use Strassen? Strassen is for 2x2 * 2x2.
    # Here B is 2x8. We can split B into four 2x2 blocks?
    # B = [B1 B2 B3 B4] where Bk are 2x2.
    # Then A*B = A*B1 + A*B2 + A*B3 + A*B4.
    # Each A*Bk is 2x2 * 2x2. Rank 7 each? Total 28?
    # But we can share terms if we combine B's first.
    
    # Let's try to construct a rank 32 solution by combining Strassen blocks.
    # If we split B into 4 blocks of 2x2, we have 4 independent 2x2*2x2 multiplications.
    # If we did them independently, rank 7*4 = 28.
    # Wait, the seed program used 56. Why?
    # Seed:
    # Group 1: A1 * Btop. A1 is 2x2. Btop is 2x8.
    # It iterates q in range(4). For each q, it does a 2x2 * 2x2 Strassen.
    # So it treats Btop as 4 blocks of 2x2.
    # So Group 1 is 4 * 7 = 28 terms.
    # Group 2 is 4 * 7 = 28 terms.
    # Total 56.
    
    # Can we do better than 28 for A1 * Btop?
    # A1 is 2x2. Btop is 2x8.
    # C_top = A1 * Btop.
    # This is a 2x2 matrix times a 2x8 matrix.
    # Is there a rank < 28 decomposition?
    # The tensor is T_{i,j,k} = A_{i,j} B_{j,k}.
    # i in 0..1, j in 0..1, k in 0..7.
    # This is the Kronecker product of T_{2,2,2} and I_4?
    # No.
    # T_{i,j,k} = sum_j A_{i,j} B_{j,k}.
    # This is equivalent to 4 independent 2x1 * 1x2? No.
    # For each k, we have a 2x2 * 2x1 multiplication?
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is just matrix-vector multiplication for each column k.
    # There are 8 columns.
    # So it's 8 independent 2x2 * 2x1 multiplications.
    # Rank of 2x2 * 2x1 is 2? (a1*x1 + a2*x2, b1*x1 + b2*x2).
    # U: [a1, b1], [a2, b2]. V: [x1, x1], [x2, x2]. W: [1, 1], [1, 1].
    # Rank 2.
    # So 8 * 2 = 16 terms?
    # Let's check.
    # Term 1: (a1, b1) * (x1, x1) -> (x1, x1). No, that's not right.
    # C1 = a1*x1 + a2*x2.
    # C2 = b1*x1 + b2*x2.
    # We need to compute this for each k.
    # Let's use 2 terms per k.
    # Term k1: U=[a1, b1], V=[x1, x1], W=[1, 1]. -> C1 += a1*x1, C2 += b1*x1.
    # Term k2: U=[a2, b2], V=[x2, x2], W=[1, 1]. -> C1 += a2*x2, C2 += b2*x2.
    # This works!
    # So A1 * Btop can be done in 16 terms.
    # Similarly A2 * Bbottom can be done in 16 terms.
    # Total rank 32.
    
    # Let's verify this logic.
    # A1 is 2x2. Btop is 2x8.
    # A1 = [[a, b], [c, d]].
    # Btop = [[e1, e2, ..., e8], [f1, f2, ..., f8]].
    # C = [[a*e1 + b*f1, ...], [c*e1 + d*f1, ...]].
    
    # Term 1 (for k=0):
    # U = [a, c] (column 0 of A1)
    # V = [e1, e1] (row 0 of Btop, repeated? No.)
    # Wait.
    # U shape (n*m, rank) = (4, rank).
    # V shape (m*p, rank) = (16, rank).
    # W shape (p*n, rank) = (16, rank).
    
    # Let's map indices.
    # U[i, r] = A[i, j_r]
    # V[j, r] = B[j, k_r]
    # W[k, r] = coeff
    
    # For term r corresponding to k=0:
    # We want to add A[:, 0] * B[0, 0] to C[:, 0].
    # And A[:, 1] * B[1, 0] to C[:, 0].
    
    # Term 0:
    # U[0, 0] = 1 (a), U[1, 0] = 0 (c? no, c is A[1,0])
    # Let's use 0-based indexing.
    # A is 2x4. A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d.
    # B is 4x8. B[0,0]=e1, B[1,0]=f1.
    # C is 2x8. C[0,0] = a*e1 + b*f1. C[1,0] = c*e1 + d*f1.
    
    # Term 0:
    # U: [1, 0, 0, 0] (a)
    # V: [1, 0, 0, 0, ...] (e1 at index 0)
    # W: [1, 0, ...] (C[0,0] += 1)
    # This adds a*e1 to C[0,0].
    
    # Term 1:
    # U: [0, 1, 0, 0] (b)
    # V: [0, 1, 0, 0, ...] (f1 at index 1)
    # W: [1, 0, ...] (C[0,0] += 1)
    # This adds b*f1 to C[0,0].
    
    # Term 2:
    # U: [0, 0, 1, 0] (c)
    # V: [1, 0, 0, 0, ...] (e1)
    # W: [0, 1, ...] (C[1,0] += 1)
    # This adds c*e1 to C[1,0].
    
    # Term 3:
    # U: [0, 0, 0, 1] (d)
    # V: [0, 1, 0, 0, ...] (f1)
    # W: [0, 1, ...] (C[1,0] += 1)
    # This adds d*f1 to C[1,0].
    
    # So 4 terms per column k.
    # 8 columns -> 32 terms for A1 * Btop?
    # Wait, my previous calculation said 16. Why?
    # I said: Term k1: U=[a1, b1]...
    # U=[a, c] is column 0 of A.
    # V=[e1, e1] is row 0 of B?
    # If V[0]=e1, V[1]=e1.
    # Then U[0]*V[0] + U[1]*V[1] = a*e1 + c*e1 = (a+c)*e1.
    # This is not what we want. We want a*e1 and c*e1 separately.
    # Unless we combine them in W?
    # No, W is fixed.
    # So we need separate terms for a and c.
    # So 4 terms per column.
    # Total 32 terms for A1 * Btop.
    # And 32 terms for A2 * Bbottom.
    # Total 64.
    # This is worse than 56.
    
    # So the standard "distribute" method is not better.
    
    # Let's stick to the 56 rank solution but try to optimize the constants or find a slightly better structure.
    # Or maybe there is a rank 48 or 40 solution?
    
    # Given the time constraint, I will return the 56 rank solution as it is valid.
    # I will add a check to ensure it runs fast.
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    current_rank = 0
    
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    val = u_loc[i * 2 + j_loc]
                    if val != 0:
                        U[i * m + j_loc][r] = val
            for j_loc in range(2):
                for k_loc in range(2):
                    val = v_loc[j_loc * 2 + k_loc]
                    if val != 0:
                        V[j_loc * p + (2 * q + k_loc)][r] = val
            for i in range(2):
                for k_loc in range(2):
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            current_rank += 1
            
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    val = u_loc[i * 2 + j_loc]
                    if val != 0:
                        U[i * m + (j_loc + 2)][r] = val
            for j_loc in range(2):
                for k_loc in range(2):
                    val = v_loc[j_loc * 2 + k_loc]
                    if val != 0:
                        V[(j_loc + 2) * p + (2 * q + k_loc)][r] = val
            for i in range(2):
                for k_loc in range(2):
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            current_rank += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
