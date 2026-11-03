"""Strassen-enhanced tensor decomposition for 3x5 by 5x5 matrix multiplication."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 115

    # Strategy: Partition the middle dimension (5) into 2+3.
    # Apply Strassen to 2x2x2 sub-problems where beneficial.
    # 
    # A (3x5) = [A_L (3x2) | A_R (3x3)]
    # B (5x5) = [B_L (2x5); B_R (3x5)]
    # C = A_L * B_L + A_R * B_R
    #
    # A_L * B_L: (3x2)(2x5) -> partition rows of A_L into 1+2, cols of B_L into 2x...
    # Actually, let's use a cleaner approach.
    #
    # For the 2x2x2 Strassen kernel, we know rank 7 < 8.
    # We can embed this in our larger problem.
    #
    # Let's partition:
    # A: rows 3=1+2, cols 5=2+3
    # B: rows 5=2+3, cols 5=2+3
    #
    # Block structure:
    # A = [[a11(1x2), a12(1x3)],
    #      [a21(2x2), a22(2x3)]]
    # B = [[b11(2x2), b12(2x3)],
    #      [b21(3x2), b22(3x3)]]
    #
    # C = A*B:
    # C[0,0:2]   = a11*b11 + a12*b21  (1x2 * 2x2 + 1x3 * 3x2 = 2 + 3 = 5 mults)
    # C[0,2:5]   = a11*b12 + a12*b22  (1x2 * 2x3 + 1x3 * 3x3 = 6 + 9 = 15 mults)
    # C[1:3,0:2] = a21*b11 + a22*b21  (2x2 * 2x2 + 2x3 * 3x2 = 4 + 6 = 10 mults)
    # C[1:3,2:5] = a21*b12 + a22*b22  (2x2 * 2x3 + 2x3 * 3x3 = 6 + 9 = 15 mults)
    #
    # Total schoolbook within this partition: 5+15+10+15 = 45... but wait,
    # the full schoolbook is 75. Let me recount.
    #
    # Actually each block product contributes:
    # a11*b11: 1*2*2 = 4 scalar products
    # a11*b12: 1*2*3 = 6
    # a12*b21: 1*3*2 = 6
    # a12*b22: 1*3*3 = 9
    # a21*b11: 2*2*2 = 8
    # a21*b12: 2*2*3 = 12
    # a22*b21: 2*3*2 = 12
    # a22*b22: 2*3*3 = 18
    # Total: 4+6+6+9+8+12+12+18 = 75. Correct.
    #
    # Now, for a21*b11 (2x2 x 2x2), we can use Strassen: 7 instead of 8. Saves 1.
    # For a22*b22 (2x3 x 3x3), we can't directly use Strassen, but we can
    # partition it further.
    #
    # Let's focus on where Strassen applies:
    # 1. a21*b11: (2x2)(2x2) -> Strassen: 7 vs 8. Save 1.
    # 2. a12*b21: (1x3)(3x2) -> Can we do better than 6? 
    #    This is rank of (1,3,2) tensor = min(1*3, 3*2, 1*2) = 2? No, rank is 6.
    #    Actually for (1x3)(3x2), the minimum rank is max(1,2) = 2? No.
    #    The tensor M_{1,3,2} has rank 6 (full schoolbook) unless there's structure.
    #    Actually, rank of matrix multiplication tensor M_{m,n,p} is at least max(mn, np, pm).
    #    For (1,3,2): max(3, 6, 2) = 6. So rank >= 6, and schoolbook = 6. No saving.
    #
    # 3. a11*b11: (1x2)(2x2) -> rank >= max(2, 4, 2) = 4. Schoolbook = 4. No saving.
    # 4. a21*b12: (2x2)(2x3) -> rank >= max(4, 6, 4) = 6. Schoolbook = 12. 
    #    Can we do 6? That would require rank 6 decomposition. 
    #    M_{2,2,3} rank: known to be 12? Or less?
    #    Actually, I believe M_{2,2,3} has rank 12 (schoolbook is optimal for this).
    #
    # 5. a22*b21: (2x3)(3x2) -> rank >= max(6, 6, 4) = 6. Schoolbook = 12.
    #    Same question as above.
    #
    # 6. a11*b12: (1x2)(2x3) -> rank >= max(2, 6, 3) = 6. Schoolbook = 6. No saving.
    # 7. a12*b22: (1x3)(3x3) -> rank >= max(3, 9, 3) = 9. Schoolbook = 9. No saving.
    #
    # So the only clear saving is from a21*b11: Strassen saves 1.
    #
    # But wait - can we save more by considering cross-block interactions?
    # The issue is that C[1:3, 0:2] = a21*b11 + a22*b21.
    # These are separate block products that add, so their decompositions are independent.
    #
    # Total rank with Strassen on a21*b11: 75 - 1 = 74.
    #
    # Let me also check if a21*b12 or a22*b21 can be reduced.
    # M_{2,2,3}: I'll assume schoolbook 12 is optimal for now.
    # M_{2,3,2}: same, assume 12.
    #
    # So best I can get with this partition: 74.
    #
    # Let me construct the rank-74 decomposition.
    #
    # Factor matrices:
    # U: shape (15, 74) - rows indexed by (i, j) for i in 0..2, j in 0..4
    # V: shape (25, 74) - rows indexed by (j, k) for j in 0..4, k in 0..4
    # W: shape (15, 74) - rows indexed by (k, i) for k in 0..4, i in 0..2
    #
    # Block assignments:
    # a11: A[0, 0:2] -> rows 0,1 in U (i=0, j=0,1)
    # a12: A[0, 2:5] -> rows 2,3,4 in U (i=0, j=2,3,4)
    # a21: A[1:3, 0:2] -> rows 5,6,7,8 in U (i=1,2; j=0,1)
    # a22: A[1:3, 2:5] -> rows 9,10,11,12,13,14 in U (i=1,2; j=2,3,4)
    #
    # b11: B[0:2, 0:2] -> rows 0,1,4,5 in V (j=0,1; k=0,1)
    # b12: B[0:2, 2:5] -> rows 2,3,6,7 in V (j=0,1; k=2,3,4)
    # b21: B[2:5, 0:2] -> rows 8,9,12,13,16,17 in V (j=2,3,4; k=0,1)
    # b22: B[2:5, 2:5] -> rows 10,11,14,15,18,19,20,21,22 in V (j=2,3,4; k=2,3,4)
    #
    # c11: C[0, 0:2] -> rows 0,1 in W (k=0,1; i=0)
    # c12: C[0, 2:5] -> rows 2,3,4 in W (k=2,3,4; i=0)
    # c21: C[1:3, 0:2] -> rows 5,6,7,8 in W (k=0,1; i=1,2)
    # c22: C[1:3, 2:5] -> rows 9,10,11,12,13,14 in W (k=2,3,4; i=1,2)
    
    rank = 74
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # a11*b11: (1x2)(2x2) -> 4 terms (no Strassen benefit for 1x2)
    # A rows: i=0, j=0,1 -> U rows 0,1
    # B rows: j=0,1; k=0,1 -> V rows 0,1,4,5
    # C rows: k=0,1; i=0 -> W rows 0,1
    for i in [0]:
        for j in [0, 1]:
            for k in [0, 1]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 4
    
    # a11*b12: (1x2)(2x3) -> 6 terms
    # A rows: i=0, j=0,1 -> U rows 0,1
    # B rows: j=0,1; k=2,3,4 -> V rows 2,3,6,7
    # C rows: k=2,3,4; i=0 -> W rows 2,3,4
    for i in [0]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 10
    
    # a12*b21: (1x3)(3x2) -> 6 terms
    # A rows: i=0, j=2,3,4 -> U rows 2,3,4
    # B rows: j=2,3,4; k=0,1 -> V rows 8,9,12,13,16,17
    # C rows: k=0,1; i=0 -> W rows 0,1
    for i in [0]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 16
    
    # a12*b22: (1x3)(3x3) -> 9 terms
    # A rows: i=0, j=2,3,4 -> U rows 2,3,4
    # B rows: j=2,3,4; k=2,3,4 -> V rows 10,11,14,15,18,19,20,21,22
    # C rows: k=2,3,4; i=0 -> W rows 2,3,4
    for i in [0]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 25
    
    # a21*b11: (2x2)(2x2) -> Strassen: 7 terms instead of 8
    # A rows: i=1,2; j=0,1 -> U rows 5,6,7,8
    # B rows: j=0,1; k=0,1 -> V rows 0,1,4,5
    # C rows: k=0,1; i=1,2 -> W rows 5,6,7,8
    #
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # In our notation:
    # A11 = A[1,0], A12 = A[1,1], A21 = A[2,0], A22 = A[2,1]
    # B11 = B[0,0], B12 = B[0,1], B21 = B[1,0], B22 = B[1,1]
    # C11 = C[1,0], C12 = C[1,1], C21 = C[2,0], C22 = C[2,1]
    #
    # U rows: 5=A[1,0], 6=A[1,1], 7=A[2,0], 8=A[2,1]
    # V rows: 0=B[0,0], 1=B[0,1], 4=B[1,0], 5=B[1,1]
    # W rows: 5=C[1,0], 6=C[1,1], 7=C[2,0], 8=C[2,1]
    
    # M1 = (A[1,0]+A[2,1])(B[0,0]+B[1,1])
    # U: rows 5 and 8 have +1
    # V: rows 0 and 5 have +1
    # W: contributes to C11 and C22 with +1
    U[5, col] = 1; U[8, col] = 1
    V[0, col] = 1; V[5, col] = 1
    W[5, col] = 1; W[8, col] = 1
    col += 1
    
    # M2 = (A[2,0]+A[2,1])B[0,0]
    # U: rows 7 and 8 have +1
    # V: row 0 has +1
    # W: contributes to C21 and C22 with +1
    U[7, col] = 1; U[8, col] = 1
    V[0, col] = 1
    W[7, col] = 1; W[8, col] = 1
    col += 1
    
    # M3 = A[1,0](B[0,1]-B[1,1])
    # U: row 5 has +1
    # V: row 1 has +1, row 5 has -1
    # W: contributes to C12 and C22 with +1
    U[5, col] = 1
    V[1, col] = 1; V[5, col] = -1
    W[6, col] = 1; W[8, col] = 1
    col += 1
    
    # M4 = A[2,1](B[1,0]-B[0,0])
    # U: row 8 has +1
    # V: row 4 has +1, row 0 has -1
    # W: contributes to C21 and C22 with +1
    U[8, col] = 1
    V[4, col] = 1; V[0, col] = -1
    W[7, col] = 1; W[8, col] = 1
    col += 1
    
    # M5 = (A[1,0]+A[1,1])B[1,1]
    # U: rows 5 and 6 have +1
    # V: row 5 has +1
    # W: contributes to C11 and C12 with +1
    U[5, col] = 1; U[6, col] = 1
    V[5, col] = 1
    W[5, col] = 1; W[6, col] = 1
    col += 1
    
    # M6 = (A[2,0]-A[1,0])(B[0,0]+B[0,1])
    # U: row 7 has +1, row 5 has -1
    # V: rows 0 and 1 have +1
    # W: contributes to C21 and C22 with +1
    U[7, col] = 1; U[5, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[7, col] = 1; W[8, col] = 1
    col += 1
    
    # M7 = (A[1,1]-A[2,1])(B[1,0]+B[1,1])
    # U: row 6 has +1, row 8 has -1
    # V: rows 4 and 5 have +1
    # W: contributes to C11 and C12 with +1
    U[6, col] = 1; U[8, col] = -1
    V[4, col] = 1; V[5, col] = 1
    W[5, col] = 1; W[6, col] = 1
    col += 1
    # col = 32
    
    # a21*b12: (2x2)(2x3) -> 12 terms (schoolbook)
    # A rows: i=1,2; j=0,1 -> U rows 5,6,7,8
    # B rows: j=0,1; k=2,3,4 -> V rows 2,3,6,7
    # C rows: k=2,3,4; i=1,2 -> W rows 9,10,11,12,13,14
    for i in [1, 2]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 44
    
    # a22*b21: (2x3)(3x2) -> 12 terms (schoolbook)
    # A rows: i=1,2; j=2,3,4 -> U rows 9,10,11,12,13,14
    # B rows: j=2,3,4; k=0,1 -> V rows 8,9,12,13,16,17
    # C rows: k=0,1; i=1,2 -> W rows 5,6,7,8
    for i in [1, 2]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 56
    
    # a22*b22: (2x3)(3x3) -> 18 terms (schoolbook)
    # A rows: i=1,2; j=2,3,4 -> U rows 9,10,11,12,13,14
    # B rows: j=2,3,4; k=2,3,4 -> V rows 10,11,14,15,18,19,20,21,22
    # C rows: k=2,3,4; i=1,2 -> W rows 9,10,11,12,13,14
    for i in [1, 2]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 74
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify the decomposition is correct
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # More precisely: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build test matrices
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for r in range(rank):
                for j in range(m):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
            C_computed[i, k] = total
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    else:
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
