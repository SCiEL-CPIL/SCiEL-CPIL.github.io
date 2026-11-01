import numpy as np

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # A is 2x3: A[i][j] = a_{i,j}
    # B is 3x4: B[j][k] = b_{j,k}
    # C is 2x4: C[i][k] = c_{i,k}
    # Indexing: U[i*m + j, col], V[j*p + k, col], W[k*n + i, col]
    
    # M1 = A[:, 0:2] * B[0:2, :]
    # This is a 2x2 block multiplication.
    # M1[i, k] = sum_{j=0}^{1} A[i, j] * B[j, k]
    # For i in {0,1}, k in {0,3} (since B has 4 cols, we take first 2? No, B is 3x4.
    # Wait, M1 should be a scalar.
    # Let's define M1 = (A[:, 0:2]) * (B[0:2, :])? No, that's a 2x4 matrix.
    # We need scalar multiplies.
    # Let's use the 2x2 Strassen on the 2x2 submatrices.
    
    # Let's decompose C = A * B.
    # A is 2x3, B is 3x4.
    # Split A into A1 (2x2, cols 0,1) and A2 (2x1, col 2).
    # Split B into B1 (2x4, rows 0,1) and B2 (1x4, row 2).
    # C = A1 * B1 + A2 * B2.
    # A1 * B1 is 2x4. A2 * B2 is 2x4.
    # A2 * B2 is rank 1 (outer product of A2 (2x1) and B2 (1x4)).
    # We can compute A2 * B2 with 1 multiply? No, A2 * B2[i, k] = A2[i] * B2[k].
    # This requires 2*4 = 8 multiplies if done naively.
    # But we can use rank 1 update? No, we need the tensor decomposition.
    # The term A2 * B2 contributes to C[i, k] as A2[i] * B2[k].
    # This is a rank 1 tensor in the decomposition sense?
    # T_{i,k, j} = A[i,j] B[j,k].
    # The part from j=2 is A[i,2] B[2,k].
    # This can be written as sum_{r=1}^{8} u_r[i] v_r[k] w_r[2]?
    # Actually, for fixed j=2, the tensor is A[:,2] B[2,:].
    # This is a rank 1 tensor in the 3-mode? No.
    # It is a matrix.
    # In the decomposition C = sum_r U_r V_r W_r^T,
    # the contribution from j=2 is U_r[i] V_r[k] W_r[2].
    # We want U_r[i] V_r[k] W_r[2] to sum to A[i,2] B[2,k].
    # This is possible with rank 1 if A[:,2] and B[2,:] are vectors?
    # No, U_r is a vector, V_r is a vector.
    # If we set W_r[2] = 1, then U_r[i] V_r[k] must sum to A[i,2] B[2,k].
    # This is an outer product of A[:,2] and B[2,:].
    # So we can just use 1 term?
    # U_r = A[:,2], V_r = B[2,:], W_r[2] = 1.
    # But U and V must be constant vectors, not depending on the input!
    # The decomposition is linear.
    # U_r[i] is a coefficient for A[i, j]? No.
    # U is a matrix of size (n*m) x rank.
    # U[i*m+j, r] is the coefficient of A[i,j] in the r-th multiply.
    # So for the j=2 part:
    # We need multiplies M_r such that sum_r M_r * (coeff in W) = C.
    # The term A[i,2] B[2,k] is a single product of two linear forms?
    # No, it's a product of A[i,2] and B[2,k].
    # This is a "pure" term.
    # In the tensor T_{i,k,j}, the slice j=2 is A[:,2] B[2,:].
    # This slice has rank 1 (as a matrix).
    # So we can decompose it with 1 term?
    # Term r: U_r has 1 at (i, 2) for all i? No.
    # U_r[i*m + 2] = 1.
    # V_r[2*p + k] = 1.
    # W_r[k*n + i] = 1.
    # Then sum_r U_r[i*m+2] V_r[2*p+k] W_r[k*n+i] = 1 * 1 * 1 = 1.
    # This gives C[i,k] += 1.
    # We need C[i,k] += A[i,2] B[2,k].
    # So we need the r-th multiply to be A[i,2] B[2,k].
    # This requires U_r[i*m+2] = 1, V_r[2*p+k] = 1, W_r[k*n+i] = 1.
    # But this is for ALL i, k.
    # So we need 8 such terms?
    # Term r_{i,k}: U_r[i*m+2]=1, V_r[2*p+k]=1, W_r[k*n+i]=1.
    # This uses 8 multiplies.
    # Can we do better?
    # The tensor slice is rank 1.
    # The rank of the tensor slice is the minimum number of terms.
    # So 8 is the minimum for this part if we isolate it.
    # But we can mix it with the other part.
    
    # Let's look at A1 * B1.
    # A1 is 2x2, B1 is 2x4.
    # C1 = A1 * B1 is 2x4.
    # This is a 2x2 by 2x4 multiplication.
    # Rank of 2x2 by 2x2 is 7 (Strassen).
    # Rank of 2x2 by 2x4?
    # We can split B1 into B1a (2x2) and B1b (2x2).
    # C1a = A1 * B1a (2x2), C1b = A1 * B1b (2x2).
    # Each 2x2 by 2x2 can be done in 7 multiplies.
    # Total 14 multiplies.
    # Plus 8 for the A2 * B2 part.
    # Total 22.
    
    # Let's verify the indices.
    # A indices: i in {0,1}, j in {0,1,2}.
    # B indices: j in {0,1,2}, k in {0,1,2,3}.
    # C indices: i in {0,1}, k in {0,1,2,3}.
    
    # Part 1: j in {0,1}.
    # A1 = A[:, 0:2], B1 = B[0:2, :].
    # C1 = A1 * B1.
    # Split B1 into B1a = B[0:2, 0:2] and B1b = B[0:2, 2:4].
    # C1a = A1 * B1a (2x2).
    # C1b = A1 * B1b (2x2).
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A1[0,0]+A1[1,1]) * (B1a[0,0]+B1a[1,1])
    # M2 = (A1[1,0]+A1[1,1]) * B1a[0,0]
    # M3 = A1[0,0] * (B1a[0,1]-B1a[1,1])
    # M4 = A1[1,1] * (B1a[1,0]-B1a[0,0])
    # M5 = (A1[0,0]+A1[0,1]) * B1a[1,1]
    # M6 = (A1[1,0]-A1[0,0]) * (B1a[0,0]+B1a[0,1])
    # M7 = (A1[0,1]-A1[1,1]) * (B1a[1,0]+B1a[1,1])
    
    # C1a[0,0] = M1 + M4 - M5 + M7
    # C1a[0,1] = M3 + M5
    # C1a[1,0] = M2 + M4
    # C1a[1,1] = M1 - M2 + M3 + M6
    
    # Similarly for C1b with B1b.
    # B1b[0,0] = B[0,2], B1b[0,1] = B[0,3]
    # B1b[1,0] = B[1,2], B1b[1,1] = B[1,3]
    
    # Part 2: j=2.
    # C2 = A[:, 2] * B[2, :].
    # C2[i, k] = A[i, 2] * B[2, k].
    # This requires 8 multiplies (one for each i, k pair).
    # Or can we use Strassen-like tricks?
    # The tensor is rank 1 in the matrix sense, but in the decomposition sense,
    # we need to form the product A[i,2] B[2,k].
    # Since A[i,2] and B[2,k] are independent variables, we need 8 multiplies.
    
    # Total rank = 7 + 7 + 8 = 22.
    
    # Let's build the matrices.
    
    # M1-M7 for C1a (k=0,1)
    # M1: A[0,0]+A[1,1] * B[0,0]+B[1,1]
    # U: A[0,0]=1, A[1,1]=1. V: B[0,0]=1, B[1,1]=1.
    # W: C[0,0]=1, C[1,1]=1.
    
    # Let's write a helper to add a term.
    def add_term(r, u_indices, v_indices, w_indices):
        for idx in u_indices:
            U[idx, r] = 1
        for idx in v_indices:
            V[idx, r] = 1
        for idx in w_indices:
            W[idx, r] = 1
            
    def add_term_neg(r, u_indices, v_indices, w_indices):
        for idx in u_indices:
            U[idx, r] = -1
        for idx in v_indices:
            V[idx, r] = 1
        for idx in w_indices:
            W[idx, r] = 1

    # Term 0: M1a
    # (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: 0, 4 (A[0,0], A[1,1])
    # V: 0, 5 (B[0,0], B[1,1])
    # W: 0, 3 (C[0,0], C[1,1])
    add_term(0, [0, 4], [0, 5], [0, 3])
    
    # Term 1: M2a
    # (A[1,0]+A[1,1]) * B[0,0]
    # U: 3, 4
    # V: 0
    # W: 2 (C[1,0]), -1 at 3 (C[1,1]) -> Wait, W is just 1s?
    # The coefficient is in W?
    # No, W[k*n+i, r] is the coefficient.
    # So W[2, 1] = 1, W[3, 1] = -1.
    # But my add_term sets W to 1.
    # I need to handle signs in W or U/V.
    # Let's put signs in U or V.
    # M2 = (A[1,0]+A[1,1]) * B[0,0].
    # Contribution to C[1,0] is +1.
    # Contribution to C[1,1] is -1.
    # So W[2, 1] = 1, W[3, 1] = -1.
    
    # Let's modify add_term to take W values.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_term_full(r, u_vals, v_vals, w_vals):
        # u_vals: list of (index, value)
        for idx, val in u_vals:
            U[idx, r] = val
        for idx, val in v_vals:
            V[idx, r] = val
        for idx, val in w_vals:
            W[idx, r] = val

    # Term 0: M1a
    # (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # C[0,0] += 1, C[1,1] += 1
    add_term_full(0, [(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)])
    
    # Term 1: M2a
    # (A[1,0]+A[1,1]) * B[0,0]
    # C[1,0] += 1, C[1,1] -= 1
    add_term_full(1, [(3, 1), (4, 1)], [(0, 1)], [(2, 1), (3, -1)])
    
    # Term 2: M3a
    # A[0,0] * (B[0,1]-B[1,1])
    # C[0,1] += 1, C[1,1] += 1
    add_term_full(2, [(0, 1)], [(1, 1), (5, -1)], [(1, 1), (3, 1)])
    
    # Term 3: M4a
    # A[1,1] * (B[1,0]-B[0,0])
    # C[0,0] += 1, C[1,0] += 1
    add_term_full(3, [(4, 1)], [(4, 1), (0, -1)], [(0, 1), (2, 1)])
    
    # Term 4: M5a
    # (A[0,0]+A[0,1]) * B[1,1]
    # C[0,0] -= 1, C[0,1] += 1
    add_term_full(4, [(0, 1), (1, 1)], [(5, 1)], [(0, -1), (1, 1)])
    
    # Term 5: M6a
    # (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # C[1,1] += 1
    add_term_full(5, [(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    
    # Term 6: M7a
    # (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # C[0,0] += 1
    add_term_full(6, [(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)])
    
    # Now for C1b (k=2,3)
    # B1b[0,0] = B[0,2] (idx 2)
    # B1b[0,1] = B[0,3] (idx 3)
    # B1b[1,0] = B[1,2] (idx 6)
    # B1b[1,1] = B[1,3] (idx 7)
    
    # Term 7: M1b
    # (A[0,0]+A[1,1]) * (B[0,2]+B[1,3])
    # C[0,2] += 1, C[1,3] += 1
    add_term_full(7, [(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)])
    
    # Term 8: M2b
    # (A[1,0]+A[1,1]) * B[0,2]
    # C[1,2] += 1, C[1,3] -= 1
    add_term_full(8, [(3, 1), (4, 1)], [(2, 1)], [(6, 1), (7, -1)])
    
    # Term 9: M3b
    # A[0,0] * (B[0,3]-B[1,3])
    # C[0,3] += 1, C[1,3] += 1
    add_term_full(9, [(0, 1)], [(3, 1), (7, -1)], [(5, 1), (7, 1)])
    
    # Term 10: M4b
    # A[1,1] * (B[1,2]-B[0,2])
    # C[0,2] += 1, C[1,2] += 1
    add_term_full(10, [(4, 1)], [(6, 1), (2, -1)], [(4, 1), (6, 1)])
    
    # Term 11: M5b
    # (A[0,0]+A[0,1]) * B[1,3]
    # C[0,2] -= 1, C[0,3] += 1
    add_term_full(11, [(0, 1), (1, 1)], [(7, 1)], [(4, -1), (5, 1)])
    
    # Term 12: M6b
    # (A[1,0]-A[0,0]) * (B[0,2]+B[0,3])
    # C[1,3] += 1
    add_term_full(12, [(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)])
    
    # Term 13: M7b
    # (A[0,1]-A[1,1]) * (B[1,2]+B[1,3])
    # C[0,2] += 1
    add_term_full(13, [(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)])
    
    # Now for C2 (j=2)
    # C2[i, k] = A[i, 2] * B[2, k].
    # A[i, 2] is at index i*3 + 2.
    # i=0: idx 2. i=1: idx 5.
    # B[2, k] is at index 2*4 + k = 8 + k.
    # k=0: 8, k=1: 9, k=2: 10, k=3: 11.
    # C[i, k] is at index k*2 + i.
    # k=0, i=0: 0. k=0, i=1: 1.
    # k=1, i=0: 2. k=1, i=1: 3.
    # k=2, i=0: 4. k=2, i=1: 5.
    # k=3, i=0: 6. k=3, i=1: 7.
    
    # Term 14: A[0,2] * B[2,0] -> C[0,0]
    add_term_full(14, [(2, 1)], [(8, 1)], [(0, 1)])
    
    # Term 15: A[0,2] * B[2,1] -> C[0,1]
    add_term_full(15, [(2, 1)], [(9, 1)], [(2, 1)])
    
    # Term 16: A[0,2] * B[2,2] -> C[0,2]
    add_term_full(16, [(2, 1)], [(10, 1)], [(4, 1)])
    
    # Term 17: A[0,2] * B[2,3] -> C[0,3]
    add_term_full(17, [(2, 1)], [(11, 1)], [(6, 1)])
    
    # Term 18: A[1,2] * B[2,0] -> C[1,0]
    add_term_full(18, [(5, 1)], [(8, 1)], [(1, 1)])
    
    # Term 19: A[1,2] * B[2,1] -> C[1,1]
    add_term_full(19, [(5, 1)], [(9, 1)], [(3, 1)])
    
    # Term 20: A[1,2] * B[2,2] -> C[1,2]
    add_term_full(20, [(5, 1)], [(10, 1)], [(5, 1)])
    
    # Term 21: A[1,2] * B[2,3] -> C[1,3]
    add_term_full(21, [(5, 1)], [(11, 1)], [(7, 1)])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
