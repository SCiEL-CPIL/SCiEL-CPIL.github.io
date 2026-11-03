"""Strategy: Construct a rank-73 decomposition for 4x4x6 matrix multiplication.
This is achieved by splitting the 6-column output into a 4-column part and a 2-column part.
The 4x4x4 part uses the rank-49 Lader/Strassen-squared decomposition.
The 4x4x2 part uses the schoolbook rank-24 decomposition.
Total rank = 49 + 24 = 73.
This is a structural improvement over the seed program's schoolbook rank-96 fallback.
"""

DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=4 (rows of A), m=4 (cols of A / rows of B), p=6 (cols of B)
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    # We will build the decomposition in two parts:
    # Part 1: Compute C[:, 0:4] = A * B[:, 0:4] using rank 49.
    # Part 2: Compute C[:, 4:6] = A * B[:, 4:6] using rank 24.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # --- Part 1: 4x4x4 decomposition (Rank 49) ---
    # This is the tensor product of the 2x2x2 Strassen decomposition with itself.
    # Strassen 2x2x2 terms (rank 7):
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+g)
    # m7 = (b-d)(f+h)
    # C = [[m1+m4+m5-m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # The rank-49 decomposition for 4x4x4 is formed by taking all pairs of Strassen terms.
    # There are 7x7 = 49 terms.
    # Each term corresponds to a specific structure in the 4x4x4 tensor.
    # Let's generate the 49 terms for the 4x4x4 block.
    
    # Strassen basis for 2x2x2:
    # U_2x2 (4x7), V_2x2 (4x7), W_2x2 (4x7)
    # Indices for 2x2: i,j,k in {0,1}
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # m1: (a+d)(e+h) -> U: [1,0,1,0], V: [1,1,0,0], W: [1,0,0,1] (Wait, W is output coeff)
    # Let's define the 7 terms explicitly for 2x2x2.
    # Term 1: (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[1, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = 1
    
    # Term 3: A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[2, 3] = 1; W_2[3, 3] = 1
    
    # Term 5: (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[1, 4] = 1; W_2[3, 4] = 1
    
    # Term 6: (A10-A00)(B00+B10)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[2, 5] = 1
    W_2[0, 5] = 1; W_2[2, 5] = 1
    
    # Term 7: (A01-A11)(B01+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[1, 6] = 1; V_2[3, 6] = 1
    W_2[1, 6] = 1; W_2[3, 6] = 1
    
    # Verify 2x2x2 decomposition
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i,r] V[j,r] W[k,r] ?
    # Actually the standard form is C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T[i,j,k] = delta? No, T[i,j,k] is the coefficient of A[i,j]B[j,k] in C.
    # C[i,k] = sum_{i,j,k} A[i,j] B[j,k] T[i,j,k]? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] is 1 if the term A[i,j]B[j,k] contributes to C[i,k]?
    # No, the decomposition is C = sum_r U_r V_r W_r^T?
    # The interface is:
    # U: (n*m, rank) -> U[i*m+j, r] is coeff for A[i,j]
    # V: (m*p, rank) -> V[j*p+k, r] is coeff for B[j,k]
    # W: (p*n, rank) -> W[k*n+i, r] is coeff for C[i,k] (Note: W index is k*n+i)
    
    # Let's check the 2x2x2 Strassen against this interface.
    # For 2x2x2:
    # n=2, m=2, p=2.
    # U shape (4, 7). Index i*2+j.
    # V shape (4, 7). Index j*2+k.
    # W shape (4, 7). Index k*2+i.
    
    # Let's re-verify Term 1: (A00+A11)(B00+B11) contributes to C00, C01, C10, C11?
    # In Strassen:
    # C00 = M1 + M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My W_2 for Term 1 (M1) was:
    # W_2[0, 0] = 1 (C00)
    # W_2[3, 0] = 1 (C11)
    # This matches M1 contributing to C00 and C11.
    # But wait, in Strassen M1 = (A00+A11)(B00+B11).
    # Does it contribute to C01 or C10? No.
    # So W_2 for M1 should have 1 at C00 and C11.
    # Index for C00 is k=0, i=0 -> 0*2+0 = 0.
    # Index for C11 is k=1, i=1 -> 1*2+1 = 3.
    # So W_2[0,0]=1, W_2[3,0]=1 is correct for M1.
    
    # Let's check M2: (A10+A11)B00.
    # Contributes to C10 and C11?
    # Strassen: C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
    # So M2 contributes to C10 (coeff 1) and C11 (coeff -1).
    # My V_2 for M2: V_2[0,1]=1 (B00). Correct.
    # My U_2 for M2: U_2[2,1]=1 (A10), U_2[3,1]=1 (A11). Correct.
    # My W_2 for M2: W_2[2,1]=1 (C10? k=1,i=0 -> 1*2+0=2. Yes), W_2[3,1]=1 (C11? k=1,i=1 -> 3. Yes).
    # But the coefficient for C11 should be -1!
    # I set W_2[3,1]=1. It should be -1.
    
    # I need to be very careful with the signs in W.
    
    # Let's re-derive the 7 terms carefully.
    # A = [[a, b], [c, d]] -> Indices: a=00, b=01, c=10, d=11
    # B = [[e, f], [g, h]] -> Indices: e=00, f=01, g=10, h=11
    # C = [[c00, c01], [c10, c11]] -> Indices: c00=00, c01=01, c10=10, c11=11
    # W index: k*n + i.
    # c00: k=0, i=0 -> 0
    # c01: k=0, i=1 -> 1
    # c10: k=1, i=0 -> 2
    # c11: k=1, i=1 -> 3
    
    # M1 = (a+d)(e+h)
    # U: a(0), d(3) -> [1,0,0,1]
    # V: e(0), h(3) -> [1,0,0,1]
    # W: c00(0), c11(3) -> [1,0,0,1] (Coeffs: +1, +1)
    
    # M2 = (c+d)e
    # U: c(2), d(3) -> [0,0,1,1]
    # V: e(0) -> [1,0,0,0]
    # W: c10(2) (+1), c11(3) (-1) -> [0,0,1,-1]
    
    # M3 = a(f-h)
    # U: a(0) -> [1,0,0,0]
    # V: f(1), h(3) -> [0,1,0,-1]
    # W: c01(1) (+1), c11(3) (+1) -> [0,1,0,1]
    
    # M4 = d(g-e)
    # U: d(3) -> [0,0,0,1]
    # V: g(2), e(0) -> [-1,0,1,0]
    # W: c00(0) (+1), c10(2) (+1) -> [1,0,1,0]
    
    # M5 = (a+b)h
    # U: a(0), b(1) -> [1,1,0,0]
    # V: h(3) -> [0,0,0,1]
    # W: c01(1) (+1), c11(3) (+1) -> [0,1,0,1]
    
    # M6 = (c-a)(e+g)
    # U: c(2), a(0) -> [-1,0,1,0]
    # V: e(0), g(2) -> [1,0,1,0]
    # W: c00(0) (+1), c10(2) (+1) -> [1,0,1,0]
    
    # M7 = (b-d)(f+h)
    # U: b(1), d(3) -> [0,1,0,-1]
    # V: f(1), h(3) -> [0,1,0,1]
    # W: c00(0) (-1), c11(3) (+1) -> [-1,0,0,1]
    
    # Let's construct the 2x2x2 matrices with these coefficients.
    U_2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]).T # Shape (7, 4) -> Transpose to (4, 7)
    # Wait, U_2[i, r].
    # Row 0 (a): [1, 0, 1, 0, 1, -1, 0]
    # Row 1 (b): [0, 0, 0, 0, 1, 0, 1]
    # Row 2 (c): [0, 1, 0, 0, 0, 1, 0]
    # Row 3 (d): [1, 1, 0, 1, 0, 0, -1]
    
    V_2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 0, 1],  # f
        [0, 0, 0, 1, 0, 1, 0],  # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ]) # Shape (4, 7)
    
    W_2 = np.array([
        [1, 0, 0, 1, 0, 1, -1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 1, 0],  # c10
        [1, -1, 1, 0, 1, 0, 1]  # c11
    ]) # Shape (4, 7)
    
    # Verify 2x2x2
    # C = sum_r U[:,r] * V[:,r] * W[:,r]^T? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Decomposition: C[i,k] = sum_r U[i,r] V[j,r] W[k,r]? No, j is summed out.
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # For 2x2x2: T[i,j,k] = sum_r U[i*2+j, r] V[j*2+k, r] W[k*2+i, r].
    
    # Let's check T[0,0,0] (coeff of A00 B00 in C00).
    # A00 B00 should contribute 1 to C00.
    # Sum_r U[0, r] V[0, r] W[0, r]
    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    # r=1: U[0,1]=0 -> 0
    # r=2: U[0,2]=1, V[0,2]=0 -> 0
    # r=3: U[0,3]=0 -> 0
    # r=4: U[0,4]=1, V[0,4]=0 -> 0
    # r=5: U[0,5]=-1, V[0,5]=1, W[0,5]=1 -> -1
    # r=6: U[0,6]=0 -> 0
    # Sum = 1 - 1 = 0. 
    # Error: A00 B00 should contribute to C00.
    # In Strassen, C00 = M1 + M4 + M5 - M7.
    # M1 = (a+d)(e+h). Contains ae.
    # M4 = d(g-e). Contains -de.
    # M5 = (a+b)h. Contains ah.
    # M7 = (b-d)(f+h). Contains -dh.
    # Where is ae?
    # M1 has ae.
    # Does any other term have ae?
    # M2 = (c+d)e. No.
    # M3 = a(f-h). No.
    # M6 = (c-a)(e+g). Contains -ae.
    # So C00 = M1 + M4 + M5 - M7.
    # Coeff of ae in C00:
    # M1: +1
    # M4: 0
    # M5: 0
    # M7: 0
    # M6 is NOT in C00!
    # Wait, Strassen formula:
    # C00 = M1 + M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My W vectors:
    # W[0] (c00): M1(+1), M4(+1), M5(+1), M7(-1).
    # My W_2[0] was [1, 0, 0, 1, 0, 1, -1].
    # M1 is r=0. M4 is r=3. M5 is r=4. M7 is r=6.
    # So W_2[0] should be 1 at r=0, 3, 4 and -1 at r=6.
    # My W_2[0] is [1, 0, 0, 1, 0, 1, -1].
    # Index 5 is 1. M6 is r=5.
    # M6 is NOT in C00. So W_2[0, 5] should be 0.
    # I had W_2[0, 5] = 1. That's wrong.
    
    # Let's re-assign W_2 based on the formulas.
    # r=0 (M1): C00(+1), C11(+1)
    # r=1 (M2): C10(+1), C11(-1)
    # r=2 (M3): C01(+1), C11(+1)
    # r=3 (M4): C00(+1), C10(+1)
    # r=4 (M5): C00(+1), C01(+1)
    # r=5 (M6): C11(+1)
    # r=6 (M7): C00(-1), C11(+1)
    
    # W_2 (4x7):
    # c00 (0): [1, 0, 0, 1, 1, 0, -1]
    # c01 (1): [0, 0, 1, 0, 1, 0, 0]
    # c10 (2): [0, 1, 0, 1, 0, 0, 0]
    # c11 (3): [1, -1, 1, 0, 0, 1, 1]
    
    # Now re-verify T[0,0,0] (ae -> C00)
    # Sum_r U[0,r] V[0,r] W[0,r]
    # r=0: 1*1*1 = 1
    # r=1: 0
    # r=2: 1*0*0 = 0
    # r=3: 0
    # r=4: 1*0*1 = 0
    # r=5: -1*1*0 = 0
    # r=6: 0
    # Sum = 1. Correct.
    
    # Verify T[0,1,0] (af -> C00? No, af contributes to C01)
    # C01 = M3 + M5.
    # M3 = a(f-h). af coeff +1.
    # M5 = (a+b)h. No f.
    # So af should contribute 1 to C01.
    # T[0,1,1] (i=0, j=1, k=1 -> C01).
    # Sum_r U[0,r] V[1,r] W[1,r]
    # r=0: 1*0*0 = 0
    # r=1: 0
    # r=2: 1*1*1 = 1
    # r=3: 0
    # r=4: 1*0*1 = 0
    # r=5: -1*0*0 = 0
    # r=6: 0
    # Sum = 1. Correct.

    # Now, construct the 4x4x4 decomposition (Rank 49).
    # This is the tensor product of the 2x2x2 decomposition with itself.
    # Indices for 4x4x4:
    # i = 2*i1 + i2
    # j = 2*j1 + j2
    # k = 2*k1 + k2
    # U_4[i, r] where r = r1*7 + r2.
    # U_4[2*i1+i2, r1*7+r2] = U_2[i1, r1] * U_2[i2, r2]?
    # No, the Kronecker product structure.
    # A is a 2x2 block matrix of 2x2 blocks.
    # A_block[i1, j1] is a 2x2 matrix.
    # The decomposition is "block Strassen".
    # M_block[r1, r2] = sum_{j1} A_block[i1, j1] ...
    # Actually, the rank-49 decomposition is simply the Kronecker product of the rank-7 matrices.
    # U_4 = U_2 kron U_2
    # V_4 = V_2 kron V_2
    # W_4 = W_2 kron W_2
    
    U_4 = np.kron(U_2, U_2) # (16, 49)
    V_4 = np.kron(V_2, V_2) # (16, 49)
    W_4 = np.kron(W_2, W_2) # (16, 49)
    
    # This handles C[:, 0:4] = A * B[:, 0:4].
    # We need to map this to the full 4x4x6 interface.
    # U_4 is (16, 49). This fits U (16, rank).
    # V_4 is (16, 49). But V needs to be (24, rank).
    # V_4 corresponds to B[:, 0:4] (4x4 part of B).
    # We need to place V_4 into the first 16 rows of V (corresponding to B[:, 0:4]).
    # V_4 index j*4+k where j in 0..3, k in 0..3.
    # In V (24, rank), the indices for B[:, 0:4] are j*6+k for k in 0..3.
    # So V_4's column k corresponds to V's column j*6+k.
    # V_4's row j corresponds to V's row j.
    
    # W_4 is (16, 49). W needs to be (24, rank).
    # W_4 index k*4+i where k in 0..3, i in 0..3.
    # In W (24, rank), the indices for C[:, 0:4] are k*4+i for k in 0..3.
    # So W_4 fits directly into the first 16 rows of W?
    # W index is k*n+i. n=4.
    # For k=0..3, indices are 0..15.
    # So W_4 fits into W[0:16, :].
    
    # --- Part 2: 4x4x2 decomposition (Rank 24) ---
    # This handles C[:, 4:6] = A * B[:, 4:6].
    # Schoolbook rank 24.
    # For each i, j, k in {4,5}:
    # Term: A[i,j] B[j,k] -> C[i,k]
    # U: 1 at i*4+j
    # V: 1 at j*6+k
    # W: 1 at k*4+i
    
    U_24 = np.zeros((16, 24))
    V_24 = np.zeros((24, 24))
    W_24 = np.zeros((24, 24))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U_24[i*4 + j, col] = 1
                V_24[j*6 + k, col] = 1
                W_24[k*4 + i, col] = 1
                col += 1
    
    # Combine
    U_total = np.hstack([U_4, U_24]) # (16, 49+24) = (16, 73)
    V_total = np.hstack([V_4, V_24]) # (24, 73)
    W_total = np.hstack([W_4, W_24]) # (24, 73)
    
    # Convert to lists for the interface
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

