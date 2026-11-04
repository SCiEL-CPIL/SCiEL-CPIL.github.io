"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition of rank 98 for 4x4x8 matrix multiplication.
    # We view the 4x4x8 problem as two independent 4x4x4 problems:
    #   C_left  = A * B_left   (4x4 * 4x4 -> 4x4)
    #   C_right = A * B_right  (4x4 * 4x4 -> 4x4)
    # Each 4x4x4 uses the rank-49 Strassen decomposition (7^2 = 49).
    # Total rank = 49 + 49 = 98.
    #
    # The tensor T[i, j, k] for 4x4x8 has indices:
    #   i in [0,4)  (output row)
    #   j in [0,4)  (input column, shared)
    #   k in [0,8)  (input row of second factor)
    #
    # For the left half (k in [0,4)): C_left[i, k] = sum_j A[i,j] * B_left[j,k]
    #   where B_left[j,k] = B[j, k] for k in 0..3
    # For the right half (k in [4,8)): C_right[i, k-4] = sum_j A[i,j] * B_right[j,k-4]
    #   where B_right[j, k-4] = B[j, k] for k in 4..7
    #
    # In the output factor W[k*n + i, r], the index is k*n + i where k is the output
    # column (0..7) and i is the output row (0..3).
    #
    # Strassen 4x4x4 rank-49:
    #   Split A, B into 2x2 blocks of 2x2 matrices:
    #     A = [[A00, A01], [A10, A11]]
    #     B = [[B00, B01], [B10, B11]]
    #   The 4x4 product C = AB is computed via Strassen on the 2x2 block structure,
    #   where each "multiplication" of 2x2 blocks is itself done via Strassen (rank 7).
    #
    # This gives 7 * 7 = 49 rank-1 terms.
    #
    # We enumerate the Strassen decomposition explicitly.
    #
    # For a 2x2x2 Strassen decomposition (rank 7), the factor matrices are:
    #   M1 = (A00 + A11)(B00 + B11)
    #   M2 = (A10 + A11)B00
    #   M3 = A00(B01 - B11)
    #   M4 = A11(B10 - B00)
    #   M5 = (A00 + A01)B11
    #   M6 = (A10 - A00)(B00 + B01)
    #   M7 = (A01 - A11)(B10 + B11)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # For 4x4x4, we apply this at the block level. Let the 2x2 blocks be indexed
    # by (a,b) where a,b in {0,1}. Each block is 2x2.
    #
    # The outer Strassen operates on the 2x2 block matrix, treating each 2x2 block
    # as a "scalar". The inner Strassen operates within each 2x2 block multiplication.
    #
    # There are 7 outer terms, each of which is a product of two 2x2 "matrices"
    # (which are sums/differences of 2x2 blocks). Each such product requires 7
    # scalar multiplications (inner Strassen).
    #
    # Total: 7 * 7 = 49 terms.
    #
    # We'll build the rank-49 factors for one 4x4x4 multiplication, then duplicate
    # for the two halves.
    #
    # Let's define the Strassen 2x2x2 decomposition factors first.
    # For 2x2x2:
    #   U has shape (2*2, 7) = (4, 7)
    #   V has shape (2*2, 7) = (4, 7)
    #   W has shape (2*2, 7) = (4, 7)
    #
    # Indexing: for 2x2, i in {0,1}, j in {0,1}, k in {0,1}
    #   U[i*2 + j, c]
    #   V[j*2 + k, c]
    #   W[k*2 + i, c]
    #
    # The 7 Strassen terms (c=0..6):
    #   c=0: M1 = (A00+A11)(B00+B11)
    #   c=1: M2 = (A10+A11)B00
    #   c=2: M3 = A00(B01-B11)
    #   c=3: M4 = A11(B10-B00)
    #   c=4: M5 = (A00+A01)B11
    #   c=5: M6 = (A10-A00)(B00+B01)
    #   c=6: M7 = (A01-A11)(B10+B11)
    #
    # For each term c, we need U[i*2+j, c], V[j*2+k, c], W[k*2+i, c] such that
    #   sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = delta of the appropriate block.
    #
    # Actually, the tensor decomposition is: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # where for 2x2x2, the tensor T[i,j,k] = 1 if the output C[i,k] = sum_j A[i,j]B[j,k].
    # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor? No!)
    #
    # Wait. The matrix multiplication tensor for 2x2x2 has T[i,j,k] = 1 if and only if
    # the term A[i,j]*B[j,k] contributes to C[i,k]. Since C[i,k] = sum_j A[i,j] B[j,k],
    # every (i,j,k) triple contributes, so T[i,j,k] = 1 for all i,j,k in {0,1}.
    #
    # No, that's the schoolbook tensor. The tensor is T[i,j,k] = 1 for all i,j,k.
    # The rank of this tensor is 7 (Strassen).
    #
    # So for 2x2x2, T[i,j,k] = 1 for all i,j,k, and we need a rank-7 decomposition.
    #
    # Let me just hardcode the known Strassen rank-7 factors for 2x2x2.
    #
    # From standard references, the Strassen 2x2x2 factors (U, V, W) of shape (4,7) each:
    #
    # U (index i*2+j, term c):
    #   i=0,j=0 (idx 0): [1, 0, 1, 0, 1, -1, 0]  -- wait, let me derive carefully.
    #
    # Let me use a cleaner derivation. For the 2x2x2 Strassen decomposition:
    #
    # The 7 products are:
    #   P0 = (A00 + A11)(B00 + B11)
    #   P1 = (A10 + A11)B00
    #   P2 = A00(B01 - B11)
    #   P3 = A11(B10 - B00)
    #   P4 = (A00 + A01)B11
    #   P5 = (A10 - A00)(B00 + B01)
    #   P6 = (A01 - A11)(B10 + B11)
    #
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    #
    # Now, for the tensor decomposition T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]:
    #
    # Each P_c is a product of a linear combination of A entries and a linear combination
    # of B entries. The contribution to C[i,k] is W[k,i,c] * P_c.
    #
    # Let's figure out U[i,j,c], V[j,k,c], W[k,i,c] for each c.
    #
    # c=0 (P0 = (A00+A11)(B00+B11)):
    #   A part: A00 + A11 -> U[0,0,0]=1, U[1,1,0]=1, others 0
    #   B part: B00 + B11 -> V[0,0,0]=1, V[1,1,0]=1, others 0
    #   C part: C00 += 1, C11 += 1 -> W[0,0,0]=1, W[1,1,0]=1, others 0
    #
    # c=1 (P1 = (A10+A11)B00):
    #   A part: A10 + A11 -> U[1,0,1]=1, U[1,1,1]=1
    #   B part: B00 -> V[0,0,1]=1
    #   C part: C10 += 1, C11 += -1 -> W[0,1,1]=1, W[1,1,1]=-1
    #
    # c=2 (P2 = A00(B01-B11)):
    #   A part: A00 -> U[0,0,2]=1
    #   B part: B01 - B11 -> V[0,1,2]=1, V[1,1,2]=-1
    #   C part: C01 += 1, C11 += 1 -> W[1,0,2]=1, W[1,1,2]=1
    #
    # c=3 (P3 = A11(B10-B00)):
    #   A part: A11 -> U[1,1,3]=1
    #   B part: B10 - B00 -> V[1,0,3]=1, V[0,0,3]=-1
    #   C part: C00 += 1, C10 += 1 -> W[0,0,3]=1, W[0,1,3]=1
    #
    # c=4 (P4 = (A00+A01)B11):
    #   A part: A00 + A01 -> U[0,0,4]=1, U[0,1,4]=1
    #   B part: B11 -> V[1,1,4]=1
    #   C part: C00 += -1, C01 += 1 -> W[0,0,4]=-1, W[1,0,4]=1
    #
    # c=5 (P5 = (A10-A00)(B00+B01)):
    #   A part: A10 - A00 -> U[1,0,5]=1, U[0,0,5]=-1
    #   B part: B00 + B01 -> V[0,0,5]=1, V[0,1,5]=1
    #   C part: C11 += 1 -> W[1,1,5]=1
    #
    # c=6 (P6 = (A01-A11)(B10+B11)):
    #   A part: A01 - A11 -> U[0,1,6]=1, U[1,1,6]=-1
    #   B part: B10 + B11 -> V[1,0,6]=1, V[1,1,6]=1
    #   C part: C00 += 1 -> W[0,0,6]=1
    #
    # Let me verify c=0:
    #   U: U[0,0,0]=1, U[1,1,0]=1
    #   V: V[0,0,0]=1, V[1,1,0]=1
    #   W: W[0,0,0]=1, W[1,1,0]=1
    #   Contribution to T[i,j,k]:
    #     i=0,j=0,k=0: U[0,0,0]*V[0,0,0]*W[0,0,0] = 1*1*1 = 1 ✓
    #     i=0,j=1,k=1: U[0,1,0]*V[1,1,0]*W[1,0,0] = 0*1*0 = 0
    #     i=1,j=0,k=0: U[1,0,0]*V[0,0,0]*W[0,1,0] = 0*1*0 = 0
    #     i=1,j=1,k=1: U[1,1,0]*V[1,1,0]*W[1,1,0] = 1*1*1 = 1 ✓
    #   But we need T[0,0,0]=1, T[0,1,0]=1, T[0,0,1]=1, T[0,1,1]=1, etc.
    #   This only gives T[0,0,0]=1 and T[1,1,1]=1. That's not right.
    #
    # The issue is that the Strassen decomposition doesn't directly give a tensor
    # decomposition in the naive way. Let me think again.
    #
    # Actually, for the matrix multiplication tensor, T[i,j,k] = 1 for all i,j,k.
    # The schoolbook decomposition has rank 8 (for 2x2x2), with one term per (i,j,k).
    #
    # Strassen's rank-7 decomposition is more subtle. Let me look up the explicit
    # factors.
    #
    # Actually, I think the cleanest approach is to just use the schoolbook for the
    # inner 2x2 and do the outer Strassen. But that would give 7 * 8 = 56, not 49.
    #
    # Alternatively, I can just implement the full Strassen 4x4x4 directly.
    #
    # Let me take a different approach: implement the Strassen 4x4x4 by explicitly
    # enumerating all 49 terms.
    #
    # For 4x4x4, split into 2x2 blocks of 2x2 matrices. The outer Strassen has 7
    # "block multiplications". Each block multiplication is a 2x2x2 Strassen (7 scalar
    # multiplications). So 7*7=49.
    #
    # Let's define:
    #   A = [[A00, A01], [A10, A11]]  where each Aij is 2x2
    #   B = [[B00, B01], [B10, B11]]  where each Bij is 2x2
    #
    # Outer Strassen:
    #   S0 = (A00 + A11)(B00 + B11)
    #   S1 = (A10 + A11)B00
    #   S2 = A00(B01 - B11)
    #   S3 = A11(B10 - B00)
    #   S4 = (A00 + A01)B11
    #   S5 = (A10 - A00)(B00 + B01)
    #   S6 = (A01 - A11)(B10 + B11)
    #
    # C00 = S0 + S3 - S4 + S6
    # C01 = S2 + S4
    # C10 = S1 + S3
    # C11 = S0 - S1 + S2 + S5
    #
    # Each S_c is a 2x2 matrix product, computed via Strassen (7 scalar mults).
    #
    # For S_c = X * Y where X and Y are 2x2 matrices (linear combinations of 2x2 blocks),
    # we need to express X and Y in terms of the original A and B entries.
    #
    # This is getting complex. Let me just hardcode the full 49-term decomposition
    # by computing it programmatically.
    #
    # Actually, the simplest correct approach: use the schoolbook for the whole thing
    # but with a smaller rank by exploiting that p=8 is even and using a block structure.
    #
    # Wait, I realize I should just implement the Strassen 4x4x4 correctly. Let me
    # compute the factors programmatically.
    
    # Define Strassen 2x2x2 factors (rank 7)
    # U2, V2, W2 of shape (4, 7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # c=0: P0 = (A00+A11)(B00+B11)
    # A: A00 (i=0,j=0) + A11 (i=1,j=1)
    # B: B00 (j=0,k=0) + B11 (j=1,k=1)
    # C: C00 (i=0,k=0) + C11 (i=1,k=1)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # c=1: P1 = (A10+A11)B00
    # A: A10 (i=1,j=0) + A11 (i=1,j=1)
    # B: B00 (j=0,k=0)
    # C: C10 (i=1,k=0) + (-1)*C11 (i=1,k=1)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # c=2: P2 = A00(B01-B11)
    # A: A00 (i=0,j=0)
    # B: B01 (j=0,k=1) - B11 (j=1,k=1)
    # C: C01 (i=0,k=1) + C11 (i=1,k=1)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # c=3: P3 = A11(B10-B00)
    # A: A11 (i=1,j=1)
    # B: B10 (j=1,k=0) - B00 (j=0,k=0)
    # C: C00 (i=0,k=0) + C10 (i=1,k=0)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # c=4: P4 = (A00+A01)B11
    # A: A00 (i=0,j=0) + A01 (i=0,j=1)
    # B: B11 (j=1,k=1)
    # C: -C00 (i=0,k=0) + C01 (i=0,k=1)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # c=5: P5 = (A10-A00)(B00+B01)
    # A: A10 (i=1,j=0) - A00 (i=0,j=0)
    # B: B00 (j=0,k=0) + B01 (j=0,k=1)
    # C: C11 (i=1,k=1)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # c=6: P6 = (A01-A11)(B10+B11)
    # A: A01 (i=0,j=1) - A11 (i=1,j=1)
    # B: B10 (j=1,k=0) + B11 (j=1,k=1)
    # C: C00 (i=0,k=0)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Now build 4x4x4 Strassen (rank 49) from this.
    # Split A, B into 2x2 blocks of 2x2 matrices.
    # A block (bi, bj) has entries A[2*bi+di, 2*bj+dj] for di,dj in {0,1}
    # B block (bj, bk) has entries B[2*bj+dj, 2*bk+dk]
    # C block (bi, bk) has entries C[2*bi+di, 2*bk+dk]
    
    n4 = 4
    rank4 = 49
    U4 = [[0]*rank4 for _ in range(n4*n4)]
    V4 = [[0]*rank4 for _ in range(n4*n4)]
    W4 = [[0]*rank4 for _ in range(n4*n4)]
    
    # For each outer Strassen term c_outer (0..6), and inner Strassen term c_inner (0..6),
    # we get one rank-1 term in the 4x4x4 decomposition.
    #
    # The outer term c_outer corresponds to a product S_c = X * Y where:
    #   X is a linear combination of A blocks
    #   Y is a linear combination of B blocks
    #   And S_c contributes to C blocks via coefficients.
    #
    # Let's define the outer Strassen coefficients:
    # For each c_outer, we have:
    #   X_c = sum_{bi,bj} x_c[bi,bj] * A_block[bi,bj]
    #   Y_c = sum_{bj,bk} y_c[bj,bk] * B_block[bj,bk]
    #   C_contribution: sum_{bi,bk} w_c[bi,bk] * C_block[bi,bk]
    
    # Outer Strassen:
    # S0 = (A00+A11)(B00+B11)
    # S1 = (A10+A11)B00
    # S2 = A00(B01-B11)
    # S3 = A11(B10-B00)
    # S4 = (A00+A01)B11
    # S5 = (A10-A00)(B00+B01)
    # S6 = (A01-A11)(B10+B11)
    
    # X coefficients (bi, bj):
    x_coeffs = [
        {(0,0):1, (1,1):1},  # S0
        {(1,0):1, (1,1):1},  # S1
        {(0,0):1},            # S2
        {(1,1):1},            # S3
        {(0,0):1, (0,1):1},  # S4
        {(1,0):1, (0,0):-1}, # S5
        {(0,1):1, (1,1):-1}, # S6
    ]
    
    # Y coefficients (bj, bk):
    y_coeffs = [
        {(0,0):1, (1,1):1},  # S0
        {(0,0):1},            # S1
        {(0,1):1, (1,1):-1}, # S2
        {(1,0):1, (0,0):-1}, # S3
        {(1,1):1},            # S4
        {(0,0):1, (0,1):1},  # S5
        {(1,0):1, (1,1):1},  # S6
    ]
    
    # W coefficients (bi, bk):
    # C00 = S0 + S3 - S4 + S6
    # C01 = S2 + S4
    # C10 = S1 + S3
    # C11 = S0 - S1 + S2 + S5
    w_coeffs = [
        {(0,0):1, (1,1):1},  # S0
        {(1,0):1, (1,1):-1}, # S1
        {(0,1):1, (1,1):1},  # S2
        {(0,0):1, (1,0):1},  # S3
        {(0,0):-1, (0,1):1}, # S4
        {(1,1):1},            # S5
        {(0,0):1},            # S6
    ]
    
    # Now, for each outer term c_o and inner term c_i, we construct a rank-1 term.
    # The inner Strassen for S_c = X*Y uses the 2x2x2 Strassen decomposition.
    # X and Y are 2x2 matrices. The inner Strassen computes X*Y using 7 scalar mults.
    #
    # For the inner Strassen, the factors U2, V2, W2 operate on the 2x2 entries of X and Y.
    # X[di, dj] = sum_{bi,bj} x_c[bi,bj] * A[2*bi+di, 2*bj+dj]
    # Y[dj, dk] = sum_{bj,bk} y_c[bj,bk] * B[2*bj+dj, 2*bk+dk]
    #
    # The inner Strassen term c_i gives:
    #   X_part = sum_{di,dj} U2[di*2+dj, c_i] * X[di, dj]
    #   Y_part = sum_{dj,dk} V2[dj*2+dk, c_i] * Y[dj, dk]
    #   C_part = sum_{di,dk} W2[dk*2+di, c_i] * C_block[di, dk]
    #
    # So:
    #   U4[i, j, c_o*7+c_i] = sum_{di,dj} U2[di*2+dj, c_i] * (coefficient of A[2*bi+di, 2*bj+dj] in X)
    #
    # The coefficient of A[a, b] in X is: sum over (bi,bj) such that 2*bi=di, 2*bj=dj... 
    # Actually, X[di, dj] = sum_{bi,bj} x_c[bi,bj] * A[2*bi+di, 2*bj+dj]
    # So A[a, b] appears in X[di, dj] where di = a - 2*bi, dj = b - 2*bj, and x_c[bi,bj] is the coefficient.
    # For given a, b: bi = a//2, bj = b//2, di = a%2, dj = b%2.
    # So X[di, dj] contains A[a,b] with coefficient x_c[a//2, b//2] where di=a%2, dj=b%2.
    
    # Let's build U4, V4, W4 for 4x4x4.
    for c_o in range(7):
        for c_i in range(7):
            c = c_o * 7 + c_i
            # U4[i, j, c]: i is output row, j is input column (from A)
            # A[i, j] where i = 2*bi+di, j = 2*bj+dj
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    di = i % 2
                    dj = j % 2
                    # X[di, dj] contains A[i,j] with coefficient x_coeffs[c_o].get((bi,bj), 0)
                    x_coeff = x_coeffs[c_o].get((bi, bj), 0)
                    if x_coeff != 0:
                        # U2[di*2+dj, c_i] is the coefficient of X[di,dj] in the inner Strassen term
                        u2_val = U2[di*2+dj][c_i]
                        U4[i*4+j][c] = x_coeff * u2_val
            
            # V4[j, k, c]: j is input column (from B's row index), k is input row
            # B[j, k] where j = 2*bj+dj, k = 2*bk+dk
            for j in range(4):
                for k in range(4):
                    bj = j // 2
                    bk = k // 2
                    dj = j % 2
                    dk = k % 2
                    y_coeff = y_coeffs[c_o].get((bj, bk), 0)
                    if y_coeff != 0:
                        v2_val = V2[dj*2+dk][c_i]
                        V4[j*4+k][c] = y_coeff * v2_val
            
            # W4[k, i, c]: k is output column, i is output row
            # C[i, k] where i = 2*bi+di, k = 2*bk+dk
            for k in range(4):
                for i in range(4):
                    bi = i // 2
                    bk = k // 2
                    di = i % 2
                    dk = k % 2
                    w_coeff = w_coeffs[c_o].get((bi, bk), 0)
                    if w_coeff != 0:
                        w2_val = W2[dk*2+di][c_i]
                        W4[k*4+i][c] = w_coeff * w2_val
    
    # Now build the full 4x4x8 decomposition (rank 98).
    # Left half (k=0..3): use U4, V4, W4 as-is.
    # Right half (k=4..7): same but with B's columns shifted.
    #
    # For the right half, B_right[j, k-4] = B[j, k]. So the B factor is the same,
    # but the output W index is shifted: W[k*4+i] for k in 4..7.
    #
    # Actually, for the right half, we're computing C_right[i, k-4] = sum_j A[i,j] * B[j, k].
    # This is the same tensor structure, just with different B entries.
    # So we can reuse the same U4, V4, W4 but with V4 using B[j, k] for k in 4..7.
    #
    # In the global V factor, V has shape (m*p, rank) = (4*8, rank).
    # For the left half, V4 uses B[j, k] for k in 0..3, so V[j*8+k, c] = V4[j*4+k, c].
    # For the right half, V uses B[j, k] for k in 4..7, so V[j*8+k, c+49] = V4[j*4+(k-4), c].
    #
    # For W: W has shape (p*n, rank) = (8*4, rank).
    # For the left half, W4 uses C[i, k] for k in 0..3, so W[k*4+i, c] = W4[k*4+i, c].
    # For the right half, W uses C[i, k-4] for k in 4..7, so W[k*4+i, c+49] = W4[(k-4)*4+i, c].
    
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Fill left half (c = 0..48)
    for i in range(n):
        for j in range(m):
            for c in range(49):
                U[i*m+j][c] = U4[i*m+j][c]
    
    for j in range(m):
        for k in range(4):  # left half of B
            for c in range(49):
                V[j*p+k][c] = V4[j*4+k][c]
    
    for k in range(4):  # left half of output
        for i in range(n):
            for c in range(49):
                W[k*n+i][c] = W4[k*4+i][c]
    
    # Fill right half (c = 49..97)
    for i in range(n):
        for j in range(m):
            for c in range(49):
                U[i*m+j][49+c] = U4[i*m+j][c]
    
    for j in range(m):
        for k in range(4, 8):  # right half of B
            for c in range(49):
                V[j*p+k][49+c] = V4[j*4+(k-4)][c]
    
    for k in range(4, 8):  # right half of output
        for i in range(n):
            for c in range(49):
                W[k*n+i][49+c] = W4[(k-4)*4+i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
