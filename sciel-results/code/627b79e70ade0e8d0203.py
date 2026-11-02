"""Matrix multiplication tensor decomposition M_{3,5,7} with rank < 105."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 110  # safety margin

    # ------------------------------------------------------------------
    # Helper: build schoolbook U, V, W for an n x m x p block
    # ------------------------------------------------------------------
    def schoolbook_block(n, m, p):
        """Return (U, V, W) as lists-of-lists for the schoolbook decomposition."""
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # ------------------------------------------------------------------
    # Helper: build Strassen 2x2x2 decomposition
    # ------------------------------------------------------------------
    def strassen_2x2x2():
        """
        Return (U, V, W) for M_{2,2,2} using Strassen's 7-term formula.
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
        The tensor convention: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        Actually, the standard convention is:
          T = sum_r u_r ⊗ v_r ⊗ w_r
          where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
          and the product is C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        Wait, let me re-read the task contract:
        "output factor follows the official trace convention W[k*n+i,r]"
        So W has shape (p*n, rank) with W[k*n+i, r].
        The bilinear map: C[i,k] = sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r]?
        No, that's not right either. Let me think more carefully.
        
        The matrix multiplication tensor T_{n,m,p} ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}
        is defined by T_{i,j;k,l} = δ_{j,k} δ_{i,l}... no wait.
        
        Actually, the standard convention is:
        C = A × B, where C[i,k] = Σ_j A[i,j] B[j,k]
        
        The tensor T has components T[i,j,k] = δ_{j,k} (this doesn't seem right for 3 indices).
        
        Let me use the explicit convention from the task:
        U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        
        The decomposition means:
        For the matrix multiplication tensor, we want:
        δ_{j,k} = Σ_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        Wait, that can't be right because the left side is 1 when j=k and 0 otherwise,
        but the right side depends on i, j, k.
        
        Let me reconsider. The matrix multiplication tensor is:
        T_{i,j,k} where i indexes rows of A (and C), j indexes columns of A / rows of B,
        k indexes columns of B (and C).
        
        T_{i,j,k} = δ_{j,k} is not correct. The tensor represents the map
        (A, B) → C where C[i,k] = Σ_j A[i,j] B[j,k].
        
        In tensor form, if we flatten A to vector a ∈ R^{n*m}, B to vector b ∈ R^{m*p},
        and C to vector c ∈ R^{n*p}, then:
        c[i*p+k] = Σ_j a[i*m+j] * b[j*p+k]
        
        So the tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} has:
        T[a_idx, b_idx, c_idx] = 1 if a_idx = i*m+j, b_idx = j*p+k, c_idx = i*p+k
        for some i, j, k, and 0 otherwise.
        
        But the task says W has shape (p*n, rank) with W[k*n+i, r]. So the C index is
        k*n+i, not i*p+k. This means the output C is stored in column-major order with
        k (column) as the outer index and i (row) as the inner index. So C[k*n+i] = C[i,k].
        
        So the tensor is:
        T[a_idx, b_idx, c_idx] = 1 if a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
        
        A rank-1 decomposition term r contributes:
        u_r ⊗ v_r ⊗ w_r where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
        
        And the full tensor is: T = Σ_r u_r ⊗ v_r ⊗ w_r
        
        Which means: T[a_idx, b_idx, c_idx] = Σ_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
        
        For the schoolbook: each (i,j,k) triple gives a term with
        U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
        
        And indeed: T[i*m+j, j*p+k, k*n+i] = 1 (from that term) and 0 from others.
        For other indices, the sum is 0. This is correct.
        
        So for Strassen, I need to express the M_{2,2,2} tensor as a sum of 7 rank-1 tensors.
        
        For M_{2,2,2}: n=2, m=2, p=2
        A is 2x2, B is 2x2, C = A*B is 2x2.
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        
        Strassen's algorithm:
        P1 = (A00+A11)(B00+B11)
        P2 = (A10+A11)B00
        P3 = A00(B01-B11)
        P4 = A11(B10-B00)
        P5 = (A00+A01)B11
        P6 = (A10-A00)(B00+B01)
        P7 = (A01-A11)(B10+B11)
        
        C00 = P1 + P4 - P5 + P7
        C01 = P3 + P5
        C10 = P2 + P4
        C11 = P1 - P2 + P3 + P6
        
        Each P_t is a product of a linear combination of A entries and a linear combination
        of B entries. In tensor form, each P_t corresponds to a rank-1 tensor where:
        The U factor has the coefficients of the A combination, the V factor has the
        coefficients of the B combination, and the W factor has the coefficients for
        how P_t contributes to each C entry.
        
        Let's write this out. A is flattened as [A00, A01, A10, A11] (row-major, i*m+j).
        B is flattened as [B00, B01, B10, B11] (row-major, j*p+k).
        C is flattened as [C00, C01, C10, C11] but with the convention W[k*n+i], so
        C index: k*n+i. So C[0,0] is at index 0*2+0=0, C[0,1] at 1*2+0=2,
        C[1,0] at 0*2+1=1, C[1,1] at 1*2+1=3.
        
        Wait, that's confusing. Let me re-examine. W has shape (p*n, rank) with
        W[k*n+i, r]. So the index into W is k*n+i where k is the column of C and
        i is the row of C. So:
        - C[i=0, k=0]: W index = 0*2+0 = 0
        - C[i=0, k=1]: W index = 1*2+0 = 2
        - C[i=1, k=0]: W index = 0*2+1 = 1
        - C[i=1, k=1]: W index = 1*2+1 = 3
        
        So W[0] corresponds to C[0,0], W[1] to C[1,0], W[2] to C[0,1], W[3] to C[1,1].
        
        Now for each P_t:
        P1 = (A00+A11)(B00+B11)
        U for P1: [1, 0, 0, 1] (A00 and A11)
        V for P1: [1, 0, 0, 1] (B00 and B11)
        W for P1: contributes to C00 (+1), C11 (+1)
        So W: [1, 0, 0, 1] (indices 0 and 3)
        
        P2 = (A10+A11)B00
        U: [0, 0, 1, 1]
        V: [1, 0, 0, 0]
        W: contributes to C10 (+1), C11 (-1)
        W: [0, 1, 0, -1] (indices 1 and 3)
        
        P3 = A00(B01-B11)
        U: [1, 0, 0, 0]
        V: [0, 1, 0, -1]
        W: contributes to C01 (+1), C11 (+1)
        W: [0, 0, 1, 1] (indices 2 and 3)
        
        P4 = A11(B10-B00)
        U: [0, 0, 0, 1]
        V: [0, 0, 1, -1]... wait, B10 is at index 1*2+0=2, B00 at index 0*2+0=0.
        So V: [-1, 0, 1, 0] (B00 coefficient -1, B10 coefficient +1)
        W: contributes to C00 (+1), C10 (+1)
        W: [1, 1, 0, 0]
        
        P5 = (A00+A01)B11
        U: [1, 1, 0, 0]
        V: [0, 0, 0, 1]
        W: contributes to C00 (-1), C01 (+1)
        W: [-1, 0, 1, 0]
        
        P6 = (A10-A00)(B00+B01)
        U: [-1, 0, 1, 0]
        V: [1, 1, 0, 0]
        W: contributes to C11 (+1)
        W: [0, 0, 0, 1]
        
        P7 = (A01-A11)(B10+B11)
        U: [0, 1, 0, -1]
        V: [0, 0, 1, 1]
        W: contributes to C00 (+1)
        W: [1, 0, 0, 0]
        
        Let me verify C00: P1 + P4 - P5 + P7
        P1: A00*B00 + A00*B11 + A11*B00 + A11*B11
        P4: -A11*B00 + A11*B10
        -P5: -A00*B11 - A01*B11
        P7: A01*B10 + A01*B11 - A11*B10 - A11*B11
        Sum: A00*B00 + A00*B11 - A00*B11 + A11*B00 - A11*B00 + A11*B10 - A11*B10 - A11*B11 + A01*B10 + A01*B11 - A01*B11
        = A00*B00 + A01*B10 ✓
        
        C01: P3 + P5
        P3: A00*B01 - A00*B11
        P5: A00*B11 + A01*B11
        Sum: A00*B01 + A01*B11 ✓
        
        C10: P2 + P4
        P2: A10*B00 + A11*B00
        P4: -A11*B00 + A11*B10
        Sum: A10*B00 + A11*B10 ✓
        
        C11: P1 - P2 + P3 + P6
        P1: A00*B00 + A00*B11 + A11*B00 + A11*B11
        -P2: -A10*B00 - A11*B00
        P3: A00*B01 - A00*B11
        P6: -A00*B00 - A00*B01 + A10*B00 + A10*B01
        Sum: A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A00*B01 - A00*B11 - A00*B00 - A00*B01 + A10*B00 + A10*B01
        = A11*B11 + A10*B01 ✓
        
        Great! So the Strassen decomposition is:
        U = [[1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1], [1,1,0,0], [-1,0,1,0], [0,1,0,-1]]
        V = [[1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0], [0,0,0,1], [1,1,0,0], [0,0,1,1]]
        W = [[1,0,0,1], [0,1,0,-1], [0,0,1,1], [1,1,0,0], [-1,0,1,0], [0,0,0,1], [1,0,0,0]]
        
        Note: these are column vectors (each column is one term). So U has shape (4,7),
        V has shape (4,7), W has shape (4,7).
        """
        U = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=np.int64).T  # shape (4, 7)
        
        V = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ], dtype=np.int64).T  # shape (4, 7)
        
        W = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=np.int64).T  # shape (4, 7)
        
        return U, V, W

    # ------------------------------------------------------------------
    # Helper: build a 2x3x3 decomposition using Strassen on a 2x2x2 sub-block
    # M_{2,3,3}: A is 2x3, B is 3x3, C = A*B is 2x3.
    # Split B into [B_left (3x2), B_right (3x1)]
    # C_left = A * B_left: this is M_{2,3,2}
    # C_right = A * B_right: this is M_{2,3,1}
    #
    # M_{2,3,1}: C[i,0] = sum_j A[i,j] B[j,0]. This is just two dot products.
    # Rank = 2*3*1 = 6 (no reduction possible for 1-column output).
    #
    # M_{2,3,2}: A is 2x3, B is 3x2, C = A*B is 2x2.
    # We can write A = [A1 A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x2, B2 is 1x2.
    # C = A1*B1 + A2*B2
    # A1*B1 is M_{2,2,2}: rank 7 (Strassen)
    # A2*B2 is M_{2,1,2}: rank 4 (schoolbook, since one of the "middle" dims is 1)
    # Total: 7 + 4 = 11
    #
    # So M_{2,3,3} has rank at most 11 + 6 = 17.
    # But wait, we can also try:
    # M_{2,3,3}: split the 3 in the middle differently.
    # Actually, let me reconsider. The best known rank for M_{2,3,3} might be 13.
    # Let me check: Bini-Capovani et al. showed M_{2,3,3} has rank 13.
    #
    # For now, let's use the construction: M_{2,3,2} (rank 11) + M_{2,3,1} (rank 6) = 17.
    # That's better than schoolbook 18, but not optimal.
    #
    # Actually, I realize I should just focus on getting a valid decomposition below 105.
    # Let me use a simpler approach: block decomposition with Strassen where possible.
    #
    # Let me try a different split for M_{3,5,7}:
    # Split A (3x5) into [A_top (2x5); A_bottom (1x5)]
    # Split B (5x7) into [B_left (5x4) | B_right (5x3)]
    #
    # C_top_left = A_top * B_left: M_{2,5,4}
    # C_top_right = A_top * B_right: M_{2,5,3}
    # C_bot_left = A_bottom * B_left: M_{1,5,4} (rank = 20, no reduction)
    # C_bot_right = A_bottom * B_right: M_{1,5,3} (rank = 15, no reduction)
    #
    # For M_{2,5,4}:
    #   Split B_left into [BL1 (5x2), BL2 (5x2)]
    #   C_top_left = A_top*BL1 + A_top*BL2: two M_{2,5,2}
    #   M_{2,5,2}: split A into [A1 (2x3), A2 (2x2)], B into [B1; B2] where B1 is 3x2, B2 is 2x2
    #   C = A1*B1 + A2*B2
    #   A1*B1: M_{2,3,2}, rank 11 (as computed above)
    #   A2*B2: M_{2,2,2}, rank 7 (Strassen)
    #   Total M_{2,5,2}: 11 + 7 = 18
    #   Two of these: 36
    #
    # For M_{2,5,3}:
    #   Split B_right into [BR1 (5x2), BR2 (5x1)]
    #   A_top*BR1: M_{2,5,2}, rank 18
    #   A_top*BR2: M_{2,5,1}, rank 10 (no reduction)
    #   Total: 28
    #
    # Grand total: 36 + 28 + 20 + 15 = 99
    #
    # That's an improvement from 105 to 99! Let me implement this.
    
    # Let me define a helper that builds M_{2,5,2} with rank 18.
    
    def build_M_2_5_2():
        """
        Build M_{2,5,2} decomposition with rank 18.
        A is 2x5, B is 5x2, C = A*B is 2x2.
        Split A = [A1 (2x3) | A2 (2x2)]
        Split B = [B1 (3x2); B2 (2x2)]
        C = A1*B1 + A2*B2
        A1*B1: M_{2,3,2}
        A2*B2: M_{2,2,2}
        """
        # M_{2,3,2}: A is 2x3, B is 3x2, C is 2x2.
        # Split A = [A11 (2x2) | A12 (2x1)]
        # Split B = [B11 (2x2); B12 (1x2)]
        # C = A11*B11 + A12*B12
        # A11*B11: M_{2,2,2}, rank 7
        # A12*B12: M_{2,1,2}, rank 4
        # Total: 11
        
        # M_{2,2,2}: Strassen, rank 7
        U_s, V_s, W_s = strassen_2x2x2()  # each shape (4, 7)
        
        # M_{2,1,2}: A is 2x1, B is 1x2, C is 2x2.
        # C[i,k] = A[i,0] * B[0,k]
        # This is an outer product. Rank 4 (each output element needs one "multiplication").
        # Actually, the tensor T_{2,1,2} has rank 4: for each (i,k) pair, one term.
        U_212 = np.zeros((2*1, 4), dtype=np.int64)
        V_212 = np.zeros((1*2, 4), dtype=np.int64)
        W_212 = np.zeros((2*2, 4), dtype=np.int64)
        col = 0
        for i in range(2):
            for k in range(2):
                U_212[i*1+0, col] = 1
                V_212[0*2+k, col] = 1
                W_212[k*2+i, col] = 1
                col += 1
        
        # Combine M_{2,3,2} = M_{2,2,2} + M_{2,1,2}
        # The U, V, W matrices need to be extended to the full 2x3 and 3x2 sizes.
        # For M_{2,3,2}: A is 2x3 (indices 0..5), B is 3x2 (indices 0..5), C is 2x2 (indices 0..3)
        # The M_{2,2,2} part uses A columns 0,1 and B rows 0,1.
        # The M_{2,1,2} part uses A column 2 and B row 2.
        
        U_232 = np.zeros((2*3, 11), dtype=np.int64)
        V_232 = np.zeros((3*2, 11), dtype=np.int64)
        W_232 = np.zeros((2*2, 11), dtype=np.int64)
        
        # Place Strassen part (columns 0-6)
        U_232[0:2, 0:7] = U_s  # A rows 0,1; A columns 0,1 -> indices 0,1
        U_232[2:4, 0:7] = U_s[:, 2:4]  # Wait, this is wrong.
        
        # Let me be more careful. In M_{2,2,2}, A is 2x2 with indices:
        # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        # In M_{2,3,2}, A is 2x3 with indices:
        # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
        # The Strassen part uses A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,3,4
        # The M_{2,1,2} part uses A[0,2], A[1,2] -> indices 2,5
        
        # So for U_232, the Strassen part maps:
        # U_s[0] -> U_232[0] (A[0,0])
        # U_s[1] -> U_232[1] (A[0,1])
        # U_s[2] -> U_232[3] (A[1,0])
        # U_s[3] -> U_232[4] (A[1,1])
        
        U_232[0, 0:7] = U_s[0, :]
        U_232[1, 0:7] = U_s[1, :]
        U_232[3, 0:7] = U_s[2, :]
        U_232[4, 0:7] = U_s[3, :]
        
        # For V_232, in M_{2,2,2}, B is 2x2 with indices:
        # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
        # In M_{2,3,2}, B is 3x2 with indices:
        # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3, B[2,0]=4, B[2,1]=5
        # The Strassen part uses B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,2,3
        
        V_232[0:4, 0:7] = V_s
        
        # For W_232, C is 2x2 in both cases, so W maps directly.
        W_232[:, 0:7] = W_s
        
        # Place M_{2,1,2} part (columns 7-10)
        # U_212[0] -> U_232[2] (A[0,2])
        # U_212[1] -> U_232[5] (A[1,2])
        U_232[2, 7:11] = U_212[0, :]
        U_232[5, 7:11] = U_212[1, :]
        
        # V_212[0] -> V_232[4] (B[2,0])
        # V_212[1] -> V_232[5] (B[2,1])
        V_232[4, 7:11] = V_212[0, :]
        V_232[5, 7:11] = V_212[1, :]
        
        W_232[:, 7:11] = W_212
        
        # Now combine M_{2,5,2} = M_{2,3,2} + M_{2,2,2}
        # A is 2x5: indices 0..9
        # B is 5x2: indices 0..9
        # C is 2x2: indices 0..3
        # M_{2,3,2} uses A cols 0,1,2 and B rows 0,1,2
        # M_{2,2,2} uses A cols 3,4 and B rows 3,4
        
        U_252 = np.zeros((2*5, 18), dtype=np.int64)
        V_252 = np.zeros((5*2, 18), dtype=np.int64)
        W_252 = np.zeros((2*2, 18), dtype=np.int64)
        
        # Place M_{2,3,2} part (cols 0-10)
        U_252[0:6, 0:11] = U_232
        V_252[0:6, 0:11] = V_232
        W_252[:, 0:11] = W_232
        
        # Place M_{2,2,2} part (cols 11-17)
        # A cols 3,4 -> indices 6,7,8,9 in 2x5 matrix
        # B rows 3,4 -> indices 6,7,8,9 in 5x2 matrix
        U_252[6:10, 11:18] = U_s
        V_252[6:10, 11:18] = V_s
        W_252[:, 11:18] = W_s
        
        return U_252, V_252, W_252

    # Now build the full M_{3,5,7} decomposition.
    # Split A (3x5) into [A_top (2x5); A_bot (1x5)]
    # Split B (5x7) into [B_left (5x4) | B_right (5x3)]
    #
    # C_top_left = A_top * B_left: M_{2,5,4}
    #   Split B_left into [BL1 (5x2), BL2 (5x2)]
    #   A_top*BL1: M_{2,5,2}, rank 18
    #   A_top*BL2: M_{2,5,2}, rank 18
    #   Total: 36
    #
    # C_top_right = A_top * B_right: M_{2,5,3}
    #   Split B_right into [BR1 (5x2), BR2 (5x1)]
    #   A_top*BR1: M_{2,5,2}, rank 18
    #   A_top*BR2: M_{2,5,1}, rank 10
    #   Total: 28
    #
    # C_bot_left = A_bot * B_left: M_{1,5,4}, rank 20
    # C_bot_right = A_bot * B_right: M_{1,5,3}, rank 15
    #
    # Grand total: 36 + 28 + 20 + 15 = 99
    
    U_252, V_252, W_252 = build_M_2_5_2()
    
    # M_{2,5,1}: A is 2x5, B is 5x1, C is 2x1.
    # C[i,0] = sum_j A[i,j] B[j,0]
    # Rank = 10 (schoolbook)
    def build_M_2_5_1():
        r = 2 * 5 * 1
        U = np.zeros((10, r), dtype=np.int64)
        V = np.zeros((5, r), dtype=np.int64)
        W = np.zeros((2, r), dtype=np.int64)
        col = 0
        for i in range(2):
            for j in range(5):
                U[i*5+j, col] = 1
                V[j*1+0, col] = 1
                W[0*2+i, col] = 1
                col += 1
        return U, V, W
    
    U_251, V_251, W_251 = build_M_2_5_1()
    
    # M_{1,5,4}: A is 1x5, B is 5x4, C is 1x4.
    # Rank = 20
    def build_M_1_5_4():
        r = 1 * 5 * 4
        U = np.zeros((5, r), dtype=np.int64)
        V = np.zeros((20, r), dtype=np.int64)
        W = np.zeros((4, r), dtype=np.int64)
        col = 0
        for j in range(5):
            for k in range(4):
                U[j, col] = 1
                V[j*4+k, col] = 1
                W[k*1+0, col] = 1
                col += 1
        return U, V, W
    
    U_154, V_154, W_154 = build_M_1_5_4()
    
    # M_{1,5,3}: A is 1x5, B is 5x3, C is 1x3.
    # Rank = 15
    def build_M_1_5_3():
        r = 1 * 5 * 3
        U = np.zeros((5, r), dtype=np.int64)
        V = np.zeros((15, r), dtype=np.int64)
        W = np.zeros((3, r), dtype=np.int64)
        col = 0
        for j in range(5):
            for k in range(3):
                U[j, col] = 1
                V[j*3+k, col] = 1
                W[k*1+0, col] = 1
                col += 1
        return U, V, W
    
    U_153, V_153, W_153 = build_M_1_5_3()
    
    # Now assemble the full M_{3,5,7} decomposition.
    # A is 3x5: rows 0,1 (top), row 2 (bottom)
    # B is 5x7: cols 0-3 (left), cols 4-6 (right)
    # C is 3x7: rows 0,1 (top), row 2 (bottom); cols 0-3 (left), cols 4-6 (right)
    #
    # W convention: W[k*n+i, r] where k is column of C, i is row of C.
    # C has 7 columns, 3 rows. So W has 21 rows.
    # C[i,k] is at W index k*3+i.
    
    # C_top_left: i in {0,1}, k in {0,1,2,3}
    #   W indices: k*3+i for i in {0,1}, k in {0,1,2,3}
    #   = {0,1, 3,4, 6,7, 9,10}
    #
    # C_top_right: i in {0,1}, k in {4,5,6}
    #   W indices: k*3+i for i in {0,1}, k in {4,5,6}
    #   = {12,13, 15,16, 18,19}
    #
    # C_bot_left: i=2, k in {0,1,2,3}
    #   W indices: k*3+2 for k in {0,1,2,3}
    #   = {2, 5, 8, 11}
    #
    # C_bot_right: i=2, k in {4,5,6}
    #   W indices: k*3+2 for k in {4,5,6}
    #   = {14, 17, 20}
    
    # For M_{2,5,4} (C_top_left), the W matrix from two M_{2,5,2} blocks:
    # First M_{2,5,2} handles B cols 0,1 -> C cols 0,1
    # Second M_{2,5,2} handles B cols 2,3 -> C cols 2,3
    
    U_full = np.zeros((15, 99), dtype=np.int64)  # 3*5 = 15
    V_full = np.zeros((35, 99), dtype=np.int64)  # 5*7 = 35
    W_full = np.zeros((21, 99), dtype=np.int64)  # 7*3 = 21
    
    col_offset = 0
    
    # C_top_left, part 1: A_top (rows 0,1) * B_left_cols_0_1 (cols 0,1)
    # A_top rows 0,1 -> A indices 0-9 (first 10)
    # B_left cols 0,1 -> B rows all, cols 0,1 -> B indices 0,1, 4,5, 8,9, 12,13, 16,17
    # C_top_left cols 0,1 -> C rows 0,1, cols 0,1 -> W indices 0,1, 3,4
    
    U_full[0:10, col_offset:col_offset+18] = U_252
    # V_252 has 10 rows (5*2). These correspond to B[0,0], B[0,1], B[1,0], B[1,1], ..., B[4,0], B[4,1]
    # In the full B (5x7), these are at indices:
    # B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8, B[2,0]=14, B[2,1]=15, B[3,0]=21, B[3,1]=22, B[4,0]=28, B[4,1]=29
    v_indices_top_left_1 = [0, 1, 7, 8, 14, 15, 21, 22, 28, 29]
    V_full[v_indices_top_left_1, col_offset:col_offset+18] = V_252
    # W_252 has 4 rows (2*2). These correspond to C[0,0], C[0,1], C[1,0], C[1,1] in the local 2x2 C.
    # In the full C, these are at W indices: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    w_indices_top_left_1 = [0, 3, 1, 4]
    W_full[w_indices_top_left_1, col_offset:col_offset+18] = W_252
    col_offset += 18
    
    # C_top_left, part 2: A_top (rows 0,1) * B_left_cols_2_3 (cols 2,3)
    U_full[0:10, col_offset:col_offset+18] = U_252
    v_indices_top_left_2 = [2, 3, 9, 10, 16, 17, 23, 24, 30, 31]
    V_full[v_indices_top_left_2, col_offset:col_offset+18] = V_252
    # C cols 2,3 -> W indices: C[0,2]=6, C[0,3]=7, C[1,2]=8, C[1,3]=9
    w_indices_top_left_2 = [6, 7, 8, 9]
    W_full[w_indices_top_left_2, col_offset:col_offset+18] = W_252
    col_offset += 18
    
    # C_top_right, part 1: A_top (rows 0,1) * B_right_cols_4_5 (cols 4,5)
    U_full[0:10, col_offset:col_offset+18] = U_252
    v_indices_top_right_1 = [4, 5, 11, 12, 18, 19, 25, 26, 32, 33]
    V_full[v_indices_top_right_1, col_offset:col_offset+18] = V_252
    # C cols 4,5 -> W indices: C[0,4]=12, C[0,5]=13, C[1,4]=15, C[1,5]=16
    w_indices_top_right_1 = [12, 13, 15, 16]
    W_full[w_indices_top_right_1, col_offset:col_offset+18] = W_252
    col_offset += 18
    
    # C_top_right, part 2: A_top (rows 0,1) * B_right_col_6 (col 6)
    U_full[0:10, col_offset:col_offset+10] = U_251
    v_indices_top_right_2 = [6, 13, 20, 27, 34]
    V_full[v_indices_top_right_2, col_offset:col_offset+10] = V_251
    # C col 6 -> W indices: C[0,6]=18, C[1,6]=19
    w_indices_top_right_2 = [18, 19]
    W_full[w_indices_top_right_2, col_offset:col_offset+10] = W_251
    col_offset += 10
    
    # C_bot_left: A_bot (row 2) * B_left (cols 0-3)
    # A_bot row 2 -> A indices 10-14
    U_full[10:15, col_offset:col_offset+20] = U_154
    v_indices_bot_left = [0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17, 21, 22, 23, 24, 28, 29, 30, 31]
    V_full[v_indices_bot_left, col_offset:col_offset+20] = V_154
    # C row 2, cols 0-3 -> W indices: 2, 5, 8, 11
    w_indices_bot_left = [2, 5, 8, 11]
    W_full[w_indices_bot_left, col_offset:col_offset+20] = W_154
    col_offset += 20
    
    # C_bot_right: A_bot (row 2) * B_right (cols 4-6)
    U_full[10:15, col_offset:col_offset+15] = U_153
    v_indices_bot_right = [4, 5, 6, 11, 12, 13, 18, 19, 20, 25, 26, 27, 32, 33, 34]
    V_full[v_indices_bot_right, col_offset:col_offset+15] = V_153
    # C row 2, cols 4-6 -> W indices: 14, 17, 20
    w_indices_bot_right = [14, 17, 20]
    W_full[w_indices_bot_right, col_offset:col_offset+15] = W_153
    col_offset += 15
    
    assert col_offset == 99, f"Expected 99, got {col_offset}"
    
    # Convert to lists
    return {'U': U_full.tolist(), 'V': V_full.tolist(), 'W': W_full.tolist()}
# EVOLVE-BLOCK-END
