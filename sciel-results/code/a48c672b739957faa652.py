"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (21, rank)
    # Trace convention: W[k*n + i, r]
    # Tensor index: C[i,k] = sum_j A[i,j]*B[j,k]
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        """Add a rank-1 term with given indices and coefficient."""
        U.append([0.0] * len(U[0]) if U else [0.0] * 1)
        V.append([0.0] * len(V[0]) if V else [0.0] * 1)
        W.append([0.0] * len(W[0]) if W else [0.0] * 1)
        # Actually, let's build column by column
        pass
    
    # Let's build using lists of columns, then transpose
    # Each column c has: U_col (length 12), V_col (length 28), W_col (length 21)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[u_idx] = coeff
        V_col[v_idx] = coeff
        W_col[w_idx] = coeff
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    def add_schoolbook_1x2x7(i, j0, j1):
        """For fixed i, sum over j in {j0, j1} and k in 0..6: A[i,j]*B[j,k] -> C[i,k]"""
        for j in [j0, j1]:
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(u_idx, v_idx, w_idx, 1.0)
    
    def add_strassen_2x2x2(i0, i1, j0, j1, k0, k1):
        """
        For submatrices:
        A_sub = [[A[i0,j0], A[i0,j1]],
                 [A[i1,j0], A[i1,j1]]]
        B_sub = [[B[j0,k0], B[j0,k1]],
                 [B[j1,k0], B[j1,k1]]]
        
        C_sub = A_sub * B_sub (2x2)
        
        Strassen:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B10-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        
        where A00=A[i0,j0], A01=A[i0,j1], A10=A[i1,j0], A11=A[i1,j1]
        B00=B[j0,k0], B01=B[j0,k1], B10=B[j1,k0], B11=B[j1,k1]
        C00=C[i0,k0], C01=C[i0,k1], C10=C[i1,k0], C11=C[i1,k1]
        """
        A00 = (i0, j0)
        A01 = (i0, j1)
        A10 = (i1, j0)
        A11 = (i1, j1)
        B00 = (j0, k0)
        B01 = (j0, k1)
        B10 = (j1, k0)
        B11 = (j1, k1)
        
        C00 = (i0, k0)
        C01 = (i0, k1)
        C10 = (i1, k0)
        C11 = (i1, k1)
        
        # M1 = (A00+A11)(B00+B11)
        # This is a product of sums, so we need 4 terms in schoolbook, but Strassen treats it as one "multiplication"
        # In tensor decomposition, each "multiplication" M_k corresponds to one rank-1 term where
        # U has the coefficients for A, V has coefficients for B, W has coefficients for C.
        
        # Actually, in Strassen's algorithm, each M_k is itself a product of two linear combinations.
        # In the tensor decomposition sense, each M_k contributes to the output C with certain coefficients.
        # But M_k is not a single scalar multiplication in the original sense - it's a bilinear form.
        
        # Let me reconsider. In Strassen, we compute:
        # M1 = (A00+A11) * (B00+B11)
        # This is ONE multiplication of two numbers (which are themselves linear combinations of input entries).
        # So in the tensor decomposition, this corresponds to ONE rank-1 term where:
        #   U has 1 at positions A00 and A11
        #   V has 1 at positions B00 and B11
        #   W has the contributions to C entries
        
        # M1 contributes to:
        #   C00 with +1
        #   C11 with +1
        
        # Let's define each Strassen multiplication as one term
        
        # M1: U: A00=1, A11=1; V: B00=1, B11=1; W: C00=+1, C11=+1
        u_idx_list = [i0*m+j0, i1*m+j1]
        v_idx_list = [j0*p+k0, j1*p+k1]
        w_idx_list = [(k0)*n+i0, (k1)*n+i1]  # C00: k=k0, i=i0; C11: k=k1, i=i1
        U_col = [0.0]*(n*m); V_col = [0.0]*(m*p); W_col = [0.0]*(p*n)
        for idx in u_idx_list: U_col[idx] = 1.0
        for idx in v_idx_list: V_col[idx] = 1.0
        W_col[w_idx_list[0]] += 1.0  # C00
        W_col[w_idx_list[1]] += 1.0  # C11
        U_cols.append(U_col); V_cols.append(V_col); W_cols.append(W_col)
        
        # M2: (A10+A11) * B00
        # U: A10=1, A11=1; V: B00=1; W: C10=+1
        U_col = [0.0]*(n*m); V_col = [0.0]*(m*p); W_col = [0.0]*(p*n)
        U_col[i1*m+j0] = 1.0
        U_col[i1*m+j1] = 1.0
        V_col[j0*p+k0] = 1.0
        W_col[k0*n+i1] = 1.0  # C10: k=k0, i=i1
        U_cols.append(U_col); V_cols.append(V_col); W_cols.append(W_col)
        
        # M3: A00 * (B01-B11)
        # U: A00=1; V: B01=1, B11=-1; W: C01=+1
        U_col = [0.0]*(n*m); V_col = [0.0]*(m*p); W_col = [0.0]*(p*n)
        U_col[i0*m+j0] = 1.0
        V_col[j0*p+k1] = 1.0
        V_col[j1*p+k1] = -1.0
        W_col[k1*n+i0] = 1.0  # C01: k=k1, i=i0
        U_cols.append(U_col); V_cols.append(V_col); W_cols.append(W_col)
        
        # M4: A11 * (B10-B00)
        # U: A11=1; V: B10=1, B00=-1; W: C00=+1, C10=+1
        U_col = [0.0]*(n*m); V_col = [0.0]*(m*p); W_col = [0.0]*(p*n)
        U_col[i1*m+j1] = 1.0
        V_col[j1*p+k0] = 1.0
        V_col[j0*p+k0] = -1.0
        W_col[k0*n+i0] = 1.0  # C00
        W_col[k0*n+i1] = 1.0  # C10
        U_cols.append(U_col); V_cols.append(V_col); W_cols.append(W_col)
        
        # M5: (A00+A01) * B11
        # U: A00=1, A01=1; V: B11=1; W: C00=-1, C01=+1
        U_col = [0.0]*(n*m); V_col = [0.0]*(m*p); W_col = [0.0]*(p*n)
        U_col[i0*m+j0] = 1.0
        U_col[i0*m+j1] = 1.0
        V_col[j1*p+k1] = 1.0
        W_col[k0*n+i0] = -1.0  # C00
        W_col[k1*n+i0] = 1.0   # C01
        U_cols.append(U_col); V_cols.append(V_col); W_cols.append(W_col)
        
        # M6: (A10-A00) * (B00+B01)
        # U: A10=1, A00=-1; V: B00=1, B01=1; W: C11=+1
        U_col = [0.0]*(n*m); V_col = [0.0]*(m*p); W_col = [0.0]*(p*n)
        U_col[i1*m+j0] = 1.0
        U_col[i0*m+j0] = -1.0
        V_col[j0*p+k0] = 1.0
        V_col[j0*p+k1] = 1.0
        W_col[k1*n+i1] = 1.0  # C11
        U_cols.append(U_col); V_cols.append(V_col); W_cols.append(W_col)
        
        # M7: (A01-A11) * (B10+B11)
        # U: A01=1, A11=-1; V: B10=1, B11=1; W: C00=+1
        U_col = [0.0]*(n*m); V_col = [0.0]*(m*p); W_col = [0.0]*(p*n)
        U_col[i0*m+j1] = 1.0
        U_col[i1*m+j1] = -1.0
        V_col[j1*p+k0] = 1.0
        V_col[j1*p+k1] = 1.0
        W_col[k0*n+i0] = 1.0  # C00
        U_cols.append(U_col); V_cols.append(V_col); W_cols.append(W_col)
    
    # Strategy:
    # For each i in 0,1,2:
    #   For each pair of j's (or single j), and for each k:
    #     Use Strassen or schoolbook
    # 
    # Let's use a hybrid approach:
    # Partition the j dimension (size 4) into two groups of 2: {0,1} and {2,3}
    # For each i and each j-group and each k-group...
    # Actually, let's think about partitioning all three dimensions.
    #
    # n=3, m=4, p=7. 
    # Let's partition m into {0,1,2,3} as two groups of 2.
    # For each i and each j-group, we have a 1x2 subproblem for that i.
    # But Strassen requires 2x2x2.
    #
    # Alternative: For each pair of i's and each pair of j's and each pair of k's, use Strassen.
    # But n=3 is odd, p=7 is odd.
    #
    # Let's try: partition n={0,1,2} as {0,1} and {2}
    # Partition m={0,1,2,3} as {0,1} and {2,3}
    # Partition p={0,1,2,3,4,5,6} as {0,1},{2,3},{4,5},{6}
    #
    # For i in {0,1}, j in {0,1}, k in {0,1}: Strassen 2x2x2 -> 7 terms
    # For i in {0,1}, j in {2,3}, k in {0,1}: Strassen -> 7 terms
    # For i in {0,1}, j in {0,1}, k in {2,3}: Strassen -> 7 terms
    # For i in {0,1}, j in {2,3}, k in {2,3}: Strassen -> 7 terms
    # For i in {0,1}, j in {0,1}, k in {4,5}: Strassen -> 7 terms
    # For i in {0,1}, j in {2,3}, k in {4,5}: Strassen -> 7 terms
    # That's 6 * 7 = 42 terms for i in {0,1}
    #
    # For i=2 (single row):
    # For j in {0,1}, k in {0,1}: schoolbook 1x2x2 -> 4 terms
    # For j in {2,3}, k in {0,1}: schoolbook -> 4 terms
    # For j in {0,1}, k in {2,3}: schoolbook -> 4 terms
    # For j in {2,3}, k in {2,3}: schoolbook -> 4 terms
    # For j in {0,1}, k in {4,5}: schoolbook -> 4 terms
    # For j in {2,3}, k in {4,5}: schoolbook -> 4 terms
    # For j in {0,1}, k in {6}: schoolbook 1x2x1 -> 2 terms
    # For j in {2,3}, k in {6}: schoolbook -> 2 terms
    # That's 6*4 + 2*2 = 24 + 4 = 28 terms for i=2
    #
    # Total: 42 + 28 = 70 terms
    # But wait, we're missing k=6 for i in {0,1}
    # For i in {0,1}, j in {0,1}, k in {6}: schoolbook 2x2x1 -> 4 terms
    # For i in {0,1}, j in {2,3}, k in {6}: schoolbook -> 4 terms
    # That's 8 more terms
    # Total: 42 + 28 + 8 = 78 terms
    #
    # Let's implement this.
    
    # i groups: (0,1) and (2,)
    # j groups: (0,1) and (2,3)
    # k groups: (0,1), (2,3), (4,5), (6,)
    
    i_groups = [(0,1), (2,)]
    j_groups = [(0,1), (2,3)]
    k_groups = [(0,1), (2,3), (4,5), (6,)]
    
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                if len(ig) == 2 and len(jg) == 2 and len(kg) == 2:
                    add_strassen_2x2x2(ig[0], ig[1], jg[0], jg[1], kg[0], kg[1])
                elif len(ig) == 1 and len(jg) == 2 and len(kg) == 1:
                    # 1x2x1: schoolbook
                    for j in jg:
                        for k in kg:
                            u_idx = ig[0]*m + j
                            v_idx = j*p + k
                            w_idx = k*n + ig[0]
                            add_term(u_idx, v_idx, w_idx, 1.0)
                elif len(ig) == 2 and len(jg) == 2 and len(kg) == 1:
                    # 2x2x1: schoolbook
                    for i in ig:
                        for j in jg:
                            for k in kg:
                                u_idx = i*m + j
                                v_idx = j*p + k
                                w_idx = k*n + i
                                add_term(u_idx, v_idx, w_idx, 1.0)
                elif len(ig) == 1 and len(jg) == 2 and len(kg) == 2:
                    # 1x2x2: schoolbook
                    for j in jg:
                        for k in kg:
                            u_idx = ig[0]*m + j
                            v_idx = j*p + k
                            w_idx = k*n + ig[0]
                            add_term(u_idx, v_idx, w_idx, 1.0)
                else:
                    # Fallback: schoolbook
                    for i in ig:
                        for j in jg:
                            for k in kg:
                                u_idx = i*m + j
                                v_idx = j*p + k
                                w_idx = k*n + i
                                add_term(u_idx, v_idx, w_idx, 1.0)
    
    rank = len(U_cols)
    
    # Transpose columns to rows
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
