import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    rank = 51
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 51
    V = [[0] * rank for _ in range(m * p)] # 27 x 51
    W = [[0] * rank for _ in range(p * n)] # 18 x 51
    
    col = 0
    
    # Helper to add a product term
    # A_idx: index in A (i*m + j)
    # B_idx: index in B (j*p + k)
    # C_idx: index in C (k*n + i)
    # coeff_U, coeff_V, coeff_W
    def add_term(A_idx, B_idx, C_idx, cU, cV, cW):
        nonlocal col
        if time.time() > deadline:
            return
        U[A_idx][col] += cU
        V[B_idx][col] += cV
        W[C_idx][col] += cW
        col += 1

    # We process 3 blocks of B columns
    for b in range(3):
        # b is 0, 1, 2
        # B columns are 3*b, 3*b+1, 3*b+2
        
        # --- Part 1: Strassen for A_2 (cols 0,1) * B_2 (rows 0,1; cols 3b, 3b+1) ---
        # A_2 indices:
        # A00: i=0, j=0 -> 0
        # A01: i=0, j=1 -> 1
        # A10: i=1, j=0 -> 3
        # A11: i=1, j=1 -> 4
        
        # B_2 indices (rows 0,1; cols 3b, 3b+1):
        # B00: j=0, k=3b -> 0*9 + 3b
        # B01: j=0, k=3b+1 -> 0*9 + 3b+1
        # B10: j=1, k=3b -> 1*9 + 3b
        # B11: j=1, k=3b+1 -> 1*9 + 3b+1
        
        # C_2 indices (rows 0,1; cols 3b, 3b+1):
        # C00: k=3b, i=0 -> 3b*2 + 0
        # C01: k=3b+1, i=0 -> (3b+1)*2 + 0
        # C10: k=3b, i=1 -> 3b*2 + 1
        # C11: k=3b+1, i=1 -> (3b+1)*2 + 1
        
        A00 = 0
        A01 = 1
        A10 = 3
        A11 = 4
        
        B00 = 3*b
        B01 = 3*b + 1
        B10 = 9 + 3*b
        B11 = 9 + 3*b + 1
        
        C00 = 3*b * 2
        C01 = (3*b + 1) * 2
        C10 = 3*b * 2 + 1
        C11 = (3*b + 1) * 2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        # But Strassen sums are distributed.
        # We add 1 term per M.
        
        # Term for M1 contribution to C00, C11, etc.
        # We need to add the coefficients to W.
        # M1 contributes +1 to C00, -1 to C10? No.
        # C00 = M1 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3
        
        # So for M1:
        # W[C00] += 1
        # W[C11] += 1
        # U[A00] += 1, U[A11] += 1
        # V[B00] += 1, V[B11] += 1
        add_term(A00, B00, C00, 1, 1, 1)
        add_term(A00, B11, C00, 1, 1, 0) # Wait, this splits the product?
        
        # No, the decomposition is:
        # U[:, c] is a vector, V[:, c] is a vector.
        # The product is U[:, c] * V[:, c] added to C via W[:, c].
        # If M1 = (A00+A11)(B00+B11), we can represent this as:
        # U has 1 at A00 and A11.
        # V has 1 at B00 and B11.
        # This creates 4 products: A00B00, A00B11, A11B00, A11B11.
        # But Strassen M1 is a single scalar product of sums.
        # In tensor decomposition, each column c corresponds to ONE product of linear forms.
        # So U[:, c] must be a linear form in A, V[:, c] a linear form in B.
        # For M1:
        # U has 1 at A00, 1 at A11.
        # V has 1 at B00, 1 at B11.
        # This is ONE column in the decomposition.
        # The resulting term is (A00+A11)(B00+B11).
        # We then add this term to the output C using W.
        # W[C00] += 1
        # W[C11] += 1
        # (Other C entries get 0 for this M1)
        
        # Let's re-implement add_term to take the whole linear form?
        # No, the standard interface is U, V, W matrices.
        # Each column c defines:
        # prod = sum(U[i,c] A[i]) * sum(V[j,c] B[j])
        # C[k] += W[k,c] * prod
        
        # So for M1:
        # U[A00]=1, U[A11]=1.
        # V[B00]=1, V[B11]=1.
        # W[C00]=1, W[C11]=1.
        # This adds M1 to C00 and C11.
        
        # We need to reset U, V, W for the column before filling?
        # No, we are building the matrices.
        # Each column is independent.
        
        # Let's write a helper that fills one column.
        def fill_col(U_vec, V_vec, W_vec):
            nonlocal col
            if time.time() > deadline:
                return
            for i, val in enumerate(U_vec):
                if val != 0:
                    U[i][col] = val
            for j, val in enumerate(V_vec):
                if val != 0:
                    V[j][col] = val
            for k, val in enumerate(W_vec):
                if val != 0:
                    W[k][col] = val
            col += 1
            
        # Re-do Part 1
        # M1
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A00] = 1; U_vec[A11] = 1
        V_vec[B00] = 1; V_vec[B11] = 1
        W_vec[C00] = 1; W_vec[C11] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # M2 = (A10+A11)B00
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A10] = 1; U_vec[A11] = 1
        V_vec[B00] = 1
        W_vec[C10] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # M3 = A00(B01-B11)
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A00] = 1
        V_vec[B01] = 1; V_vec[B11] = -1
        W_vec[C01] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # M4 = A11(B10-B00)
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A11] = 1
        V_vec[B10] = 1; V_vec[B00] = -1
        W_vec[C00] = -1; W_vec[C10] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # M5 = (A00+A01)B11
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A00] = 1; U_vec[A01] = 1
        V_vec[B11] = 1
        W_vec[C00] = 1; W_vec[C01] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # M6 = (A10-A00)(B00+B01)
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A10] = 1; U_vec[A00] = -1
        V_vec[B00] = 1; V_vec[B01] = 1
        W_vec[C11] = -1
        fill_col(U_vec, V_vec, W_vec)
        
        # M7 = (A01-A11)(B10+B11)
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A01] = 1; U_vec[A11] = -1
        V_vec[B10] = 1; V_vec[B11] = 1
        W_vec[C11] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # --- Part 2: Schoolbook for A_2 * B_2_col2 ---
        # B_2_col2: row 0,1; col 3b+2
        B20 = 3*b + 2       # j=0, k=3b+2
        B21 = 9 + 3*b + 2   # j=1, k=3b+2
        
        C20 = (3*b + 2) * 2       # k=3b+2, i=0
        C21 = (3*b + 2) * 2 + 1   # k=3b+2, i=1
        
        # P1 = A00 B20
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A00] = 1
        V_vec[B20] = 1
        W_vec[C20] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # P2 = A01 B21
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A01] = 1
        V_vec[B21] = 1
        W_vec[C20] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # P3 = A10 B20
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A10] = 1
        V_vec[B20] = 1
        W_vec[C21] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # P4 = A11 B21
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A11] = 1
        V_vec[B21] = 1
        W_vec[C21] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # --- Part 3: Schoolbook for A_1 * B_1 ---
        # A_1: col 2. A02, A12.
        A02 = 2
        A12 = 5
        
        # B_1: row 2; cols 3b, 3b+1, 3b+2
        B30 = 18 + 3*b       # j=2, k=3b
        B31 = 18 + 3*b + 1   # j=2, k=3b+1
        B32 = 18 + 3*b + 2   # j=2, k=3b+2
        
        C30 = 3*b * 2       # k=3b, i=0
        C31 = (3*b + 1) * 2 # k=3b+1, i=0
        C32 = (3*b + 2) * 2 # k=3b+2, i=0
        C40 = 3*b * 2 + 1   # k=3b, i=1
        C41 = (3*b + 1) * 2 + 1 # k=3b+1, i=1
        C42 = (3*b + 2) * 2 + 1 # k=3b+2, i=1
        
        # Q1 = A02 B30
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A02] = 1
        V_vec[B30] = 1
        W_vec[C30] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # Q2 = A02 B31
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A02] = 1
        V_vec[B31] = 1
        W_vec[C31] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # Q3 = A02 B32
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A02] = 1
        V_vec[B32] = 1
        W_vec[C32] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # Q4 = A12 B30
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A12] = 1
        V_vec[B30] = 1
        W_vec[C40] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # Q5 = A12 B31
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A12] = 1
        V_vec[B31] = 1
        W_vec[C41] = 1
        fill_col(U_vec, V_vec, W_vec)
        
        # Q6 = A12 B32
        U_vec = [0]*(n*m); V_vec = [0]*(m*p); W_vec = [0]*(p*n)
        U_vec[A12] = 1
        V_vec[B32] = 1
        W_vec[C42] = 1
        fill_col(U_vec, V_vec, W_vec)
        
    return {"U": U, "V": V, "W": W}
