def construct():
    n, m, p = 2, 5, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_strassen_block(i0, j0, k0):
        # A block: i in {i0, i0+1}, j in {j0, j0+1}
        # B block: j in {j0, j0+1}, k in {k0, k0+1}
        # C block: k in {k0, k0+1}, i in {i0, i0+1}
        
        # U indices
        u00 = i0 * m + j0
        u01 = i0 * m + (j0 + 1)
        u10 = (i0 + 1) * m + j0
        u11 = (i0 + 1) * m + (j0 + 1)
        
        # V indices
        v00 = j0 * p + k0
        v01 = j0 * p + (k0 + 1)
        v10 = (j0 + 1) * p + k0
        v11 = (j0 + 1) * p + (k0 + 1)
        
        # W indices
        w00 = k0 * n + i0
        w01 = (k0 + 1) * n + i0
        w10 = k0 * n + (i0 + 1)
        w11 = (k0 + 1) * n + (i0 + 1)
        
        # Strassen terms
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # We create 7 vectors of length 10, 30, 12
        
        # Helper to create zero vectors
        def make_vec(length, indices_values):
            vec = [0] * length
            for idx, val in indices_values:
                vec[idx] = val
            return vec
            
        # Term 1
        add_term(
            make_vec(n*m, [(u00, 1), (u11, 1)]),
            make_vec(m*p, [(v00, 1), (v11, 1)]),
            make_vec(p*n, [(w00, 1), (w01, 1), (w10, 1), (w11, 1)]) # Placeholder, will fix W mapping
        )
        # Wait, W mapping for C:
        # C00 -> w00
        # C01 -> w01 (k+1, i)
        # C10 -> w10 (k, i+1)
        # C11 -> w11 (k+1, i+1)
        
        # The W vector for a term should have 1s at the positions of the C entries that are formed by this term?
        # No, the tensor decomposition is $\sum r U_r V_r W_r$.
        # $C = \sum_r (U_r \cdot A) (V_r \cdot B) W_r$?
        # No, the standard form is:
        # $C_{ij} = \sum_r \lambda_r (U_r \cdot A) (V_r \cdot B)$?
        # The interface says:
        # U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # This implies $C = \sum_r u_r v_r^T w_r^T$?
        # Usually, the decomposition is $T = \sum_r u_r \otimes v_r \otimes w_r$.
        # Where $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
        # So for each term r, we define $U_r$ (vector of coeffs for A), $V_r$ (vector of coeffs for B), $W_r$ (vector of coeffs for C).
        # The contribution to $C_{i,k}$ is $W_{k,i,r} \times (U_r \cdot A) \times (V_r \cdot B)$.
        
        # So for Strassen M1:
        # $M1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This value is added to $C_{00}, C_{01}, C_{10}, C_{11}$ with specific coefficients.
        # $C_{00} = M1 + M4 - M5 + M7$
        # $C_{01} = M3 + M5$
        # $C_{10} = M2 + M4$
        # $C_{11} = M1 - M2 + M3 + M6$
        
        # So for Term 1 (M1):
        # U: coeffs for A. $A_{00}$ has 1, $A_{11}$ has 1.
        # V: coeffs for B. $B_{00}$ has 1, $B_{11}$ has 1.
        # W: coeffs for C. $C_{00}$ gets 1, $C_{01}$ gets 0, $C_{10}$ gets 0, $C_{11}$ gets 1?
        # Wait, $M1$ contributes 1 to $C_{00}$ and 1 to $C_{11}$.
        # So W vector has 1 at $w00$ and 1 at $w11$.
        
        # Let's re-verify Strassen formulas.
        # $M_1 = (A_{11} + A_{22})(B_{11} + B_{22})$
        # $C_{11} = M_1 + M_4 - M_5 + M_7$
        # $C_{12} = M_3 + M_5$
        # $C_{21} = M_2 + M_4$
        # $C_{22} = M_1 - M_2 + M_3 + M_6$
        
        # Indices in 1-based:
        # A11, A12, A21, A22.
        # B11, B12, B21, B22.
        # C11, C12, C21, C22.
        
        # M1: A11, A22. B11, B22.
        # Contributes to C11 (+1), C22 (+1).
        
        # M2: (A21+A22)B11.
        # A21, A22. B11.
        # Contributes to C21 (+1).
        
        # M3: A11(B12-B22).
        # A11. B12, -B22.
        # Contributes to C12 (+1), C22 (+1).
        
        # M4: A22(B21-B11).
        # A22. B21, -B11.
        # Contributes to C11 (+1), C21 (+1).
        
        # M5: (A11+A12)B22.
        # A11, A12. B22.
        # Contributes to C12 (+1), C11 (-1).
        
        # M6: (A21-A11)(B11+B12).
        # A21, -A11. B11, B12.
        # Contributes to C22 (+1).
        
        # M7: (A12-A22)(B21+B22).
        # A12, -A22. B21, B22.
        # Contributes to C11 (+1).
        
        # Check C11: M1 + M4 - M5 + M7.
        # 1 + 1 - 1 + 1 = 2?
        # Standard Strassen:
        # $C_{11} = M_1 + M_4 - M_5 + M_7$
        # $C_{12} = M_3 + M_5$
        # $C_{21} = M_2 + M_4$
        # $C_{22} = M_1 - M_2 + M_3 + M_6$
        
        # Let's check M1 contribution to C11: 1.
        # M4 contribution to C11: 1.
        # M5 contribution to C11: -1.
        # M7 contribution to C11: 1.
        # Sum: 2?
        # Wait, $M_1 = (A_{11}+A_{22})(B_{11}+B_{22}) = A_{11}B_{11} + A_{11}B_{22} + A_{22}B_{11} + A_{22}B_{22}$.
        # $C_{11} = A_{11}B_{11} + A_{12}B_{21}$.
        # This doesn't look right.
        
        # Let's use a known correct set of Strassen coefficients.
        # Reference: Wikipedia "Strassen's algorithm"
        # $M_1 = (A_{11} + A_{22})(B_{11} + B_{22})$
        # $M_2 = (A_{21} + A_{22})B_{11}$
        # $M_3 = A_{11}(B_{12} - B_{22})$
        # $M_4 = A_{22}(B_{21} - B_{11})$
        # $M_5 = (A_{11} + A_{12})B_{22}$
        # $M_6 = (A_{21} - A_{11})(B_{11} + B_{12})$
        # $M_7 = (A_{12} - A_{22})(B_{21} + B_{22})$
        
        # $C_{11} = M_1 + M_4 - M_5 + M_7$
        # $C_{12} = M_3 + M_5$
        # $C_{21} = M_2 + M_4$
        # $C_{22} = M_1 - M_2 + M_3 + M_6$
        
        # Let's trace $C_{11} = A_{11}B_{11} + A_{12}B_{21}$.
        # M1 has $A_{11}B_{11}$.
        # M4 has $A_{22}B_{21} - A_{22}B_{11}$.
        # M5 has $A_{11}B_{22} + A_{12}B_{22}$.
        # M7 has $A_{12}B_{21} + A_{12}B_{22} - A_{22}B_{21} - A_{22}B_{22}$.
        
        # Sum for C11:
        # $A_{11}B_{11}$ (from M1)
        # $+ A_{22}B_{21} - A_{22}B_{11}$ (from M4)
        # $- (A_{11}B_{22} + A_{12}B_{22})$ (from -M5)
        # $+ A_{12}B_{21} + A_{12}B_{22} - A_{22}B_{21} - A_{22}B_{22}$ (from M7)
        
        # Terms:
        # $A_{11}B_{11}$: 1.
        # $A_{12}B_{21}$: 1.
        # $A_{22}B_{21}$: $1 - 1 = 0$.
        # $A_{22}B_{11}$: $-1$. (Wait, $C_{11}$ should not have $A_{22}B_{11}$).
        # Did I miss a term?
        # M1 also has $A_{11}B_{22}, A_{22}B_{11}, A_{22}B_{22}$.
        # M1: $A_{11}B_{11} + A_{11}B_{22} + A_{22}B_{11} + A_{22}B_{22}$.
        
        # Recalculate C11 sum:
        # $A_{11}B_{11}$: $1 (M1) - 1 (M4) + 1 (M7) = 1$.
        # $A_{11}B_{22}$: $1 (M1) - 1 (M5) = 0$.
        # $A_{22}B_{11}$: $1 (M1) - 1 (M4) = 0$.
        # $A_{22}B_{22}$: $1 (M1) - 1 (M7) = 0$.
        # $A_{12}B_{21}$: $1 (M7)$.
        # $A_{12}B_{22}$: $-1 (M5) + 1 (M7) = 0$.
        
        # Result: $A_{11}B_{11} + A_{12}B_{21}$. Correct.
        
        # So the coefficients for W are:
        # Term 1 (M1): W[00]=1, W[11]=1.
        # Term 2 (M2): W[10]=1.
        # Term 3 (M3): W[01]=1, W[11]=1.
        # Term 4 (M4): W[00]=1, W[10]=1.
        # Term 5 (M5): W[01]=1, W[00]=-1.
        # Term 6 (M6): W[11]=1.
        # Term 7 (M7): W[00]=1.

        # We will implement this.

    # Block 1: i=0, j=0, k=0. A(0,0..1), B(0..1, 0..1), C(0..1, 0)
    add_strassen_block(0, 0, 0)
    # Block 2: i=0, j=0, k=2. A(0,0..1), B(0..1, 2..3), C(2..3, 0)
    add_strassen_block(0, 0, 2)
    # Block 3: i=0, j=0, k=4. A(0,0..1), B(0..1, 4..5), C(4..5, 0)
    add_strassen_block(0, 0, 4)
    
    # Block 4: i=1, j=0, k=0. A(1,0..1), B(0..1, 0..1), C(0..1, 1)
    add_strassen_block(1, 0, 0)
    # Block 5: i=1, j=0, k=2.
    add_strassen_block(1, 0, 2)
    # Block 6: i=1, j=0, k=4.
    add_strassen_block(1, 0, 4)
    
    # Now handle the rest of A (j=2,3,4)
    # A2 is 2x3.
    # Split into A21 (2x2, j=2,3) and A22 (2x1, j=4).
    
    # A21 * B21 (2x2 * 2x6)
    # B21 rows j=2,3. Cols 0..5.
    # Split B21 into 3 blocks of 2 cols.
    add_strassen_block(0, 2, 0) # i=0, j=2, k=0
    add_strassen_block(0, 2, 2) # i=0, j=2, k=2
    add_strassen_block(0, 2, 4) # i=0, j=2, k=4
    
    add_strassen_block(1, 2, 0) # i=1, j=2, k=0
    add_strassen_block(1, 2, 2) # i=1, j=2, k=2
    add_strassen_block(1, 2, 4) # i=1, j=2, k=4
    
    # A22 * B22 (2x1 * 1x6)
    # A22 is col j=4. Rows i=0,1.
    # B22 is row j=4. Cols k=0..5.
    # C contribution is outer product.
    # For each k in 0..5:
    # C[0, k] += A[0, 4] * B[4, k]
    # C[1, k] += A[1, 4] * B[4, k]
    # This is 2 terms per k?
    # Term for C[0,k]: U[0,4]=1, V[4,k]=1, W[k,0]=1.
    # Term for C[1,k]: U[1,4]=1, V[4,k]=1, W[k,1]=1.
    
    for k in range(6):
        add_term(
            make_vec(10, [(0*5+4, 1)]),
            make_vec(30, [(4*6+k, 1)]),
            make_vec(12, [(k*2+0, 1)])
        )
        add_term(
            make_vec(10, [(1*5+4, 1)]),
            make_vec(30, [(4*6+k, 1)]),
            make_vec(12, [(k*2+1, 1)])
        )
        
    # Convert to lists of lists
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U[i][r] = U_cols[r][i]
        for i in range(m*p):
            V[i][r] = V_cols[r][i]
        for i in range(p*n):
            W[i][r] = W_cols[r][i]
            
    return {"U": U, "V": V, "W": W}
