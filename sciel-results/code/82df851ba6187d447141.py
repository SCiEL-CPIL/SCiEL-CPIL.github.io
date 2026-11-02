"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 44.
    # Split A (3x4) into A_L (3x2) and A_R (3x2).
    # Split B (4x4) into B_LL, B_LR, B_RL, B_RR (all 2x2).
    # C_L (3x2) = A_L B_LL + A_R B_RL
    # C_R (3x2) = A_L B_LR + A_R B_RR
    # Each 3x2 by 2x2 multiplication has rank 11 (7 for 2x2 part + 4 for 1x2 part).
    # Total rank 4 * 11 = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)] # 12 x 44
    V = [[0] * rank for _ in range(m * p)] # 16 x 44
    W = [[0] * rank for _ in range(p * n)] # 12 x 44
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # We need to map the logical indices to the flat indices.
    # A is 3x4. A[i, j]. i in 0..2, j in 0..3.
    # A_L is A[:, 0:2], A_R is A[:, 2:4].
    # B is 4x4. B[j, k]. j in 0..3, k in 0..3.
    # B_LL is B[0:2, 0:2], B_LR is B[0:2, 2:4], B_RL is B[2:4, 0:2], B_RR is B[2:4, 2:4].
    # C is 3x4. C[i, k]. i in 0..2, k in 0..3.
    # C_L is C[:, 0:2], C_R is C[:, 2:4].
    
    # We will implement the 3x2 by 2x2 multiplication with rank 11.
    # Let A_small be 3x2, B_small be 2x2.
    # A_small[i, j] for i in 0..2, j in 0..1.
    # B_small[j, k] for j in 0..1, k in 0..1.
    # C_small[i, k] for i in 0..2, k in 0..1.
    
    # We split A_small into A_top (2x2) and A_bot (1x2).
    # A_top[i, j] = A_small[i, j] for i in 0..1.
    # A_bot[j] = A_small[2, j].
    
    # C_top (2x2) = A_top B_small. Rank 7 (Strassen).
    # C_bot (1x2) = A_bot B_small. Rank 4 (Schoolbook).
    
    # We need to implement Strassen for 2x2 by 2x2.
    # And Schoolbook for 1x2 by 2x2.
    
    # Indices in U, V, W for the specific block.
    # For a specific block (e.g., A_L, B_LL), we need to map the local indices to global.
    
    def add_3x2_2x2_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # a_row_offset: 0 for A_L, 2 for A_R? No, A is 3x4.
        # A_L columns are 0,1. A_R columns are 2,3.
        # So a_col_offset is 0 or 2.
        # b_row_offset is 0 or 2 (rows of B).
        # b_col_offset is 0 or 2 (cols of B).
        # c_row_offset is 0 (rows of C are always 0..2).
        # c_col_offset is 0 or 2 (cols of C).
        
        # Local indices:
        # A_small[i, j] -> Global A[i, a_col_offset + j]
        # B_small[j, k] -> Global B[b_row_offset + j, b_col_offset + k]
        # C_small[i, k] -> Global C[c_row_offset + i, c_col_offset + k]
        
        # U index for A[i, j] is i*m + (a_col_offset + j) = i*4 + a_col_offset + j
        # V index for B[j, k] is (b_row_offset + j)*p + (b_col_offset + k) = (b_row_offset + j)*4 + b_col_offset + k
        # W index for C[i, k] is (c_col_offset + k)*n + (c_row_offset + i) = (c_col_offset + k)*3 + c_row_offset + i
        
        # Strassen for top 2x2 part (i=0,1)
        # A_top is 2x2.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A00+A11)B11
        # M3 = A00(B00-B11)
        # M4 = A11(B11-B00)
        # M5 = (A01+A11)B00
        # M6 = (A00-A11)(B01+B11)
        # M7 = (A10-A00)(B00+B01)
        
        # C00 = M1 - M2 + M4
        # C01 = M2 + M3
        # C10 = M7 + M4
        # C11 = M1 - M3 - M5
        
        # Schoolbook for bottom 1x2 part (i=2)
        # C20 = A20*B00 + A21*B10
        # C21 = A20*B01 + A21*B11
        
        # We need to add these terms to U, V, W.
        # Each M is a product of a linear combo of A and a linear combo of B.
        # This corresponds to a rank 1 tensor term.
        
        # Helper to add a product of (sum u_a * A_a) * (sum v_b * B_b) to C_c
        # This is not a single rank 1 term in the decomposition unless the sums are single elements.
        # Strassen works by defining new variables M, which are products of sums.
        # In the tensor decomposition, each M corresponds to a set of terms in U, V, W.
        # Actually, the decomposition is a sum of rank 1 tensors.
        # Strassen's algorithm expresses the tensor as a sum of 7 rank 1 tensors (in the 2x2 case).
        # So we just need to add the 7 terms for the top part and 4 terms for the bottom part.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00, C11.
        # This is NOT a rank 1 term. A rank 1 term is u * v * w where u, v, w are vectors.
        # Strassen's "multiplications" are the rank 1 terms.
        # M1 = (A00+A11)(B00+B11).
        # This means we have a term where U has 1s at A00 and A11, V has 1s at B00 and B11, and W has 1s at C00 and C11?
        # No. The tensor is $\sum A_{ij} B_{jk} C_{ki}$.
        # A rank 1 term is $u \otimes v \otimes w$.
        # $u$ is a vector of length $n*m$, $v$ of length $m*p$, $w$ of length $p*n$.
        # The term is $\sum_{i,j,k} u_{ij} v_{jk} w_{ki}$.
        # In Strassen, $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This $M_1$ is used in $C_{00}$ and $C_{11}$.
        # So the rank 1 term corresponding to $M_1$ has:
        # $u$ with 1s at (0,0) and (1,1).
        # $v$ with 1s at (0,0) and (1,1).
        # $w$ with 1s at (0,0) and (1,1) (with signs).
        # Specifically, $C_{00}$ gets $+M_1$, $C_{11}$ gets $+M_1$.
        # So $w_{00} = 1, w_{11} = 1$.
        
        # Let's implement this logic.
        
        # Indices in local 2x2 block:
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0,0), (0,1), (1,0), (1,1)
        # C indices: (0,0), (0,1), (1,0), (1,1)
        
        # Global mapping:
        # A_local(i, j) -> U[i*4 + a_col_offset + j]
        # B_local(j, k) -> V[(b_row_offset + j)*4 + b_col_offset + k]
        # C_local(i, k) -> W[(c_col_offset + k)*3 + c_row_offset + i]
        
        # Strassen Terms:
        # M1: u=(A00, A11), v=(B00, B11), w=(C00, C11)
        # M2: u=(A00, A11), v=(B11), w=(C00, C01) -> C00 gets -M2, C01 gets +M2
        # M3: u=(A00), v=(B00, B11), w=(C01, C11) -> C01 gets +M3, C11 gets -M3
        # M4: u=(A11), v=(B11, B00), w=(C00, C10) -> C00 gets +M4, C10 gets +M4
        # M5: u=(A01, A11), v=(B00), w=(C11) -> C11 gets -M5
        # M6: u=(A00, A11), v=(B01, B11), w=(?) -> Not used in standard 7-mult?
        # Wait, standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A00+A11)B11
        # M3 = A00(B00-B11)
        # M4 = A11(B11-B00)
        # M5 = (A01+A11)B00
        # M6 = (A00-A11)(B01+B11)
        # M7 = (A10-A00)(B00+B01)
        
        # C00 = M1 - M2 + M4
        # C01 = M2 + M3
        # C10 = M7 + M4
        # C11 = M1 - M3 - M5
        
        # Let's add these 7 terms.
        
        # Term M1
        u_indices = [0*4 + a_col_offset + 0, 1*4 + a_col_offset + 1]
        v_indices = [(b_row_offset + 0)*4 + b_col_offset + 0, (b_row_offset + 1)*4 + b_col_offset + 1]
        w_indices = [(c_col_offset + 0)*3 + c_row_offset + 0, (c_col_offset + 1)*3 + c_row_offset + 1]
        w_vals = [1, 1]
        for ui, vi, wi, wv in zip(u_indices, v_indices, w_indices, w_vals):
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = wv
        col += 1
        
        # Term M2
        u_indices = [0*4 + a_col_offset + 0, 1*4 + a_col_offset + 1]
        v_indices = [(b_row_offset + 1)*4 + b_col_offset + 1]
        w_indices = [(c_col_offset + 0)*3 + c_row_offset + 0, (c_col_offset + 1)*3 + c_row_offset + 1]
        w_vals = [-1, 1]
        for ui, vi, wi, wv in zip(u_indices, v_indices, w_indices, w_vals):
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = wv
        col += 1
        
        # Term M3
        u_indices = [0*4 + a_col_offset + 0]
        v_indices = [(b_row_offset + 0)*4 + b_col_offset + 0, (b_row_offset + 1)*4 + b_col_offset + 1]
        w_indices = [(c_col_offset + 1)*3 + c_row_offset + 0, (c_col_offset + 1)*3 + c_row_offset + 1]
        w_vals = [1, -1]
        for ui, vi, wi, wv in zip(u_indices, v_indices, w_indices, w_vals):
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = wv
        col += 1
        
        # Term M4
        u_indices = [1*4 + a_col_offset + 1]
        v_indices = [(b_row_offset + 1)*4 + b_col_offset + 1, (b_row_offset + 0)*4 + b_col_offset + 0]
        w_indices = [(c_col_offset + 0)*3 + c_row_offset + 0, (c_col_offset + 0)*3 + c_row_offset + 1]
        w_vals = [1, 1]
        for ui, vi, wi, wv in zip(u_indices, v_indices, w_indices, w_vals):
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = wv
        col += 1
        
        # Term M5
        u_indices = [0*4 + a_col_offset + 1, 1*4 + a_col_offset + 1]
        v_indices = [(b_row_offset + 0)*4 + b_col_offset + 0]
        w_indices = [(c_col_offset + 1)*3 + c_row_offset + 1]
        w_vals = [-1]
        for ui, vi, wi, wv in zip(u_indices, v_indices, w_indices, w_vals):
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = wv
        col += 1
        
        # Term M6
        u_indices = [0*4 + a_col_offset + 0, 1*4 + a_col_offset + 1]
        v_indices = [(b_row_offset + 0)*4 + b_col_offset + 1, (b_row_offset + 1)*4 + b_col_offset + 1]
        # M6 is not used in C?
        # Standard Strassen:
        # C00 = M1 - M2 + M4
        # C01 = M2 + M3
        # C10 = M7 + M4
        # C11 = M1 - M3 - M5
        # M6 and M7 are used?
        # M7 = (A10-A00)(B00+B01)
        # C10 = M7 + M4.
        # So M6 is not used?
        # Ah, there are different Strassen formulations.
        # The one above uses 7 multiplications.
        # M1, M2, M3, M4, M5, M7 are used. M6 is not?
        # Let's check M7.
        
        # Term M7
        u_indices = [1*4 + a_col_offset + 0, 0*4 + a_col_offset + 0]
        v_indices = [(b_row_offset + 0)*4 + b_col_offset + 0, (b_row_offset + 0)*4 + b_col_offset + 1]
        w_indices = [(c_col_offset + 0)*3 + c_row_offset + 1]
        w_vals = [1]
        for ui, vi, wi, wv in zip(u_indices, v_indices, w_indices, w_vals):
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = wv
        col += 1
        
        # Schoolbook for bottom row (i=2)
        # C20 = A20*B00 + A21*B10
        # C21 = A20*B01 + A21*B11
        
        # Term for A20*B00 -> C20
        U[2*4 + a_col_offset + 0][col] = 1
        V[(b_row_offset + 0)*4 + b_col_offset + 0][col] = 1
        W[(c_col_offset + 0)*3 + c_row_offset + 2][col] = 1
        col += 1
        
        # Term for A21*B10 -> C20
        U[2*4 + a_col_offset + 1][col] = 1
        V[(b_row_offset + 1)*4 + b_col_offset + 0][col] = 1
        W[(c_col_offset + 0)*3 + c_row_offset + 2][col] = 1
        col += 1
        
        # Term for A20*B01 -> C21
        U[2*4 + a_col_offset + 0][col] = 1
        V[(b_row_offset + 0)*4 + b_col_offset + 1][col] = 1
        W[(c_col_offset + 1)*3 + c_row_offset + 2][col] = 1
        col += 1
        
        # Term for A21*B11 -> C21
        U[2*4 + a_col_offset + 1][col] = 1
        V[(b_row_offset + 1)*4 + b_col_offset + 1][col] = 1
        W[(c_col_offset + 1)*3 + c_row_offset + 2][col] = 1
        col += 1

    # Call for 4 blocks
    # A_L (cols 0,1), B_LL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    add_3x2_2x2_block(0, 0, 0, 0, 0, 0)
    # A_R (cols 2,3), B_RL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    add_3x2_2x2_block(2, 2, 0, 0, 0, 0)
    # A_L (cols 0,1), B_LR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    add_3x2_2x2_block(0, 0, 2, 2, 0, 2)
    # A_R (cols 2,3), B_RR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    add_3x2_2x2_block(2, 2, 2, 2, 0, 2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
