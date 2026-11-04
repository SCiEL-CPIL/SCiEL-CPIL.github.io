import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition term by term.
    # U, V, W are lists of lists.
    # We don't know the rank ahead of time, so we use a list of tuples (u_idx, v_idx, w_idx, coeff)
    # and then assemble the matrices.
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        if coeff == 0:
            return
        terms.append((u_idx, v_idx, w_idx, coeff))

    # Helper to add schoolbook terms for a block product
    # A is r x c1, B is c1 x c2
    # Global indices:
    # i_global = i_offset + i_local
    # j_global = j_offset + j_local
    # k_global = k_offset + k_local
    def add_schoolbook_block(r, c1, c2, i_off, j_off, k_off):
        for i in range(r):
            for k in range(c2):
                for j in range(c1):
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    add_term(u_idx, v_idx, w_idx, 1)

    # Helper to add Strassen terms for 2x2 block
    def add_strassen_block(i_off, j_off, k_off):
        # Local indices 0, 1
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to express each M as sum of terms.
        # M1: (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
        # This requires 4 terms.
        # Wait, Strassen reduces the number of multiplications, but in tensor decomposition,
        # each "multiplication" M_r is a product of two linear forms.
        # A linear form in A is a sum of a_ij.
        # A linear form in B is a sum of b_jk.
        # The product (sum a_ij) (sum b_jk) is a sum of a_ij b_jk terms.
        # In the tensor decomposition, we add one term for each non-zero coefficient in the expansion?
        # No. The tensor decomposition represents the bilinear form.
        # If we use Strassen, we compute M1, ..., M7.
        # M1 is a scalar. It is computed by multiplying two vectors.
        # The contribution of M1 to C is M1 * (matrix of coefficients).
        # So for each M_r, we add terms corresponding to the non-zero entries of the coefficient matrix for C.
        # But M_r itself is a product of linear forms.
        # L_A = sum alpha_ij a_ij
        # L_B = sum beta_jk b_jk
        # M_r = L_A L_B
        # Contribution to C_{ik} is coeff_{ik} * M_r.
        # So we add a term for each non-zero coeff_{ik} with U = alpha, V = beta, W = coeff.
        
        # Let's implement this logic.
        
        # Define linear forms for M1..M7
        # M1: A_form: a00+a11, B_form: b00+b11
        # M2: A_form: a10+a11, B_form: b00
        # M3: A_form: a00, B_form: b01-b11
        # M4: A_form: a11, B_form: b10-b00
        # M5: A_form: a00+a01, B_form: b11
        # M6: A_form: a10-a00, B_form: b00+b01
        # M7: A_form: a01-a11, B_form: b10+b11
        
        # Coefficients for C:
        # C00: M1(1), M4(1), M5(-1), M7(1)
        # C01: M3(1), M5(1)
        # C10: M2(1), M4(1)
        # C11: M1(1), M2(-1), M3(1), M6(1)
        
        m1_a = [(0,0,1), (1,1,1)]
        m1_b = [(0,0,1), (1,1,1)]
        m1_c = [(0,0,1)]
        
        m2_a = [(1,0,1), (1,1,1)]
        m2_b = [(0,0,1)]
        m2_c = [(1,0,1)]
        
        m3_a = [(0,0,1)]
        m3_b = [(0,1,1), (1,1,-1)]
        m3_c = [(0,1,1)]
        
        m4_a = [(1,1,1)]
        m4_b = [(1,0,1), (0,0,-1)]
        m4_c = [(0,0,1), (1,0,1)]
        
        m5_a = [(0,0,1), (0,1,1)]
        m5_b = [(1,1,1)]
        m5_c = [(0,0,-1), (0,1,1)]
        
        m6_a = [(1,0,1), (0,0,-1)]
        m6_b = [(0,0,1), (0,1,1)]
        m6_c = [(1,1,1)]
        
        m7_a = [(0,1,1), (1,1,-1)]
        m7_b = [(1,0,1), (1,1,1)]
        m7_c = [(0,0,1)]
        
        def add_m(a_form, b_form, c_form):
            for (ai, aj, ac) in a_form:
                for (bi, bj, bc) in b_form:
                    for (ci, cj, cc) in c_form:
                        # Global indices
                        gi = i_off + ci
                        gj = j_off + ai # Note: A index is (i, j). In C, i is row, j is col?
                        # Wait. A is (i, j). B is (j, k). C is (k, i).
                        # A_form uses (i, j).
                        # B_form uses (j, k).
                        # C_form uses (k, i).
                        
                        # In my definitions above:
                        # m1_a: (0,0) means i=0, j=0.
                        # m1_b: (0,0) means j=0, k=0.
                        # m1_c: (0,0) means k=0, i=0.
                        
                        u_idx = gi * m + (j_off + aj) # A[i, j]
                        v_idx = (j_off + bi) * p + (k_off + bj) # B[j, k] -- Wait.
                        # B index is (j, k).
                        # In b_form, the first index is j, second is k.
                        # So bi is j, bj is k.
                        v_idx = (j_off + bi) * p + (k_off + bj)
                        
                        w_idx = (k_off + ci) * n + gi # C[k, i]
                        # In c_form, first index is k, second is i.
                        # So ci is k, cj is i.
                        # But I used gi = i_off + ci.
                        # If ci is k, then gi should be k_off + ci.
                        # And the i in w_idx should be i_off + cj.
                        
                        # Let's re-verify indices.
                        # A[i, j] -> U[i*m + j]
                        # B[j, k] -> V[j*p + k]
                        # C[k, i] -> W[k*n + i]
                        
                        # a_form: (i, j, coeff)
                        # b_form: (j, k, coeff)
                        # c_form: (k, i, coeff)
                        
                        ui = i_off + ai
                        uj = j_off + aj
                        vi = j_off + bi
                        vk = k_off + bj
                        wi = k_off + ci
                        wj = i_off + cj # i in C[k, i]
                        
                        u_idx = ui * m + uj
                        v_idx = vi * p + vk
                        w_idx = wi * n + wj
                        
                        add_term(u_idx, v_idx, w_idx, ac * bc * cc)

        add_m(m1_a, m1_b, m1_c)
        add_m(m2_a, m2_b, m2_c)
        add_m(m3_a, m3_b, m3_c)
        add_m(m4_a, m4_b, m4_c)
        add_m(m5_a, m5_b, m5_c)
        add_m(m6_a, m6_b, m6_c)
        add_m(m7_a, m7_b, m7_c)

    # Helper to add 3x3 block using 2+1 split (Rank 26)
    def add_3x3_block(i_off, j_off, k_off):
        # Split 3 into 2 and 1.
        # A = [[A22, A21], [A12, A11]]
        # A22: 2x2, A21: 2x1, A12: 1x2, A11: 1x1
        # B = [[B22, B21], [B12, B11]]
        
        # C22 (2x2) = A22 B22 + A21 B12
        # C21 (2x1) = A22 B21 + A21 B11
        # C12 (1x2) = A12 B22 + A11 B12
        # C11 (1x1) = A12 B21 + A11 B11
        
        # 1. C22 part 1: A22 (2x2) B22 (2x2) -> 2x2
        # Use Strassen for this 2x2 sub-block.
        # Indices for A22: i in [0,1], j in [0,1] (local to 3x3)
        # Global i: i_off + [0,1], j: j_off + [0,1]
        # k for B22: k_off + [0,1]
        add_strassen_block(i_off, j_off, k_off)
        
        # 2. C22 part 2: A21 (2x1) B12 (1x2) -> 2x2
        # A21: i in [0,1], j=2 (local)
        # B12: j=2 (local), k in [0,1]
        # Product C22: k in [0,1], i in [0,1]
        # Schoolbook for 2x1 by 1x2.
        # r=2, c1=1, c2=2.
        # i_off, j_off+2, k_off
        add_schoolbook_block(2, 1, 2, i_off, j_off+2, k_off)
        
        # 3. C21 (2x1) = A22 (2x2) B21 (2x1) + A21 (2x1) B11 (1x1)
        # Part 1: A22 (2x2) B21 (2x1) -> 2x1
        # i in [0,1], j in [0,1], k=2 (local)
        # r=2, c1=2, c2=1.
        add_schoolbook_block(2, 2, 1, i_off, j_off, k_off+2)
        
        # Part 2: A21 (2x1) B11 (1x1) -> 2x1
        # i in [0,1], j=2, k=2
        # r=2, c1=1, c2=1.
        add_schoolbook_block(2, 1, 1, i_off, j_off+2, k_off+2)
        
        # 4. C12 (1x2) = A12 (1x2) B22 (2x2) + A11 (1x1) B12 (1x2)
        # Part 1: A12 (1x2) B22 (2x2) -> 1x2
        # i=2, j in [0,1], k in [0,1]
        # r=1, c1=2, c2=2.
        add_schoolbook_block(1, 2, 2, i_off+2, j_off, k_off)
        
        # Part 2: A11 (1x1) B12 (1x2) -> 1x2
        # i=2, j=2, k in [0,1]
        # r=1, c1=1, c2=2.
        add_schoolbook_block(1, 1, 2, i_off+2, j_off+2, k_off)
        
        # 5. C11 (1x1) = A12 (1x2) B21 (2x1) + A11 (1x1) B11 (1x1)
        # Part 1: A12 (1x2) B21 (2x1) -> 1x1
        # i=2, j in [0,1], k=2
        # r=1, c1=2, c2=1.
        add_schoolbook_block(1, 2, 1, i_off+2, j_off, k_off+2)
        
        # Part 2: A11 (1x1) B11 (1x1) -> 1x1
        # i=2, j=2, k=2
        # r=1, c1=1, c2=1.
        add_schoolbook_block(1, 1, 1, i_off+2, j_off+2, k_off+2)

    # Construct the 5x5 decomposition
    # Block 1: A11 (3x3) B11 (3x3) -> C11 (3x3)
    add_3x3_block(0, 0, 0)
    
    # Block 2: A12 (3x2) B21 (2x3) -> C21 (3x3)
    # A12: i in [0,2], j in [3,4]
    # B21: j in [3,4], k in [0,2]
    # C21: k in [0,2], i in [0,2]
    # r=3, c1=2, c2=3.
    add_schoolbook_block(3, 2, 3, 0, 3, 0)
    
    # Block 3: A12 (3x2) B22 (2x2) -> C22 (3x2)
    # A12: i in [0,2], j in [3,4]
    # B22: j in [3,4], k in [3,4]
    # C22: k in [3,4], i in [0,2]
    # r=3, c1=2, c2=2.
    add_schoolbook_block(3, 2, 2, 0, 3, 3)
    
    # Block 4: A21 (2x3) B11 (3x3) -> C11 (2x3)
    # A21: i in [3,4], j in [0,2]
    # B11: j in [0,2], k in [0,2]
    # C11: k in [0,2], i in [3,4]
    # r=2, c1=3, c2=3.
    add_schoolbook_block(2, 3, 3, 3, 0, 0)
    
    # Block 5: A21 (2x3) B12 (3x2) -> C12 (2x2)
    # A21: i in [3,4], j in [0,2]
    # B12: j in [0,2], k in [3,4]
    # C12: k in [3,4], i in [3,4]
    # r=2, c1=3, c2=2.
    add_schoolbook_block(2, 3, 2, 3, 0, 3)
    
    # Block 6: A22 (2x2) B21 (2x3) -> C21 (2x3)
    # A22: i in [3,4], j in [3,4]
    # B21: j in [3,4], k in [0,2]
    # C21: k in [0,2], i in [3,4]
    # r=2, c1=2, c2=3.
    add_schoolbook_block(2, 2, 3, 3, 3, 0)
    
    # Block 7: A22 (2x2) B22 (2x2) -> C22 (2x2)
    # A22: i in [3,4], j in [3,4]
    # B22: j in [3,4], k in [3,4]
    # C22: k in [3,4], i in [3,4]
    # Use Strassen.
    add_strassen_block(3, 3, 3)

    # Assemble U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx][c] = coeff
        V[v_idx][c] = coeff
        W[w_idx][c] = coeff
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
