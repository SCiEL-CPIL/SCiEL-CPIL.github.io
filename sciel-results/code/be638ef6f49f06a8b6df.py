"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_strassen_2x2x2(a_rows, b_cols_start, b_cols_end, c_rows):
        nonlocal current_rank
        # a_rows: list of 2 row indices in A
        # b_cols_start, b_cols_end: column range in B for the 2x2 block
        # c_rows: list of 2 row indices in C (always 0, 1 here)
        
        # Strassen coefficients
        # P1 = (a00+a11)(b00+b11)
        # P2 = (a10+a11)b00
        # P3 = a00(b11-b01)
        # P4 = a11(a10-a00) -- Wait, standard Strassen:
        # M1 = (a+d)(e+f)
        # M2 = (c+d)e
        # M3 = a(f-e)
        # M4 = d(b-c)
        # M5 = (a+b)f
        # M6 = (c-d)(e+f)
        # M7 = (b-c)(e+d)
        
        # A indices: a_rows[0], a_rows[1]
        # B indices: b_cols_start, b_cols_start+1 (rows 0,1 of the sub-block B)
        # Note: B in this sub-problem is 2x2.
        # Global B rows are fixed for this sub-problem?
        # In our split, A1 is rows 0,1 of A. B1 is rows 0,1 of B.
        # So B sub-block uses global B rows 0,1.
        
        a0, a1 = a_rows
        b0, b1 = b_cols_start, b_cols_start + 1 # These are COLUMNS of the 2x2 block?
        # Wait, B is 2x2. Rows 0,1. Cols b0, b1.
        
        # We need to define the 7 multiplications.
        # Each multiplication is a term in U, V, W.
        
        # Helper to add a term:
        # u_idx, v_idx, w_idx, coeff
        # U[u_idx] += coeff
        # V[v_idx] += coeff
        # W[w_idx] += coeff
        
        def add_term(u_idx, v_idx, w_idx, coeff):
            nonlocal current_rank
            U[u_idx][current_rank] += coeff
            V[v_idx][current_rank] += coeff
            W[w_idx][current_rank] += coeff
        
        # Strassen Terms
        # M1: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # This is a sum of 4 products? No, Strassen combines them.
        # The "multiplication" is the scalar product of the linear combinations.
        # In the tensor decomposition, each "multiplication" corresponds to one column in U, V, W.
        # So for M1, we need to set U, V, W such that their product gives the term.
        # U has entries from A. V has entries from B.
        # M1 = (A_a0 + A_a1) * (B_b00 + B_b11)
        # This is not a single monomial. It's a product of sums.
        # We cannot represent (x+y)(z+w) as a single rank-1 term $u \cdot v$.
        # We need to expand?
        # No, the rank of the tensor counts the number of scalar multiplications.
        # Strassen uses 7 scalar multiplications.
        # Each scalar multiplication is of the form $L_A \cdot L_B$.
        # $L_A$ is a linear combination of A entries.
        # $L_B$ is a linear combination of B entries.
        # So for M1, $L_A = A_{a0,0} + A_{a1,1}$?
        # Wait, in the sub-problem, A is $2 \times 2$.
        # $A_{sub} = \begin{pmatrix} A_{a0,0} & A_{a0,1} \\ A_{a1,0} & A_{a1,1} \end{pmatrix}$?
        # No, A in the sub-problem is just the $2 \times 2$ block.
        # Let's denote sub-A entries as $a_{00}, a_{01}, a_{10}, a_{11}$.
        # These map to global A indices.
        # $a_{00} \leftrightarrow A[a0, 0]$?
        # In our split, A1 is $A[0:2, 0:2]$.
        # So $a_{00} = A[0,0], a_{01}=A[0,1], a_{10}=A[1,0], a_{11}=A[1,1]$.
        
        # Sub-B entries $b_{00}, b_{01}, b_{10}, b_{11}$.
        # $b_{00} = B[0, b0], b_{01} = B[0, b1]$
        # $b_{10} = B[1, b0], b_{11} = B[1, b1]$
        
        # M1 = (a00+a11)(b00+b11)
        # U term: A[a0,0] + A[a1,1] -> U[a0*m+0] += 1, U[a1*m+1] += 1?
        # Wait, U index is $i \cdot m + j$.
        # $a_{00}$ is $i=0, j=0$. Index $0 \cdot 5 + 0 = 0$.
        # $a_{11}$ is $i=1, j=1$. Index $1 \cdot 5 + 1 = 6$.
        # So U[0] += 1, U[6] += 1.
        
        # V term: B[0, b0] + B[1, b1].
        # V index $j \cdot p + k$.
        # $b_{00}$ is $j=0, k=b0$. Index $0 \cdot 6 + b0 = b0$.
        # $b_{11}$ is $j=1, k=b1$. Index $1 \cdot 6 + b1 = 6 + b1$.
        # So V[b0] += 1, V[6+b1] += 1.
        
        # W term: Output C[0,0] and C[1,1]?
        # Strassen M1 contributes to C00 and C11.
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 adds +1 to C00 and +1 to C11.
        # W index $k \cdot n + i$.
        # C00 is $k=0, i=0$. Index $0 \cdot 2 + 0 = 0$.
        # C11 is $k=1, i=1$. Index $1 \cdot 2 + 1 = 3$.
        # So W[0] += 1, W[3] += 1.
        
        # This logic holds. We just need to map the linear combinations correctly.
        
        # Let's define the 7 Strassen multiplications explicitly.
        
        # M1: (a00+a11)(b00+b11)
        add_term(0, 0, 0, 1) # Placeholder, will do properly below
        
        current_rank += 1
        return

    # This manual mapping is error-prone. Let's use a more systematic approach.
    # We will generate the U, V, W for each block type.
    
    # Reset
    current_rank = 0
    
    def add_2x2x2_strassen(a_row_offset, b_row_offset, c_col_offset):
        # a_row_offset: starting row in A (0 or 2)
        # b_row_offset: starting row in B (0, 2, or 4) -- Wait, B rows are 0..4.
        # In our split:
        # Block 1: A rows 0,1. B rows 0,1. C cols 0,1.
        # Block 2: A rows 0,1. B rows 0,1. C cols 2,3.
        # Block 3: A rows 0,1. B rows 0,1. C cols 4,5.
        # Block 4: A rows 0,1. B rows 2,3. C cols 0,1.
        # ...
        
        # Indices in local 2x2x2:
        # A: 0,1; 0,1
        # B: 0,1; 0,1
        # C: 0,1; 0,1
        
        # Global A indices:
        # a00: a_row_offset * 5 + 0
        # a01: a_row_offset * 5 + 1
        # a10: (a_row_offset+1) * 5 + 0
        # a11: (a_row_offset+1) * 5 + 1
        
        # Global B indices:
        # b00: b_row_offset * 6 + c_col_offset
        # b01: b_row_offset * 6 + c_col_offset + 1
        # b10: (b_row_offset+1) * 6 + c_col_offset
        # b11: (b_row_offset+1) * 6 + c_col_offset + 1
        
        # Global C indices (W):
        # c00: c_col_offset * 2 + 0
        # c01: c_col_offset * 2 + 1
        # c10: (c_col_offset+1) * 2 + 0
        # c11: (c_col_offset+1) * 2 + 1
        
        a00_idx = a_row_offset * 5 + 0
        a01_idx = a_row_offset * 5 + 1
        a10_idx = (a_row_offset + 1) * 5 + 0
        a11_idx = (a_row_offset + 1) * 5 + 1
        
        b00_idx = b_row_offset * 6 + c_col_offset
        b01_idx = b_row_offset * 6 + (c_col_offset + 1)
        b10_idx = (b_row_offset + 1) * 6 + c_col_offset
        b11_idx = (b_row_offset + 1) * 6 + (c_col_offset + 1)
        
        c00_idx = c_col_offset * 2 + 0
        c01_idx = c_col_offset * 2 + 1
        c10_idx = (c_col_offset + 1) * 2 + 0
        c11_idx = (c_col_offset + 1) * 2 + 1
        
        # M1: (a00+a11)(b00+b11)
        U[a00_idx][current_rank] += 1
        U[a11_idx][current_rank] += 1
        V[b00_idx][current_rank] += 1
        V[b11_idx][current_rank] += 1
        W[c00_idx][current_rank] += 1
        W[c11_idx][current_rank] += 1
        current_rank += 1
        
        # M2: (a10+a11)b00
        U[a10_idx][current_rank] += 1
        U[a11_idx][current_rank] += 1
        V[b00_idx][current_rank] += 1
        W[c10_idx][current_rank] += 1
        W[c11_idx][current_rank] -= 1
        current_rank += 1
        
        # M3: a00(b11-b01)
        U[a00_idx][current_rank] += 1
        V[b11_idx][current_rank] += 1
        V[b01_idx][current_rank] -= 1
        W[c01_idx][current_rank] += 1
        W[c11_idx][current_rank] += 1
        current_rank += 1
        
        # M4: a11(a01-a10)
        U[a11_idx][current_rank] += 1
        U[a01_idx][current_rank] += 1
        U[a10_idx][current_rank] -= 1
        V[b00_idx][current_rank] += 1 # Wait, M4 is d(b-c). b,c are A entries?
        # Strassen: M4 = d(b-c). Here d=a11, b=a01, c=a10.
        # So M4 = a11 * (a01 - a10).
        # This is a product of A entries? No, M4 must be A * B.
        # Standard Strassen:
        # M1 = (a+d)(e+f)
        # M2 = (c+d)e
        # M3 = a(f-e)
        # M4 = d(b-c)
        # M5 = (a+b)f
        # M6 = (c-d)(e+f)
        # M7 = (b-c)(e+d)
        # a,b,c,d are from A. e,f,g,h are from B.
        # M4 = d(b-c). This is A * A? No.
        # b and c are from A. d is from A.
        # This is not a valid bilinear term for matrix multiplication $AB$.
        # $M_k$ must be linear in A and linear in B.
        # In Strassen, $M_4 = d(b-c)$ is WRONG.
        # Correct Strassen:
        # M1 = (a+d)(e+f)
        # M2 = (c+d)e
        # M3 = a(f-e)
        # M4 = d(b-c) -> This uses only A?
        # No, $b,c$ are elements of A. $d$ is element of A.
        # So $M_4$ is quadratic in A?
        # I am misremembering Strassen.
        
        # Let's re-verify Strassen.
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+f)
        # M2 = (c+d)e
        # M3 = a(f-e)
        # M4 = d(b-c)  <-- This is indeed d * (b-c). Both from A.
        # This is NOT bilinear in A and B.
        # Ah, $M_4$ is used to compute $C_{21}$ and $C_{22}$?
        # $C_{21} = M2 + M4$.
        # $C_{22} = M1 - M2 + M3 + M6$.
        # If $M_4$ is $d(b-c)$, it doesn't depend on B.
        # That can't be right.
        
        # Correct Strassen formulas:
        # M1 = (a+d)(e+f)
        # M2 = (c+d)e
        # M3 = a(f-e)
        # M4 = d(b-c)  -> Wait, is it $d(g-c)$? No.
        # Let's check a reliable source.
        # M1 = (a+d)(e+f)
        # M2 = (c+d)e
        # M3 = a(f-e)
        # M4 = d(b-c)
        # M5 = (a+b)f
        # M6 = (c-d)(e+f)
        # M7 = (b-c)(e+d)
        
        # $C_{11} = M1 + M4 - M5 + M7$
        # $C_{12} = M3 + M5$
        # $C_{21} = M2 + M4$
        # $C_{22} = M1 - M2 + M3 + M6$
        
        # If $M_4 = d(b-c)$, then $C_{21} = (c+d)e + d(b-c) = ce + de + db - dc$.
        # True $C_{21} = ce + dg$.
        # They don't match.
        
        # My Strassen memory is faulty.
        # Let's use the standard 7-multiplication set that IS bilinear.
        # Actually, all $M_i$ must be bilinear.
        # $M_4$ in standard Strassen is $d(b-c)$?
        # No, $b,c$ are from A. $d$ is from A.
        # It seems I am confusing the indices.
        
        # Let's use a known correct set:
        # M1 = (a+d)(e+f)
        # M2 = (c+d)e
        # M3 = a(f-e)
        # M4 = d(b-c)  -> This is definitely wrong if b,c are A.
        
        # Let's look at Wikipedia:
        # M1 = (a + d)(e + f)
        # M2 = (c + d)e
        # M3 = a(f - e)
        # M4 = d(b - c)
        # M5 = (a + b)f
        # M6 = (c - d)(e + f)
        # M7 = (b - c)(e + d)
        
        # If M4 is d(b-c), and b,c are from A, then M4 is independent of B.
        # This implies the algorithm is not for matrix multiplication?
        # No, it is.
        # Wait, $C_{21} = M2 + M4$.
        # $M2 = (c+d)e = ce + de$.
        # $M4 = d(b-c) = db - dc$.
        # Sum = $ce + de + db - dc$.
        # Target $C_{21} = ce + dg$.
        # This does not match.
        
        # I will NOT implement Strassen manually to avoid bugs.
        # I will use the trivial 4-multiplication per 2x2 block.
        # Rank for 2x2x6: 3 blocks * 4 mults = 12.
        # Rank for 2x2x6: 3 blocks * 4 mults = 12.
        # Rank for 2x1x6: 12 mults.
        # Total Rank: 12 + 12 + 12 = 36.
        
        # Wait, trivial 2x2x2 is 4.
        # 3 blocks of 2x2x2 -> 12.
        # This is much better than 21.
        # And much easier to implement correctly.
        
        pass

    # Let's implement the trivial 4-multiplication for 2x2x2.
    
    def add_2x2x2_trivial(a_row_offset, b_row_offset, c_col_offset):
        nonlocal current_rank
        # A: 2x2, B: 2x2, C: 2x2
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        a00_idx = a_row_offset * 5 + 0
        a01_idx = a_row_offset * 5 + 1
        a10_idx = (a_row_offset + 1) * 5 + 0
        a11_idx = (a_row_offset + 1) * 5 + 1
        
        b00_idx = b_row_offset * 6 + c_col_offset
        b01_idx = b_row_offset * 6 + (c_col_offset + 1)
        b10_idx = (b_row_offset + 1) * 6 + c_col_offset
        b11_idx = (b_row_offset + 1) * 6 + (c_col_offset + 1)
        
        c00_idx = c_col_offset * 2 + 0
        c01_idx = c_col_offset * 2 + 1
        c10_idx = (c_col_offset + 1) * 2 + 0
        c11_idx = (c_col_offset + 1) * 2 + 1
        
        # Term 1: A00 * B00 -> C00
        U[a00_idx][current_rank] = 1
        V[b00_idx][current_rank] = 1
        W[c00_idx][current_rank] = 1
        current_rank += 1
        
        # Term 2: A01 * B10 -> C00
        U[a01_idx][current_rank] = 1
        V[b10_idx][current_rank] = 1
        W[c00_idx][current_rank] = 1
        current_rank += 1
        
        # Term 3: A00 * B01 -> C01
        U[a00_idx][current_rank] = 1
        V[b01_idx][current_rank] = 1
        W[c01_idx][current_rank] = 1
        current_rank += 1
        
        # Term 4: A01 * B11 -> C01
        U[a01_idx][current_rank] = 1
        V[b11_idx][current_rank] = 1
        W[c01_idx][current_rank] = 1
        current_rank += 1
        
        # Term 5: A10 * B00 -> C10
        U[a10_idx][current_rank] = 1
        V[b00_idx][current_rank] = 1
        W[c10_idx][current_rank] = 1
        current_rank += 1
        
        # Term 6: A11 * B10 -> C10
        U[a11_idx][current_rank] = 1
        V[b10_idx][current_rank] = 1
        W[c10_idx][current_rank] = 1
        current_rank += 1
        
        # Term 7: A10 * B01 -> C11
        U[a10_idx][current_rank] = 1
        V[b01_idx][current_rank] = 1
        W[c11_idx][current_rank] = 1
        current_rank += 1
        
        # Term 8: A11 * B11 -> C11
        U[a11_idx][current_rank] = 1
        V[b11_idx][current_rank] = 1
        W[c11_idx][current_rank] = 1
        current_rank += 1

    # Add 3 blocks for A1 (rows 0,1) with B1 (rows 0,1)
    add_2x2x2_trivial(0, 0, 0)
    add_2x2x2_trivial(0, 0, 2)
    add_2x2x2_trivial(0, 0, 4)
    
    # Add 3 blocks for A2 (rows 0,1) with B2 (rows 2,3)
    add_2x2x2_trivial(0, 2, 0)
    add_2x2x2_trivial(0, 2, 2)
    add_2x2x2_trivial(0, 2, 4)
    
    # Add 12 terms for A3 (rows 0,1) with B3 (row 4)
    # A3 is 2x1. B3 is 1x6.
    # C = A3 * B3.
    # C0j = A04 * B4j
    # C1j = A14 * B4j
    
    a0_idx = 0 * 5 + 4
    a1_idx = 1 * 5 + 4
    
    for j in range(6):
        b_idx = 4 * 6 + j
        c0_idx = j * 2 + 0
        c1_idx = j * 2 + 1
        
        # Term A04 * B4j -> C0j
        U[a0_idx][current_rank] = 1
        V[b_idx][current_rank] = 1
        W[c0_idx][current_rank] = 1
        current_rank += 1
        
        # Term A14 * B4j -> C1j
        U[a1_idx][current_rank] = 1
        V[b_idx][current_rank] = 1
        W[c1_idx][current_rank] = 1
        current_rank += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
