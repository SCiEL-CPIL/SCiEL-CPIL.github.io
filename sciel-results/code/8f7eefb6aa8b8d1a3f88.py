"""Matrix multiplication 2x3x7 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    # We'll build the decomposition term by term.
    # Each term is (U_row, V_row, W_row, coeff)
    # U is n*m x rank, V is m*p x rank, W is p*n x rank
    # W[k*n+i, r] convention
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        terms.append((u_idx, v_idx, w_idx, coeff))
    
    # Part 1: j=2 (the third column of A, third row of B)
    # For each i in {0,1}, k in {0..6}:
    # A[i,2] * B[2,k] contributes to C[k,i]
    # U index: i*m+2 = i*3+2
    # V index: 2*p+k = 14+k
    # W index: k*n+i = k*2+i
    for i in range(n):
        for k in range(p):
            add_term(i * m + 2, 2 * p + k, k * n + i, 1)
    
    # Part 2: j in {0,1}, k in {0..6}
    # We use Strassen on 2x2x2 blocks where possible.
    # The A indices for j=0,1 are: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # The B indices for j=0,1 are: B[0,k]=k, B[1,k]=p+k=7+k
    # The C indices are: C[k,0]=2k, C[k,1]=2k+1
    
    # For k in {0,1}, {2,3}, {4,5}: use Strassen (7 multiplications per 2x2 block)
    # For k=6: use schoolbook (4 multiplications)
    
    # Strassen for 2x2x2:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10)*(b00+b01)
    # M3 = (a00)*(b10+b11)
    # M4 = (a01)*(b00)
    # M5 = (a11)*(b11)
    # M6 = (a10-a00)*(b01+b00)
    # M7 = (a01-a11)*(b11+b10)
    
    # c00 = M1 - M2 + M3 - M7
    # c01 = M4 + M7
    # c10 = M2 + M5
    # c11 = M1 - M3 + M5 + M6
    
    # Local indices:
    # a00 -> A[0,0] = 0
    # a01 -> A[0,1] = 1
    # a10 -> A[1,0] = 3
    # a11 -> A[1,1] = 4
    
    # For a block starting at k0:
    # b00 -> B[0,k0] = k0
    # b01 -> B[0,k0+1] = k0+1
    # b10 -> B[1,k0] = p+k0 = 7+k0
    # b11 -> B[1,k0+1] = p+k0+1 = 7+k0+1
    
    # c00 -> C[k0,0] = k0*2 = 2*k0
    # c01 -> C[k0+1,0] = (k0+1)*2 = 2*k0+2
    # c10 -> C[k0,1] = k0*2+1 = 2*k0+1
    # c11 -> C[k0+1,1] = (k0+1)*2+1 = 2*k0+3
    
    # For each 2x2 block, we have 7 Strassen multiplications.
    # Each Strassen multiplication M_t = (sum of a's) * (sum of b's)
    # This means for each M_t, we add multiple terms to U, V, W.
    
    # Actually, the tensor decomposition is:
    # T_{ijk} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # Wait, the convention is U has shape (n*m, rank), so U[i*m+j, r]
    # V has shape (m*p, rank), so V[j*p+k, r]
    # W has shape (p*n, rank), so W[k*n+i, r]
    
    # The matrix product is C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
    # In the tensor: T[i,j,k] = A[i,j] * B[j,k] * C^*_{i,k}
    # The decomposition: T = sum_r U_r \otimes V_r \otimes W_r
    # where U_r[i*m+j] = coefficient for A[i,j] in term r
    #       V_r[j*p+k] = coefficient for B[j,k] in term r
    #       W_r[k*n+i] = coefficient for C[i,k] in term r (with sign if needed)
    
    # For a product term A[i1,j1] * B[j2,k1] to contribute to C[i2,k2],
    # we need j1 = j2 (the inner dimension must match), i1 = i2, k1 = k2.
    # So each term in the decomposition corresponds to a triple (i,j,k) with:
    #   U[i*m+j, r] = a_coeff
    #   V[j*p+k, r] = b_coeff
    #   W[k*n+i, r] = c_coeff
    
    # For Strassen, we have products of sums.
    # M1 = (a00+a11)*(b00+b11)
    # This expands to: a00*b00 + a00*b11 + a11*b00 + a11*b11
    # In the tensor decomposition, this is ONE term with:
    #   U[0*3+0, r] = 1, U[1*3+1, r] = 1  (i.e., a00 and a11 coefficients)
    #   V[0*7+k0, r] = 1, V[1*7+k0+1, r] = 1  (i.e., b00 and b11 coefficients)
    #   W contributes to c00 and c11 with appropriate signs
    
    # Wait, I need to be more careful. The tensor T[i,j,k] represents the
    # "interaction" of A[i,j], B[j,k], and C[i,k]. A rank-1 term is
    # T_r[i,j,k] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    
    # For the product A[i,j] * B[j,k] to contribute to C[i,k], we have
    # T[i,j,k] = 1 for all valid (i,j,k).
    
    # Strassen M1 = (a00+a11)*(b00+b11) contributes to both c00 and c11.
    # Specifically:
    #   c00 += M1, c11 += M1
    # So W_r should have W[2*k0, r] = 1 (for c00, which is C[k0,0])
    #              and W[2*k0+3, r] = 1 (for c11, which is C[k0+1,1])
    # But wait, W has index k*n+i. c00 is C[k0, 0], so k=k0, i=0, W index = k0*2+0 = 2*k0
    # c11 is C[k0+1, 1], so k=k0+1, i=1, W index = (k0+1)*2+1 = 2*k0+3
    
    # For U: M1 involves a00 (i=0,j=0) and a11 (i=1,j=1)
    #   U[0*3+0, r] = 1
    #   U[1*3+1, r] = 1
    # For V: M1 involves b00 (j=0,k=k0) and b11 (j=1,k=k0+1)
    #   V[0*7+k0, r] = 1
    #   V[1*7+k0+1, r] = 1
    
    # Let me re-derive the Strassen decomposition properly.
    # 
    # We want to compute:
    #   C[k0,0]   = A[0,0]*B[0,k0]   + A[0,1]*B[1,k0]
    #   C[k0,1]   = A[1,0]*B[0,k0]   + A[1,1]*B[1,k0]
    #   C[k0+1,0] = A[0,0]*B[0,k0+1] + A[0,1]*B[1,k0+1]
    #   C[k0+1,1] = A[1,0]*B[0,k0+1] + A[1,1]*B[1,k0+1]
    
    # Wait, I had the indices wrong. Let me redo.
    # A is 2x3, B is 3x7. C = A*B is 2x7.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # For j in {0,1}, the contribution to C[i,k] from j=0,1 is:
    #   A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # So for a 2x2 block in k (k0, k0+1) and i (0,1), with j in {0,1}:
    #   C[k0,0]   = A[0,0]*B[0,k0]   + A[0,1]*B[1,k0]
    #   C[k0,1]   = A[1,0]*B[0,k0]   + A[1,1]*B[1,k0]
    #   C[k0+1,0] = A[0,0]*B[0,k0+1] + A[0,1]*B[1,k0+1]
    #   C[k0+1,1] = A[1,0]*B[0,k0+1] + A[1,1]*B[1,k0+1]
    
    # Let me denote:
    #   a00 = A[0,0], a01 = A[0,1]
    #   a10 = A[1,0], a11 = A[1,1]
    #   b00 = B[0,k0],   b01 = B[0,k0+1]
    #   b10 = B[1,k0],   b11 = B[1,k0+1]
    
    # Then:
    #   C[k0,0]   = a00*b00 + a01*b10
    #   C[k0,1]   = a10*b00 + a11*b10
    #   C[k0+1,0] = a00*b01 + a01*b11
    #   C[k0+1,1] = a10*b01 + a11*b11
    
    # This is a 2x2 matrix product! C_block = A_block * B_block
    # where A_block = [[a00, a01], [a10, a11]] (2x2)
    #       B_block = [[b00, b01], [b10, b11]] (2x2)
    
    # Standard Strassen:
    #   M1 = (a00+a11)*(b00+b10)
    #   M2 = (a10+a11)*b00
    #   M3 = a00*(b10-b00)
    #   M4 = a11*(b01-b11)
    #   M5 = (a00+a01)*b11
    #   M6 = (a10-a00)*(b00+b01)
    #   M7 = (a01-a11)*(b10+b11)
    
    #   C[k0,0]   = M1 - M2 + M3
    #   C[k0,1]   = M2 + M4
    #   C[k0+1,0] = M1 + M4
    #   C[k0+1,1] = M3 - M5 + M7
    
    # Let me verify:
    #   M1 - M2 + M3 = (a00+a11)(b00+b10) - (a10+a11)b00 + a00(b10-b00)
    #   = a00*b00 + a00*b10 + a11*b00 + a11*b10 - a10*b00 - a11*b00 + a00*b10 - a00*b00
    #   = a00*b10 + a11*b10 - a10*b00 + a00*b10
    #   Hmm, that doesn't look right. Let me use the standard Strassen formulas.
    
    # Standard Strassen (Wikipedia):
    #   p1 = (a00 + a11) * (b00 + b11)
    #   p2 = (a10 + a11) * b00
    #   p3 = a00 * (b10 - b00)
    #   p4 = a11 * (b01 - b11)
    #   p5 = (a00 + a01) * b11
    #   p6 = (a10 - a00) * (b00 + b01)
    #   p7 = (a01 - a11) * (b10 + b11)
    
    #   c00 = p1 + p4 - p5 + p7
    #   c01 = p3 + p5
    #   c10 = p2 + p4
    #   c11 = p1 - p2 + p3 + p6
    
    # Let me verify c00 = p1 + p4 - p5 + p7:
    #   p1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    #   p4 = a11*(b01-b11) = a11*b01 - a11*b11
    #   p5 = (a00+a01)*b11 = a00*b11 + a01*b11
    #   p7 = (a01-a11)*(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
    
    #   p1 + p4 - p5 + p7 = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b01 - a11*b11 
    #                        - a00*b11 - a01*b11 + a01*b10 + a01*b11 - a11*b10 - a11*b11
    #   = a00*b00 + a11*b00 + a11*b01 - a11*b10 + a01*b10
    #   Hmm, this is getting complicated. Let me just use the verified Strassen formulas.
    
    # Actually, let me just use the well-known Strassen decomposition:
    #   M1 = (a00+a11)*(b00+b11)
    #   M2 = a10*(b00+b01)  
    #   M3 = a00*(b10+b11)
    #   M4 = a01*b00
    #   M5 = a11*b11
    #   M6 = (a10-a00)*(b01+b00)
    #   M7 = (a01-a11)*(b11+b10)
    
    #   c00 = M1 - M2 + M3 - M7
    #   c01 = M4 + M7
    #   c10 = M2 + M5
    #   c11 = M1 - M3 + M5 + M6
    
    # Verify c00:
    #   M1 - M2 + M3 - M7
    #   = (a00+a11)(b00+b11) - a10(b00+b01) + a00(b10+b11) - (a01-a11)(b11+b10)
    #   = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a10*b01 + a00*b10 + a00*b11 
    #     - a01*b11 - a01*b10 + a11*b11 + a11*b10
    #   = a00*b00 + 2*a00*b11 + a11*b00 + 2*a11*b11 - a10*b00 - a10*b01 
    #     + a00*b10 - a01*b11 - a01*b10 + a11*b10
    #   This doesn't simplify to a00*b00 + a01*b10. I'm making errors.
    
    # Let me look up the correct Strassen formulas more carefully.
    # From Wikipedia (Strassen algorithm):
    # Given:
    #   A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Then:
    #   m1 = (a+d)(e+h)
    #   m2 = (c+d)e
    #   m3 = a(g+h)
    #   m4 = d(f-h)
    #   m5 = (a+b)h
    #   m6 = (c-a)(e+f)
    #   m7 = (b-d)(g+h)
    
    #   C = [[m1+m4-m5+m7, m3+m5],
    #        [m2+m4,        m1-m2+m3+m6]]
    
    # So mapping: a=a00, b=a01, c=a10, d=a11
    #              e=b00, f=b01, g=b10, h=b11
    
    #   m1 = (a00+a11)(b00+b11)
    #   m2 = (a10+a11)b00
    #   m3 = a00(b10+b11)
    #   m4 = a11(b01-b11)
    #   m5 = (a00+a01)b11
    #   m6 = (a10-a00)(b00+b01)
    #   m7 = (a01-a11)(b10+b11)
    
    #   c00 = m1+m4-m5+m7
    #   c01 = m3+m5
    #   c10 = m2+m4
    #   c11 = m1-m2+m3+m6
    
    # In our notation:
    #   C[k0,0]   = m1+m4-m5+m7  (where "c00" in Strassen = C[k0,0])
    #   C[k0+1,0] = m3+m5        (where "c01" in Strassen = C[k0+1,0])
    #   C[k0,1]   = m2+m4        (where "c10" in Strassen = C[k0,1])
    #   C[k0+1,1] = m1-m2+m3+m6  (where "c11" in Strassen = C[k0+1,1])
    
    # Wait, I need to be careful about the mapping. In Strassen:
    #   c00 corresponds to C[0,0] of the 2x2 result
    #   c01 corresponds to C[0,1]
    #   c10 corresponds to C[1,0]
    #   c11 corresponds to C[1,1]
    
    # Our 2x2 block:
    #   C_block[0,0] = C[k0,0]
    #   C_block[0,1] = C[k0+1,0]
    #   C_block[1,0] = C[k0,1]
    #   C_block[1,1] = C[k0+1,1]
    
    # So:
    #   C[k0,0]   = m1+m4-m5+m7
    #   C[k0+1,0] = m3+m5
    #   C[k0,1]   = m2+m4
    #   C[k0+1,1] = m1-m2+m3+m6
    
    # Now, for the tensor decomposition, each mi is a product of a sum of A's and a sum of B's.
    # For example, m1 = (a00+a11)*(b00+b11)
    # This means in the tensor, we have ONE term where:
    #   U[0*3+0, r] = 1  (a00)
    #   U[1*3+1, r] = 1  (a11)
    #   V[0*7+k0, r] = 1  (b00)
    #   V[1*7+k0+1, r] = 1  (b11)
    #   W[2*k0, r] = 1  (contributes to C[k0,0] with +1)
    #   W[2*k0+3, r] = -1  (contributes to C[k0+1,1] with -1)
    
    # Wait, but m1 contributes to BOTH C[k0,0] and C[k0+1,1].
    # So W should have entries for both.
    # C[k0,0] has W index: k=k0, i=0, so W[k0*2+0] = W[2*k0]
    # C[k0+1,1] has W index: k=k0+1, i=1, so W[(k0+1)*2+1] = W[2*k0+3]
    
    # For m1: coefficient in C[k0,0] is +1, in C[k0+1,1] is +1
    # So W[2*k0, r] = 1, W[2*k0+3, r] = 1
    
    # Let me build all 7 Strassen terms for a block starting at k0.
    
    def add_strassen_block(k0):
        # A indices: a00=0, a01=1, a10=3, a11=4
        # B indices: b00=k0, b01=k0+1, b10=7+k0, b11=7+k0+1
        # C indices: C[k0,0]=2*k0, C[k0+1,0]=2*k0+2, C[k0,1]=2*k0+1, C[k0+1,1]=2*k0+3
        
        # m1 = (a00+a11)*(b00+b11)
        #   U: a00=0, a11=4
        #   V: b00=k0, b11=7+k0+1
        #   C: c00 with +1, c11 with +1
        u = {0: 1, 4: 1}
        v = {k0: 1, 7 + k0 + 1: 1}
        w = {2 * k0: 1, 2 * k0 + 3: 1}
        add_strassen_term(u, v, w)
        
        # m2 = (a10+a11)*b00
        #   U: a10=3, a11=4
        #   V: b00=k0
        #   C: c10 with +1, c11 with -1
        u = {3: 1, 4: 1}
        v = {k0: 1}
        w = {2 * k0 + 1: 1, 2 * k0 + 3: -1}
        add_strassen_term(u, v, w)
        
        # m3 = a00*(b10+b11)
        #   U: a00=0
        #   V: b10=7+k0, b11=7+k0+1
        #   C: c01 with +1, c11 with +1
        u = {0: 1}
        v = {7 + k0: 1, 7 + k0 + 1: 1}
        w = {2 * k0 + 2: 1, 2 * k0 + 3: 1}
        add_strassen_term(u, v, w)
        
        # m4 = a11*(b01-b11)
        #   U: a11=4
        #   V: b01=k0+1, b11=7+k0+1 with -1
        #   C: c00 with +1, c10 with +1
        u = {4: 1}
        v = {k0 + 1: 1, 7 + k0 + 1: -1}
        w = {2 * k0: 1, 2 * k0 + 1: 1}
        add_strassen_term(u, v, w)
        
        # m5 = (a00+a01)*b11
        #   U: a00=0, a01=1
        #   V: b11=7+k0+1
        #   C: c00 with -1, c01 with +1
        u = {0: 1, 1: 1}
        v = {7 + k0 + 1: 1}
        w = {2 * k0: -1, 2 * k0 + 2: 1}
        add_strassen_term(u, v, w)
        
        # m6 = (a10-a00)*(b00+b01)
        #   U: a10=3, a00=0 with -1
        #   V: b00=k0, b01=k0+1
        #   C: c11 with +1
        u = {3: 1, 0: -1}
        v = {k0: 1, k0 + 1: 1}
        w = {2 * k0 + 3: 1}
        add_strassen_term(u, v, w)
        
        # m7 = (a01-a11)*(b10+b11)
        #   U: a01=1, a11=4 with -1
        #   V: b10=7+k0, b11=7+k0+1
        #   C: c00 with +1
        u = {1: 1, 4: -1}
        v = {7 + k0: 1, 7 + k0 + 1: 1}
        w = {2 * k0: 1}
        add_strassen_term(u, v, w)
    
    # Helper to add a Strassen term
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: {A_index: coeff}
        # v_coeffs: {B_index: coeff}
        # w_coeffs: {C_index: coeff}
        # A_index = i*m+j, B_index = j*p+k, C_index = k*n+i
        # We need to convert to (i,j) for U, (j,k) for V, (k,i) for W
        
        # For U: A_index = i*m+j, so i = A_index // m, j = A_index % m
        # For V: B_index = j*p+k, so j = B_index // p, k = B_index % p
        # For W: C_index = k*n+i, so k = C_index // n, i = C_index % n
        
        u_list = []
        for a_idx, coeff in u_coeffs.items():
            i = a_idx // m
            j = a_idx % m
            u_list.append((i * m + j, coeff))
        
        v_list = []
        for b_idx, coeff in v_coeffs.items():
            j = b_idx // p
            k = b_idx % p
            v_list.append((j * p + k, coeff))
        
        w_list = []
        for c_idx, coeff in w_coeffs.items():
            k = c_idx // n
            i = c_idx % n
            w_list.append((k * n + i, coeff))
        
        # Each term is a rank-1 tensor: U_r \otimes V_r \otimes W_r
        # U_r has non-zero entries at u_list positions
        # V_r has non-zero entries at v_list positions
        # W_r has non-zero entries at w_list positions
        terms.append((u_list, v_list, w_list))
    
    # Add Strassen blocks for k0 = 0, 2, 4
    for k0 in [0, 2, 4]:
        add_strassen_block(k0)
    
    # Schoolbook for k=6 (j in {0,1})
    # C[6,0] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    # C[6,1] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    
    # Term 1: A[0,0]*B[0,6] -> C[6,0]
    add_term(0, 0 * p + 6, 6 * n + 0, 1)  # U[0], V[6], W[12]
    # Term 2: A[0,1]*B[1,6] -> C[6,0]
    add_term(1, 1 * p + 6, 6 * n + 0, 1)  # U[1], V[13], W[12]
    # Term 3: A[1,0]*B[0,6] -> C[6,1]
    add_term(3, 0 * p + 6, 6 * n + 1, 1)  # U[3], V[6], W[13]
    # Term 4: A[1,1]*B[1,6] -> C[6,1]
    add_term(4, 1 * p + 6, 6 * n + 1, 1)  # U[4], V[13], W[13]
    
    # Now build U, V, W from terms
    # Count total rank
    # Part 1: n*p = 2*7 = 14 terms (each is a simple product)
    # Part 2: 3 Strassen blocks * 7 = 21 terms
    # Part 3: 4 schoolbook terms
    # Total: 14 + 21 + 4 = 39
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(terms):
        if isinstance(term[0], int):
            # Simple term: (u_idx, v_idx, w_idx, coeff)
            u_idx, v_idx, w_idx, coeff = term
            U[u_idx][r] = coeff
            V[v_idx][r] = coeff
            W[w_idx][r] = coeff
        else:
            # Strassen term: (u_list, v_list, w_list)
            u_list, v_list, w_list = term
            for u_idx, coeff in u_list:
                U[u_idx][r] += coeff
            for v_idx, coeff in v_list:
                V[v_idx][r] += coeff
            for w_idx, coeff in w_list:
                W[w_idx][r] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
