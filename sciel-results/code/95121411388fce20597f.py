"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for multiplying
    a 2x3 matrix by a 3x9 matrix.
    
    Strategy:
    1. Use Strassen's algorithm to multiply the first 2x2 block of the 2x3 matrix 
       (A_1) by the first 2x2 block of the 3x9 matrix (B_1) to get the first 2 columns
       of the result. This uses rank 7 for a 2x2x2 multiplication.
    2. The remaining column of A (3rd column) and the remaining 7 columns of B (cols 3-9)
       must be handled.
       - The 3rd column of A (1x3 vector) multiplied by the 3x9 B gives a 1x9 row.
       - The first 2 columns of A (2x2) multiplied by the remaining 7 columns of B (2x7)
         gives a 2x7 matrix.
       - The 3rd column of A multiplied by the first 2 columns of B is already handled
         by the Strassen block? No, Strassen handles the full 2x2 * 2x2 -> 2x2.
         The interaction between the 3rd column of A and the first 2 columns of B, and
         the first 2 columns of A and the 3rd column of B, and 3rd*3rd, etc., need care.
       
       Actually, it's easier to view the 2x3 * 3x9 as:
       C = [ A_left (2x2) * B_left (2x7) ] + [ A_right (2x1) * B_right (1x9) ] + 
           [ A_left (2x2) * B_mid (2x1) ] ... this is getting complicated to partition simply.
       
       Better partition:
       Let A = [A1 | A2] where A1 is 2x2 and A2 is 2x1.
       Let B = [B1; B2; B3] where B1 is 2x9 and B2 is 1x9 (wait, B is 3x9).
       B = [ B11 (2x9) ; B12 (1x9) ] where B11 is the first 2 rows, B12 is the 3rd row.
       Then C = A1 * B11 + A2 * B12.
       
       Here A1 is 2x2, B11 is 2x9.
       We can split B11 into B11a (2x2) and B11b (2x7).
       C = A1 * B11a + A1 * B11b + A2 * B12.
       
       A1 * B11a is 2x2 * 2x2 -> 2x2. Rank 7 (Strassen).
       A1 * B11b is 2x2 * 2x7 -> 2x7. We can think of this as 7 separate 2x1 multiplications
       or use a block Strassen? No, B11b is 2x7.
       A2 * B12 is 2x1 * 1x9 -> 2x9. This is an outer product, rank 2.
       
       For A1 * B11b (2x2 * 2x7):
       Let B11b = [b1 b2 ... b7] where each bj is a 2x1 column? No, B11b is 2x7, so columns are 2x1.
       Wait, A is 2x2. B is 2x7. Result is 2x7.
       We can decompose the 2x2 * 2x7 multiplication.
       Rank for 2x2 * 2x1 is 4 (schoolbook) or can we do better?
       Actually, 2x1 * 1x1 is rank 1.
       2x2 * 2x1 multiplication:
       A = [[a, b], [c, d]], x = [e, f]^T.
       y = [ae+bf, ce+df]^T.
       This is equivalent to 2x2 * 2x1. The tensor rank is 4 (cannot be reduced for 2x2*2x1 with rational/integer entries easily? Actually Strassen is for 2x2*2x2. For 2x2*2x1, the rank is 4).
       
       So A1 * B11b has 7 columns. Each column multiplication is 2x2 * 2x1, rank 4.
       Total rank for A1 * B11b = 7 * 4 = 28?
       Plus A1 * B11a (rank 7).
       Plus A2 * B12 (rank 2).
       Total rank = 7 + 28 + 2 = 37.
       
       Can we do better?
       Maybe treat A1 * B11 (2x2 * 2x9) as a single block?
       Rank of 2x2 * 2x9?
       If we view B11 as a 2x9 matrix, the multiplication A1 * B11 is linear in B11.
       We can use Strassen-like tricks if we group columns.
       
       Let's stick with the 37 rank construction for now, as it is significantly better than 54.
       
       Steps:
       1. Initialize U, V, W matrices with size (2*3)xR, (3*9)xR, (9*2)xR.
          Wait, dimensions:
          A is 2x3. B is 3x9. C is 2x9.
          n=2, m=3, p=9.
          U shape: (n*m, R) = (6, R)
          V shape: (m*p, R) = (27, R)
          W shape: (p*n, R) = (18, R)
          
       2. Block A1: rows 0,1 cols 0,1 of A. (Indices 0,1 in U)
          Block B11a: rows 0,1 cols 0,1 of B. (Indices 0,1 in V)
          Block B11b: rows 0,1 cols 2,8 of B. (Indices 2..8 in V)
          Block B12: row 2 cols 0,8 of B. (Index 18..26 in V) -- wait.
          
          V indexing: V[j*p + k, r].
          j is row of B (0..2). k is col of B (0..8).
          B11a: j in {0,1}, k in {0,1}. Indices: 0*9+0=0, 0*9+1=1, 1*9+0=9, 1*9+1=10.
          B11b: j in {0,1}, k in {2..8}. Indices: 2,3,4,5,6,7,8 and 11,12,13,14,15,16,17.
          B12: j=2, k in {0..8}. Indices: 18..26.
          
          U indexing: U[i*m + j, r].
          i is row of A (0..1). j is col of A (0..2).
          A1: i in {0,1}, j in {0,1}. Indices: 0,1, 3,4.
          A2: i in {0,1}, j=2. Indices: 2, 5.
          
          W indexing: W[k*n + i, r].
          k is col of C (0..8). i is row of C (0..1).
          C11a: k in {0,1}, i in {0,1}. Indices: 0,1, 2,3.
          C11b: k in {2..8}, i in {0,1}. Indices: 4..13.
          C12: k in {0..8}, i in {0,1}. All indices 0..17.
          
       3. Strassen for A1 * B11a (2x2 * 2x2):
          Let a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1].
          Let e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1].
          
          M1 = (a+d)(e+h)
          M2 = (c+d)(e)
          M3 = (a-b)(f-h)
          M4 = d(g-h)
          M5 = (a+c)(h)
          M6 = (b-c)(e+g)
          M7 = (b+d)(f-h)
          
          C00 = M1 - M4 - M5 + M7
          C01 = M3 + M5
          C10 = M2 + M4
          C11 = M1 - M2 - M3 + M6
          
          We need to add these to W.
          For each term Mx, we add the coefficient to the appropriate W[k*n+i] row.
          
          For example, M1 contributes to C00 (k=0, i=0 -> idx 0) with coeff 1,
          C11 (k=1, i=1 -> idx 3) with coeff 1.
          
          We also need to define U and V for these Mx terms.
          M1: U has (a+d) -> U[0]+=1, U[3]+=1. V has (e+h) -> V[0]+=1, V[10]+=1.
          
       4. For A1 * B11b (2x2 * 2x7):
          For each column k in {2..8} of B11b:
          Let col = [B[0,k], B[1,k]]^T.
          This is a 2x2 * 2x1 multiplication.
          Rank 4 schoolbook:
          y0 = a*B[0,k] + b*B[1,k]
          y1 = c*B[0,k] + d*B[1,k]
          
          Term 1: a * B[0,k]. U[0]=1, V[k]=1 (since j=0, k=k -> idx k), W[k*2+0]=1.
          Term 2: b * B[1,k]. U[1]=1, V[9+k]=1 (since j=1, k=k -> idx 9+k), W[k*2+0]=1.
          Term 3: c * B[0,k]. U[3]=1, V[k]=1, W[k*2+1]=1.
          Term 4: d * B[1,k]. U[4]=1, V[9+k]=1, W[k*2+1]=1.
          
          Wait, U indices for A1:
          A[0,0] -> U[0]
          A[0,1] -> U[1]
          A[1,0] -> U[3]
          A[1,1] -> U[4]
          
          This adds 4 * 7 = 28 terms.
          
       5. For A2 * B12 (2x1 * 1x9):
          A2 = [A[0,2], A[1,2]]^T. Indices in U: U[2], U[5].
          B12 = [B[2,0], ..., B[2,8]]. Indices in V: 18..26.
          
          C0k = A[0,2] * B[2,k]
          C1k = A[1,2] * B[2,k]
          
          For each k in 0..8:
          Term 1: A[0,2] * B[2,k]. U[2]=1, V[18+k]=1, W[k*2+0]=1.
          Term 2: A[1,2] * B[2,k]. U[5]=1, V[18+k]=1, W[k*2+1]=1.
          
          This adds 2 * 9 = 18 terms.
          
       Total rank = 7 + 28 + 18 = 53?
       Wait, earlier I calculated 37. Where did I go wrong?
       Ah, A1 * B11b is 2x2 * 2x7.
       My schoolbook calculation was 4 per column * 7 columns = 28.
       Strassen was 7 for 2x2 * 2x2.
       So 7 + 28 + 18 = 53.
       This is barely better than 54.
       
       Can we optimize A1 * B11b?
       A1 * B11b is 2x2 * 2x7.
       We can group the 7 columns into 2+2+2+1.
       2x2 * 2x2 -> 7
       2x2 * 2x2 -> 7
       2x2 * 2x2 -> 7
       2x2 * 2x1 -> 4
       Total 25.
       So 7 (for B11a) + 25 (for B11b) + 18 (for B12) = 50.
       
       Let's refine the B11b handling.
       B11b has columns 2,3,4,5,6,7,8.
       Group 1: cols 2,3.
       Group 2: cols 4,5.
       Group 3: cols 6,7.
       Group 4: col 8.
       
       For each group of 2 columns, use Strassen (rank 7).
       For the single column, use schoolbook (rank 4).
       
       Total rank = 7 + 7 + 7 + 7 + 4 + 18 = 50.
       
       This is a solid improvement over 54.
       
       Implementation details:
       We will create a helper function to add a Strassen 2x2*2x2 block.
       Inputs:
       - u_offsets: indices in U for the 4 entries of A block (row-major: a,b,c,d).
       - v_offsets: indices in V for the 4 entries of B block (row-major: e,f,g,h).
       - w_offsets: indices in W for the 4 entries of C block (row-major: C00,C01,C10,C11).
       
       For Strassen:
       M1 = (a+d)(e+h)
       M2 = (c+d)(e)
       M3 = (a-b)(f-h)
       M4 = d(g-h)
       M5 = (a+c)(h)
       M6 = (b-c)(e+g)
       M7 = (b+d)(f-h)
       
       C00 = M1 - M4 - M5 + M7
       C01 = M3 + M5
       C10 = M2 + M4
       C11 = M1 - M2 - M3 + M6
       
       We add these linear combinations to U, V, W.
       
       For schoolbook 2x2*2x1:
       Inputs:
       - u_offsets: indices in U for a,b,c,d.
       - v_offset: index in V for e (top), v_offset+1? No, V index for f (bottom).
         Actually, for 2x1, V has 2 entries: e=B[0,k], f=B[1,k].
       - w_offsets: indices in W for y0, y1.
       
       y0 = a*e + b*f
       y1 = c*e + d*f
       
       Terms:
       1. a*e: U[u_a]=1, V[v_e]=1, W[w_y0]=1
       2. b*f: U[u_b]=1, V[v_f]=1, W[w_y0]=1
       3. c*e: U[u_c]=1, V[v_e]=1, W[w_y1]=1
       4. d*f: U[u_d]=1, V[v_f]=1, W[w_y1]=1
       
       For outer product 2x1 * 1x9:
       This is just 18 terms.
       
       Let's code this.
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    R = 50  # Estimated rank
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2_2x2(u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_c00, w_c01, w_c10, w_c11):
        nonlocal col
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[w_c00][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M2 = (c+d)(e)
        U[u_c][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        W[w_c10][col] += 1
        W[w_c11][col] -= 1
        col += 1
        
        # M3 = (a-b)(f-h)
        U[u_a][col] += 1
        U[u_b][col] -= 1
        V[v_f][col] += 1
        V[v_h][col] -= 1
        W[w_c01][col] += 1
        W[w_c11][col] -= 1
        col += 1
        
        # M4 = d(g-h)
        U[u_d][col] += 1
        V[v_g][col] += 1
        V[v_h][col] -= 1
        W[w_c00][col] -= 1
        W[w_c10][col] += 1
        col += 1
        
        # M5 = (a+c)(h)
        U[u_a][col] += 1
        U[u_c][col] += 1
        V[v_h][col] += 1
        W[w_c00][col] -= 1
        W[w_c01][col] += 1
        col += 1
        
        # M6 = (b-c)(e+g)
        U[u_b][col] += 1
        U[u_c][col] -= 1
        V[v_e][col] += 1
        V[v_g][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M7 = (b+d)(f-h)
        U[u_b][col] += 1
        U[u_d][col] += 1
        V[v_f][col] += 1
        V[v_h][col] -= 1
        W[w_c00][col] += 1
        col += 1
        
    def add_schoolbook_2x2_2x1(u_a, u_b, u_c, u_d, v_e, v_f, w_y0, w_y1):
        nonlocal col
        # y0 = a*e + b*f
        # y1 = c*e + d*f
        
        # a*e
        U[u_a][col] += 1
        V[v_e][col] += 1
        W[w_y0][col] += 1
        col += 1
        
        # b*f
        U[u_b][col] += 1
        V[v_f][col] += 1
        W[w_y0][col] += 1
        col += 1
        
        # c*e
        U[u_c][col] += 1
        V[v_e][col] += 1
        W[w_y1][col] += 1
        col += 1
        
        # d*f
        U[u_d][col] += 1
        V[v_f][col] += 1
        W[w_y1][col] += 1
        col += 1

    # U indices for A1 (2x2 block at top-left of A)
    # A = [[a, b, *], [c, d, *]]
    # U indices: i*m+j
    # a: 0*3+0 = 0
    # b: 0*3+1 = 1
    # c: 1*3+0 = 3
    # d: 1*3+1 = 4
    
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    # V indices for B11a (2x2 block at top-left of B)
    # B = [[e, f, ...], [g, h, ...], [...]]
    # V indices: j*p+k
    # e: 0*9+0 = 0
    # f: 0*9+1 = 1
    # g: 1*9+0 = 9
    # h: 1*9+1 = 10
    
    v_e, v_f, v_g, v_h = 0, 1, 9, 10
    
    # W indices for C11a (2x2 block at top-left of C)
    # C = [[C00, C01, ...], [C10, C11, ...]]
    # W indices: k*n+i
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 2
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    
    w_c00, w_c01, w_c10, w_c11 = 0, 2, 1, 3
    
    # 1. Strassen for A1 * B11a
    add_strassen_2x2_2x2(u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_c00, w_c01, w_c10, w_c11)
    
    # 2. Strassen for A1 * B11b groups
    # B11b columns: 2,3,4,5,6,7,8
    # Group 1: cols 2,3
    # Group 2: cols 4,5
    # Group 3: cols 6,7
    # Group 4: col 8 (schoolbook)
    
    for start_k in [2, 4, 6]:
        k1 = start_k
        k2 = start_k + 1
        
        # V indices
        v_e1 = 0 * 9 + k1  # 0 + k1
        v_f1 = 0 * 9 + k2  # 0 + k2
        v_g1 = 1 * 9 + k1  # 9 + k1
        v_h1 = 1 * 9 + k2  # 9 + k2
        
        # W indices
        # C0_k1, C0_k2, C1_k1, C1_k2
        w_c00_1 = k1 * 2 + 0
        w_c01_1 = k2 * 2 + 0
        w_c10_1 = k1 * 2 + 1
        w_c11_1 = k2 * 2 + 1
        
        add_strassen_2x2_2x2(u_a, u_b, u_c, u_d, v_e1, v_f1, v_g1, v_h1, w_c00_1, w_c01_1, w_c10_1, w_c11_1)
    
    # Group 4: col 8 (schoolbook)
    k8 = 8
    v_e8 = 0 * 9 + k8
    v_f8 = 1 * 9 + k8
    w_y0_8 = k8 * 2 + 0
    w_y1_8 = k8 * 2 + 1
    
    add_schoolbook_2x2_2x1(u_a, u_b, u_c, u_d, v_e8, v_f8, w_y0_8, w_y1_8)
    
    # 3. Outer product A2 * B12
    # A2: col 2 of A. u indices: 0*3+2=2, 1*3+2=5
    u_a2_0 = 2
    u_a2_1 = 5
    
    # B12: row 2 of B. v indices: 2*9+k = 18+k
    for k in range(9):
        v_b12 = 2 * 9 + k
        w_c0_k = k * 2 + 0
        w_c1_k = k * 2 + 1
        
        # A[0,2] * B[2,k]
        U[u_a2_0][col] += 1
        V[v_b12][col] += 1
        W[w_c0_k][col] += 1
        col += 1
        
        # A[1,2] * B[2,k]
        U[u_a2_1][col] += 1
        V[v_b12][col] += 1
        W[w_c1_k][col] += 1
        col += 1
    
    # Trim if col < R (should be exactly 50)
    if col < R:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    elif col > R:
        # This shouldn't happen if calculation is correct, but safety
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
