"""Rewritten program for 4x4 * 4x6 matrix multiplication tensor decomposition.

Strategy:
1. Start from the known Strassen-based decomposition (rank 77) built by tiling
   the 4x4 * 4x6 product into four 2x2 * 2x3 blocks. Each 2x2*2x3 block uses
   inner Strassen 2x2*2x2 (rank 7) for the first two columns of the right factor
   plus schoolbook 2x2*2x1 (rank 4) for the third column -> rank 11 per block,
   total 4*11 = 44... wait, that's not right. Let me recompute.

Actually, 4x4 * 4x6:
- Split into blocks: A is 4x4, B is 4x6.
- We can split A into 2x2 blocks (4 blocks) and B into 2x3 blocks (4 blocks).
- Each output 2x3 block = A_block (2x2) * B_block (2x3).
- For 2x2 * 2x3: split B_block into B1 (2x2) and B2 (2x1).
  - A*B1 uses Strassen rank 7.
  - A*B2 uses schoolbook rank 4.
  - Total per block: 7+4 = 11.
- 4 blocks * 11 = 44.

Wait, that's much better than 77. Let me verify: the knowledge says "reducing total from 84 to 77". 
Let me re-read: "Replacing schoolbook 2x2x3 (rank 12) per Strassen term with inner Strassen 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4) = rank 11 per term, reducing total from 84 to 77"

So 84/12 = 7 terms? That doesn't match 4 blocks. Let me think again.

Actually for 4x4 * 4x6:
- n=4, m=4, p=6
- Schoolbook: 4*4*6 = 96
- The "84" might come from a different tiling. Let me think: if we use Strassen on the 4x4 part...

Actually, let me just build the 44-rank decomposition directly and verify it.

Let me construct it carefully:

A is 4x4, B is 4x6. Output C is 4x6.

Split A into 2x2 blocks:
  A11 A12
  A21 A22
where each is 2x2.

Split B into 2x3 blocks:
  B11 B12
  B21 B22
where each is 2x3.

C11 = A11*B11 + A12*B21  (2x3)
C12 = A11*B12 + A12*B22  (2x3)
C21 = A21*B11 + A22*B21  (2x3)
C22 = A21*B12 + A22*B22  (2x3)

Each Ci,j is a 2x3 matrix = (2x2)*(2x3) + (2x2)*(2x3).

For a single (2x2)*(2x3) product: let X be 2x2, Y be 2x3 = [Y1 | Y2] where Y1 is 2x2, Y2 is 2x1.
X*Y = X*Y1 + X*Y2.
- X*Y1: Strassen, rank 7.
- X*Y2: schoolbook, rank 4 (2*2*1=4).
Total: 11.

So each Ci,j requires 2 * 11 = 22 rank terms? No wait, Ci,j = A*Bi,j where A is one 2x2 block and Bi,j is one 2x3 block. So each Ci,j = one (2x2)*(2x3) product = 11 rank terms.

Total: 4 blocks * 11 = 44.

Let me verify this is correct by actually constructing it.
"""

import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6  # A is n x m, B is m x p, C is n x p
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Convention: W[k*n + i, r] where i is row of C, k is col of C
    
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    # Start with empty lists, we'll append columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add one rank-1 term. u_vec length 16, v_vec length 24, w_vec length 24."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def zeros(length):
        return [0] * length
    
    # Helper: Strassen 2x2*2x2 decomposition
    # Inputs: A (2x2), B (2x2). Output: C = A*B (2x2).
    # Strassen: 
    #   M1 = (A11+A22)*(B11+B22)
    #   M2 = (A21+A22)*B11
    #   M3 = A11*(B12-B22)
    #   M4 = A22*(B21-B11)
    #   M5 = (A11+A12)*B22
    #   M6 = (A21-A11)*(B11+B12)
    #   M7 = (A12-A22)*(B21+B22)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # For tensor decomposition, each Mj contributes one rank-1 term.
    # The U, V, W vectors encode which entries of A, B, C are involved.
    #
    # A indices: A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    #   In U: index = i*m + j where i is row in A (0..1 local), j is col in A (0..1 local)
    #   But we need global indices. Let's use offsets.
    #
    # B indices: B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    #   In V: index = j*p + k where j is row in B (0..1 local), k is col in B (0..1 local)
    #
    # C indices: C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    #   In W: index = k*n + i where k is col in C (0..1 local), i is row in C (0..1 local)
    #
    # For Strassen term M1 = (A11+A22)*(B11+B22):
    #   U: A11 + A22 -> u[A11]=1, u[A22]=1
    #   V: B11 + B22 -> v[B11]=1, v[B22]=1
    #   W: contributes to C11 (+1), C22 (+1) -> w[C11]=1, w[C22]=1
    #
    # Actually for the tensor decomposition, each rank-1 term is a product of:
    #   (linear form in A) * (linear form in B) * (linear form in C)
    # and the sum over all terms equals the multiplication tensor.
    #
    # For M1 = (A11+A22)*(B11+B22), this single product contributes:
    #   A11*B11 to C11 and C22
    #   A11*B22 to C11 and C22
    #   A22*B11 to C11 and C22
    #   A22*B22 to C11 and C22
    # But in the full Strassen formula, M1 appears in C11 with +1 and C22 with +1.
    # So the rank-1 term for M1 has:
    #   U: e_A11 + e_A22
    #   V: e_B11 + e_B22
    #   W: e_C11 + e_C22
    # And the contribution is (A11+A22)(B11+B22) * (coefficient for C11) for C11 part, etc.
    #
    # Wait, I need to be more careful. The tensor T = sum_{i,j,k} e_i^A * e_j^B * e_k^C
    # where e_i^A picks out A[i,j], etc. The decomposition is T = sum_r u_r * v_r * w_r
    # where u_r is in A-space, v_r in B-space, w_r in C-space.
    #
    # For Strassen, each Mj is a bilinear form in A and B, and it contributes linearly
    # to entries of C. So each Mj gives one rank-1 term where:
    #   u_r = the linear form in A for Mj
    #   v_r = the linear form in B for Mj  
    #   w_r = the linear form in C (sum of basis vectors for each C entry Mj contributes to, with signs)
    
    # Let me define local index helpers
    # A local indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B local indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C local indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    def strassen_2x2_2x2(a_off, b_off, c_off):
        """Add 7 rank-1 terms for Strassen 2x2*2x2.
        a_off: (row_off, col_off) offset in A for the 2x2 block
        b_off: (row_off, col_off) offset in B for the 2x2 block  
        c_off: (row_off, col_off) offset in C for the 2x2 block
        """
        ar, ac = a_off
        br, bc = b_off
        cr, cc = c_off
        
        # Local to global index functions
        def a_idx(i, j): return (ar+i)*m + (ac+j)  # A is 4x4, m=4
        def b_idx(i, j): return (br+i)*p + (bc+j)  # B is 4x6, p=6
        def c_idx(i, j): return (cc+j)*n + (cr+i)  # C is 4x6, W index = k*n+i
        
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,0)] = 1; u[a_idx(1,1)] = 1
        v[b_idx(0,0)] = 1; v[b_idx(1,1)] = 1
        w[c_idx(0,0)] = 1; w[c_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,0)] = 1; u[a_idx(1,1)] = 1
        v[b_idx(0,0)] = 1
        w[c_idx(1,0)] = 1; w[c_idx(1,1)] = -1
        add_term(u, v, w)
        
        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,0)] = 1
        v[b_idx(0,1)] = 1; v[b_idx(1,1)] = -1
        w[c_idx(0,1)] = 1; w[c_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,1)] = 1
        v[b_idx(1,0)] = 1; v[b_idx(0,0)] = -1
        w[c_idx(0,0)] = 1; w[c_idx(1,0)] = 1
        add_term(u, v, w)
        
        # M5 = (A11+A12)*B22 -> C11 -= M5, C12 += M5
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,0)] = 1; u[a_idx(0,1)] = 1
        v[b_idx(1,1)] = 1
        w[c_idx(0,0)] = -1; w[c_idx(0,1)] = 1
        add_term(u, v, w)
        
        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,0)] = 1; u[a_idx(0,0)] = -1
        v[b_idx(0,0)] = 1; v[b_idx(0,1)] = 1
        w[c_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,1)] = 1; u[a_idx(1,1)] = -1
        v[b_idx(1,0)] = 1; v[b_idx(1,1)] = 1
        w[c_idx(0,0)] = 1
        add_term(u, v, w)
    
    def schoolbook_2x2_2x1(a_off, b_off, c_off):
        """Add 4 rank-1 terms for schoolbook 2x2*2x1.
        A is 2x2, B is 2x1, C is 2x1.
        C1 = A11*B1 + A12*B2
        C2 = A21*B1 + A22*B2
        """
        ar, ac = a_off
        br, bc = b_off
        cr, cc = c_off
        
        def a_idx(i, j): return (ar+i)*m + (ac+j)
        def b_idx(i, j): return (br+i)*p + (bc+j)
        def c_idx(i, j): return (cc+j)*n + (cr+i)
        
        # C1 += A11*B1
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,0)] = 1
        v[b_idx(0,0)] = 1
        w[c_idx(0,0)] = 1
        add_term(u, v, w)
        
        # C1 += A12*B2
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,1)] = 1
        v[b_idx(1,0)] = 1
        w[c_idx(0,0)] = 1
        add_term(u, v, w)
        
        # C2 += A21*B1
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,0)] = 1
        v[b_idx(0,0)] = 1
        w[c_idx(1,0)] = 1
        add_term(u, v, w)
        
        # C2 += A22*B2
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,1)] = 1
        v[b_idx(1,0)] = 1
        w[c_idx(1,0)] = 1
        add_term(u, v, w)
    
    # Now build the full 4x4 * 4x6 decomposition
    # A blocks: A11 at (0,0), A12 at (0,2), A21 at (2,0), A22 at (2,2)
    # B blocks: B11 at (0,0), B12 at (0,3), B21 at (2,0), B22 at (2,3)
    #   Each B block is 2x3, so B11 covers rows 0-1, cols 0-2; B12 covers rows 0-1, cols 3-5
    #   B21 covers rows 2-3, cols 0-2; B22 covers rows 2-3, cols 3-5
    # C blocks: C11 at (0,0), C12 at (0,3), C21 at (2,0), C22 at (2,3)
    #   Each C block is 2x3
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For each (A_block * B_block) where A_block is 2x2 and B_block is 2x3:
    # Split B_block into B_left (2x2) and B_right (2x1)
    # A*B_left: Strassen 2x2*2x2 -> 7 terms
    # A*B_right: schoolbook 2x2*2x1 -> 4 terms
    # Total: 11 terms per product
    
    def product_2x2_2x3(a_off, b_off, c_off):
        """Compute A(2x2) * B(2x3) = C(2x3).
        Split B into B_left(2x2) at cols 0-1 and B_right(2x1) at col 2.
        C_left(2x2) = A * B_left via Strassen
        C_right(2x1) = A * B_right via schoolbook
        """
        ar, ac = a_off
        br, bc = b_off
        cr, cc = c_off
        
        # B_left is 2x2 at (br, bc)
        strassen_2x2_2x2(a_off, (br, bc), (cr, cc))
        
        # B_right is 2x1 at (br, bc+2)
        schoolbook_2x2_2x1(a_off, (br, bc+2), (cr, cc+2))
    
    # C11 = A11*B11 + A12*B21
    product_2x2_2x3((0,0), (0,0), (0,0))   # A11 * B11
    product_2x2_2x3((0,2), (2,0), (0,0))   # A12 * B21
    
    # C12 = A11*B12 + A12*B22
    product_2x2_2x3((0,0), (0,3), (0,3))   # A11 * B12
    product_2x2_2x3((0,2), (2,3), (0,3))   # A12 * B22
    
    # C21 = A21*B11 + A22*B21
    product_2x2_2x3((2,0), (0,0), (2,0))   # A21 * B11
    product_2x2_2x3((2,2), (2,0), (2,0))   # A22 * B21
    
    # C22 = A21*B12 + A22*B22
    product_2x2_2x3((2,0), (0,3), (2,3))   # A21 * B12
    product_2x2_2x3((2,2), (2,3), (2,3))   # A22 * B22
    
    rank = len(U_cols)
    
    # Convert column lists to row-major matrices
    U = [[U_cols[r][i] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(V_rows)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(W_rows)]
    
    return {"U": U, "V": V, "W": W}
