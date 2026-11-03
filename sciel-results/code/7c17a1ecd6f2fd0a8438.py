"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    # Rank 49 for 4x4x4 Strassen + 16 for 4x4x1 schoolbook = 65
    rank = 65
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Part 1: Strassen 4x4x4
    # Strassen 4x4x4 uses 7 multiplications of 2x2x2.
    # Each 2x2x2 uses Strassen base case (rank 7).
    # Total 7 * 7 = 49 columns.
    
    # Strassen 2x2x2 base case definitions
    # U rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W rows: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Note: W index is k*n+i, but for 2x2x2 local, it's k*2+i.
    
    # Strassen 2x2x2 factors (U, V, W)
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B01-B11) -> C01 += M3
    # M4 = A11(B10-B00) -> C11 += M4
    # M5 = (A00+A10)B11 -> C10 += M5
    # M6 = (A11-A10)(B00+B10) -> C10 += M6
    # M7 = (A01-A11)(B11+B10) -> C01 += M7
    
    # Let's define the 7 Strassen 2x2x2 decompositions explicitly.
    # Each is a set of 7 scalar multiplications.
    
    # Helper to add a 2x2x2 Strassen block to global U, V, W
    # u_rows: mapping from local row (0..3) to global row in U
    # v_rows: mapping from local row (0..3) to global row in V
    # w_rows: mapping from local row (0..3) to global row in W (local k*2+i)
    # col_start: starting column index
    def add_strassen_2x2(u_map, v_map, w_map, col_start, sign_u=1, sign_v=1, sign_w=1):
        # 7 multiplications
        # 1: A00+A11, B00+B11 -> C00, C11
        # 2: A10+A11, B00 -> C10
        # 3: A00, B01-B11 -> C01
        # 4: A11, B10-B00 -> C11
        # 5: A00+A10, B11 -> C10
        # 6: A11-A10, B00+B10 -> C10
        # 7: A01-A11, B11+B10 -> C01
        
        # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # Mult 1
        # U: rows 0, 3 have +1
        U[u_map[0]][col_start] += sign_u
        U[u_map[3]][col_start] += sign_u
        # V: rows 0, 3 have +1
        V[v_map[0]][col_start] += sign_v
        V[v_map[3]][col_start] += sign_v
        # W: rows 0, 3 have +1 (C00, C11)
        W[w_map[0]][col_start] += sign_w
        W[w_map[3]][col_start] += sign_w
        
        # Mult 2
        # U: rows 2, 3 have +1
        U[u_map[2]][col_start+1] += sign_u
        U[u_map[3]][col_start+1] += sign_u
        # V: row 0 has +1
        V[v_map[0]][col_start+1] += sign_v
        # W: row 2 has +1 (C10)
        W[w_map[2]][col_start+1] += sign_w
        
        # Mult 3
        # U: row 0 has +1
        U[u_map[0]][col_start+2] += sign_u
        # V: rows 1, 3 have +1, -1
        V[v_map[1]][col_start+2] += sign_v
        V[v_map[3]][col_start+2] -= sign_v
        # W: row 1 has +1 (C01)
        W[w_map[1]][col_start+2] += sign_w
        
        # Mult 4
        # U: row 3 has +1
        U[u_map[3]][col_start+3] += sign_u
        # V: rows 2, 0 have +1, -1
        V[v_map[2]][col_start+3] += sign_v
        V[v_map[0]][col_start+3] -= sign_v
        # W: row 3 has +1 (C11)
        W[w_map[3]][col_start+3] += sign_w
        
        # Mult 5
        # U: rows 0, 2 have +1
        U[u_map[0]][col_start+4] += sign_u
        U[u_map[2]][col_start+4] += sign_u
        # V: row 3 has +1
        V[v_map[3]][col_start+4] += sign_v
        # W: row 2 has +1 (C10)
        W[w_map[2]][col_start+4] += sign_w
        
        # Mult 6
        # U: rows 3, 2 have +1, -1
        U[u_map[3]][col_start+5] += sign_u
        U[u_map[2]][col_start+5] -= sign_u
        # V: rows 0, 2 have +1
        V[v_map[0]][col_start+5] += sign_v
        V[v_map[2]][col_start+5] += sign_v
        # W: row 2 has +1 (C10)
        W[w_map[2]][col_start+5] += sign_w
        
        # Mult 7
        # U: rows 1, 3 have +1, -1
        U[u_map[1]][col_start+6] += sign_u
        U[u_map[3]][col_start+6] -= sign_u
        # V: rows 3, 2 have +1
        V[v_map[3]][col_start+6] += sign_v
        V[v_map[2]][col_start+6] += sign_v
        # W: row 1 has +1 (C01)
        W[w_map[1]][col_start+6] += sign_w

    # 4x4x4 Strassen decomposition
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B11+B10)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M4
    
    # We need to map the 2x2 blocks to global indices.
    # A is 4x4. A00 is rows 0-1, cols 0-1. A01 is rows 0-1, cols 2-3.
    # A10 is rows 2-3, cols 0-1. A11 is rows 2-3, cols 2-3.
    # Global U row for A[I][J] where I in {0,1} (block row), J in {0,1} (block col)
    # Local (i,j) in 2x2. Global row = I*2 + i, J*2 + j? No, U is flattened row-major.
    # U index = (I*2+i)*4 + (J*2+j).
    
    # Let's define the maps for each of the 7 M terms.
    # M1: (A00+A11)(B00+B11)
    # A part: A00 and A11.
    # A00 global rows: 0,1,4,5 (indices 0,1, 4,5). Local (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    # A11 global rows: 8,9,12,13.
    # So for M1, U has entries in A00 and A11.
    
    # This is getting complex to hardcode manually. Let's use a programmatic approach.
    
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 4x4x4
    # 7 terms
    # Term 1: (A00+A11)(B00+B11)
    # Term 2: (A10+A11)B00
    # Term 3: A00(B01-B11)
    # Term 4: A11(B10-B00)
    # Term 5: (A00+A10)B11
    # Term 6: (A11-A10)(B00+B10)
    # Term 7: (A01-A11)(B11+B10)
    
    # We will build the U, V, W vectors for each term.
    # Each term is a product of two 2x2 matrices.
    # We can use the 2x2x2 Strassen decomposition for each.
    
    # Let's define a function to get the global row index for a 2x2 block in A or B.
    # A is 4x4. Block (bi, bj) where bi, bj in {0,1}.
    # Local (i,j) in 0..1.
    # Global row = (bi*2+i)*4 + (bj*2+j)
    
    def a_row(bi, bj, i, j):
        return (bi*2+i)*4 + (bj*2+j)
        
    def b_row(bi, bj, i, j):
        # B is 4x5. But for 4x4 part, we only use first 4 cols.
        # Block (bi, bj) where bi, bj in {0,1}.
        # Local (i,j) in 0..1.
        # Global row = (bi*2+i)*5 + (bj*2+j)
        return (bi*2+i)*5 + (bj*2+j)
        
    def c_row(bi, bj, i, j):
        # C is 5x4? No, C is 4x5.
        # Trace convention: W[k*n+i, r]. n=4.
        # Row index = k*4 + i.
        # C is product of A(4x4) and B(4x5). C is 4x5.
        # Block (bi, bj) where bi, bj in {0,1} for the 4x4 part.
        # Local (i,j) in 0..1.
        # Global row = (bi*2+i)*4 + (bj*2+j)? No.
        # W row index is k*n + i. Here k is row of C (0..3), i is col of C (0..4)?
        # Wait, the problem says "output factor follows the official trace convention W[k*n+i,r]".
        # And W shape is (p*n, rank). p=5, n=4. So 20 rows.
        # Usually, tensor M_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        # Here A is n x m, B is m x p. C is n x p.
        # M_{i,k,j} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}?
        # Let's check the seed code.
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
        # Product: sum_j U[i,j] * V[j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        # This matches C[i,k] = sum_j A[i,j] B[j,k].
        # So W row index is k*n + i. k is col of B / row of C? No.
        # In C[i,k], i is row, k is col.
        # So W row index = k*4 + i.
        
        # For the 4x4 part of C (cols 0..3):
        # Block (bi, bj) where bi is row block, bj is col block.
        # Local (i,j) in 0..1.
        # Global row of C = bi*2+i, col of C = bj*2+j.
        # W row index = (col)*4 + (row) = (bj*2+j)*4 + (bi*2+i).
        
        return (bj*2+j)*4 + (bi*2+i)

    # Now, for each of the 7 terms, we need to define the U, V, W contributions.
    # We will use the 2x2x2 Strassen decomposition.
    # The 2x2x2 Strassen decomposition has 7 columns.
    # We will iterate over the 7 terms, and for each, iterate over the 7 Strassen 2x2x2 columns.
    
    # Term 1: (A00+A11)(B00+B11)
    # U involves A00 and A11.
    # V involves B00 and B11.
    # W involves C00 and C11 (from the 4x4 C).
    
    # We will define the mapping for each term.
    # For term t, we define lists of (global_row, coeff) for U, V, W.
    # But it's easier to just call add_strassen_2x2 with the correct maps.
    
    # Let's define the maps for each term.
    # Term 1: (A00+A11)(B00+B11)
    # A part: A00 and A11.
    # In Strassen 2x2, U has rows 0,1,2,3 corresponding to A00,A01,A10,A11 of the 2x2 matrix.
    # Here, the "2x2 matrix" for A is formed by A00 and A11.
    # This is not a simple block. It's a linear combination.
    # Actually, the 4x4 Strassen algorithm defines 7 new matrices M1..M7.
    # M1 = (A00+A11)(B00+B11).
    # This is a product of two 2x2 matrices.
    # Let X = A00+A11, Y = B00+B11.
    # X and Y are 2x2 matrices.
    # We can decompose X*Y using 2x2x2 Strassen.
    # The U for this decomposition will be linear combinations of rows of X.
    # X = A00+A11. So U will have entries from A00 and A11.
    
    # This is correct. We just need to define the U, V, W maps for X, Y, Z (where Z is the result block in C).
    
    # Let's define a helper to build the maps.
    # For a term like (A00+A11)(B00+B11) -> C00 + C11 (partially)
    # Actually, in 4x4 Strassen:
    # C00 = M1 - M4 + M6
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M4
    
    # So each M contributes to specific C blocks.
    
    # We will create 7 "super-columns" groups.
    # For each M_t, we run 2x2x2 Strassen.
    # The U for M_t is derived from the A blocks involved.
    # The V for M_t is derived from the B blocks involved.
    # The W for M_t is derived from the C blocks involved.
    
    # Let's implement this.
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # A blocks: A00, A11.
    # B blocks: B00, B11.
    # C blocks: C00, C11.
    
    # We need to map the local 2x2 indices to the global indices for these blocks.
    # For A00+A11, the local 2x2 matrix X has:
    # X[0,0] = A00[0,0] + A11[0,0]
    # X[0,1] = A00[0,1] + A11[0,1]
    # X[1,0] = A00[1,0] + A11[1,0]
    # X[1,1] = A00[1,1] + A11[1,1]
    
    # So for a local index (i,j) in X, the global rows are a_row(0,0,i,j) and a_row(1,1,i,j).
    # Both have coeff +1.
    
    def get_u_map(blocks_a):
        # blocks_a is a list of (bi, bj, coeff)
        # Returns a function or dict mapping local (i,j) to list of (global_row, coeff)
        pass # We will just build the vectors directly in the loop.
        
    # Let's just iterate.
    
    # Term 1: M1
    # A: A00(+1), A11(+1)
    # B: B00(+1), B11(+1)
    # C: C00(+1), C11(+1)
    
    col = 0
    # M1
    u_maps = { 
        0: [(a_row(0,0,0,0), 1), (a_row(1,1,0,0), 1)],
        1: [(a_row(0,0,0,1), 1), (a_row(1,1,0,1), 1)],
        2: [(a_row(0,0,1,0), 1), (a_row(1,1,1,0), 1)],
        3: [(a_row(0,0,1,1), 1), (a_row(1,1,1,1), 1)]
    }
    v_maps = {
        0: [(b_row(0,0,0,0), 1), (b_row(1,1,0,0), 1)],
        1: [(b_row(0,0,0,1), 1), (b_row(1,1,0,1), 1)],
        2: [(b_row(0,0,1,0), 1), (b_row(1,1,1,0), 1)],
        3: [(b_row(0,0,1,1), 1), (b_row(1,1,1,1), 1)]
    }
    w_maps = {
        0: [(c_row(0,0,0,0), 1), (c_row(1,1,0,0), 1)], # C00 local(0,0), C11 local(0,0)
        1: [(c_row(0,0,0,1), 1), (c_row(1,1,0,1), 1)], # C00 local(0,1), C11 local(0,1)
        2: [(c_row(0,1,0,0), 1), (c_row(1,1,1,0), 1)], # C00 local(1,0), C11 local(1,0)
        3: [(c_row(0,1,0,1), 1), (c_row(1,1,1,1), 1)]  # C00 local(1,1), C11 local(1,1)
    }
    # Wait, the w_maps local index (i,j) in 2x2 C block corresponds to local row i, col j.
    # In add_strassen_2x2, w_map[0] is C00, w_map[1] is C01, w_map[2] is C10, w_map[3] is C11.
    # This is for the 2x2 matrix C.
    # Here, the 2x2 matrix C is formed by C00 and C11? No.
    # The product M1 is a 2x2 matrix. Let's call it Z1.
    # Z1 contributes to C00 and C11? No, Z1 is just a 2x2 matrix.
    # In the 4x4 Strassen, C00 = M1 - M4 + M6.
    # So M1 is a 2x2 matrix that is added to C00.
    # So the "C" part of the 2x2x2 decomposition is just the 2x2 matrix Z1.
    # And Z1 is placed in the C00 block of the 4x4 C.
    # So the w_map should map local (i,j) to global row for C00[i,j].
    # So w_map[0] = c_row(0,0,0,0), w_map[1] = c_row(0,0,0,1), etc.
    
    # Let's redefine w_maps for each term correctly.
    
    # Term 1: M1 -> C00
    w_maps = {
        0: [(c_row(0,0,0,0), 1)],
        1: [(c_row(0,0,0,1), 1)],
        2: [(c_row(0,0,1,0), 1)],
        3: [(c_row(0,0,1,1), 1)]
    }
    add_strassen_2x2(u_maps, v_maps, w_maps, col)
    col += 7
    
    # Term 2: M2 = (A10+A11)B00 -> C10
    u_maps = { 
        0: [(a_row(1,0,0,0), 1), (a_row(1,1,0,0), 1)],
        1: [(a_row(1,0,0,1), 1), (a_row(1,1,0,1), 1)],
        2: [(a_row(1,0,1,0), 1), (a_row(1,1,1,0), 1)],
        3: [(a_row(1,0,1,1), 1), (a_row(1,1,1,1), 1)]
    }
    v_maps = {
        0: [(b_row(0,0,0,0), 1)],
        1: [(b_row(0,0,0,1), 1)],
        2: [(b_row(0,0,1,0), 1)],
        3: [(b_row(0,0,1,1), 1)]
    }
    w_maps = {
        0: [(c_row(1,0,0,0), 1)],
        1: [(c_row(1,0,0,1), 1)],
        2: [(c_row(1,0,1,0), 1)],
        3: [(c_row(1,0,1,1), 1)]
    }
    add_strassen_2x2(u_maps, v_maps, w_maps, col)
    col += 7
    
    # Term 3: M3 = A00(B01-B11) -> C01
    u_maps = { 
        0: [(a_row(0,0,0,0), 1)],
        1: [(a_row(0,0,0,1), 1)],
        2: [(a_row(0,0,1,0), 1)],
        3: [(a_row(0,0,1,1), 1)]
    }
    v_maps = {
        0: [(b_row(0,1,0,0), 1), (b_row(1,1,0,0), -1)],
        1: [(b_row(0,1,0,1), 1), (b_row(1,1,0,1), -1)],
        2: [(b_row(0,1,1,0), 1), (b_row(1,1,1,0), -1)],
        3: [(b_row(0,1,1,1), 1), (b_row(1,1,1,1), -1)]
    }
    w_maps = {
        0: [(c_row(0,1,0,0), 1)],
        1: [(c_row(0,1,0,1), 1)],
        2: [(c_row(0,1,1,0), 1)],
        3: [(c_row(0,1,1,1), 1)]
    }
    add_strassen_2x2(u_maps, v_maps, w_maps, col)
    col += 7
    
    # Term 4: M4 = A11(B10-B00) -> C11
    u_maps = { 
        0: [(a_row(1,1,0,0), 1)],
        1: [(a_row(1,1,0,1), 1)],
        2: [(a_row(1,1,1,0), 1)],
        3: [(a_row(1,1,1,1), 1)]
    }
    v_maps = {
        0: [(b_row(1,0,0,0), 1), (b_row(0,0,0,0), -1)],
        1: [(b_row(1,0,0,1), 1), (b_row(0,0,0,1), -1)],
        2: [(b_row(1,0,1,0), 1), (b_row(0,0,1,0), -1)],
        3: [(b_row(1,0,1,1), 1), (b_row(0,0,1,1), -1)]
    }
    w_maps = {
        0: [(c_row(1,1,0,0), 1)],
        1: [(c_row(1,1,0,1), 1)],
        2: [(c_row(1,1,1,0), 1)],
        3: [(c_row(1,1,1,1), 1)]
    }
    add_strassen_2x2(u_maps, v_maps, w_maps, col)
    col += 7
    
    # Term 5: M5 = (A00+A10)B11 -> C10
    u_maps = { 
        0: [(a_row(0,0,0,0), 1), (a_row(1,0,0,0), 1)],
        1: [(a_row(0,0,0,1), 1), (a_row(1,0,0,1), 1)],
        2: [(a_row(0,0,1,0), 1), (a_row(1,0,1,0), 1)],
        3: [(a_row(0,0,1,1), 1), (a_row(1,0,1,1), 1)]
    }
    v_maps = {
        0: [(b_row(1,1,0,0), 1)],
        1: [(b_row(1,1,0,1), 1)],
        2: [(b_row(1,1,1,0), 1)],
        3: [(b_row(1,1,1,1), 1)]
    }
    w_maps = {
        0: [(c_row(1,0,0,0), 1)],
        1: [(c_row(1,0,0,1), 1)],
        2: [(c_row(1,0,1,0), 1)],
        3: [(c_row(1,0,1,1), 1)]
    }
    add_strassen_2x2(u_maps, v_maps, w_maps, col)
    col += 7
    
    # Term 6: M6 = (A11-A10)(B00+B10) -> C10
    u_maps = { 
        0: [(a_row(1,1,0,0), 1), (a_row(1,0,0,0), -1)],
        1: [(a_row(1,1,0,1), 1), (a_row(1,0,0,1), -1)],
        2: [(a_row(1,1,1,0), 1), (a_row(1,0,1,0), -1)],
        3: [(a_row(1,1,1,1), 1), (a_row(1,0,1,1), -1)]
    }
    v_maps = {
        0: [(b_row(0,0,0,0), 1), (b_row(1,0,0,0), 1)],
        1: [(b_row(0,0,0,1), 1), (b_row(1,0,0,1), 1)],
        2: [(b_row(0,0,1,0), 1), (b_row(1,0,1,0), 1)],
        3: [(b_row(0,0,1,1), 1), (b_row(1,0,1,1), 1)]
    }
    w_maps = {
        0: [(c_row(1,0,0,0), 1)],
        1: [(c_row(1,0,0,1), 1)],
        2: [(c_row(1,0,1,0), 1)],
        3: [(c_row(1,0,1,1), 1)]
    }
    add_strassen_2x2(u_maps, v_maps, w_maps, col)
    col += 7
    
    # Term 7: M7 = (A01-A11)(B11+B10) -> C01
    u_maps = { 
        0: [(a_row(0,1,0,0), 1), (a_row(1,1,0,0), -1)],
        1: [(a_row(0,1,0,1), 1), (a_row(1,1,0,1), -1)],
        2: [(a_row(0,1,1,0), 1), (a_row(1,1,1,0), -1)],
        3: [(a_row(0,1,1,1), 1), (a_row(1,1,1,1), -1)]
    }
    v_maps = {
        0: [(b_row(1,1,0,0), 1), (b_row(1,0,0,0), 1)],
        1: [(b_row(1,1,0,1), 1), (b_row(1,0,0,1), 1)],
        2: [(b_row(1,1,1,0), 1), (b_row(1,0,1,0), 1)],
        3: [(b_row(1,1,1,1), 1), (b_row(1,0,1,1), 1)]
    }
    w_maps = {
        0: [(c_row(0,1,0,0), 1)],
        1: [(c_row(0,1,0,1), 1)],
        2: [(c_row(0,1,1,0), 1)],
        3: [(c_row(0,1,1,1), 1)]
    }
    add_strassen_2x2(u_maps, v_maps, w_maps, col)
    col += 7
    
    # Part 2: Schoolbook 4x4x1
    # A(4x4) * B(4x1) -> C(4x1)
    # 16 multiplications.
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*5+4][col] = 1  # B row j, col 4 (0-indexed 4 is the 5th col)
            W[4*4+i][col] = 1  # C row i, col 4. W index = k*4+i = 4*4+i
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
