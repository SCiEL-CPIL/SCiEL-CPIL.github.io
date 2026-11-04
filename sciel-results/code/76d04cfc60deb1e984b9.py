DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _build():
    n, m, p = 3, 4, 6
    # Split A into row-blocks T (rows 0,1) and B (row 2); col-blocks 1 (0,1) and 2 (2,3)
    # Split B into row-blocks 1 (0,1) and 2 (2,3); col-blocks 0,1,2 (0,1),(2,3),(4,5)
    # C sub-blocks: rows (T: 0,1), (B: 2); cols 0,1,2

    def strassen(a_r, a_c, b_r, b_c, c_r, c_c, U, V, W, col):
        # 2x2x2 Strassen at global positions
        # U index: (a_r+i)*m + (a_c+j); V: (b_r+j)*p + (b_c+k); W: (c_r+k)*n + (c_r_row+i)
        # a is rows a_r..a_r+1, cols a_c..a_c+1
        # b is rows b_r..b_r+1, cols b_c..b_c+1
        # c is rows c_r..c_r+1, cols c_r_row..c_r_row+1
        a00, a01 = (a_r, a_c), (a_r, a_c+1)
        a10, a11 = (a_r+1, a_c), (a_r+1, a_c+1)
        b00, b01 = (b_r, b_c), (b_r, b_c+1)
        b10, b11 = (b_r+1, b_c), (b_r+1, b_c+1)
        c00, c01 = (c_r, c_r), (c_r, c_r+1)
        c10, c11 = (c_r+1, c_r), (c_r+1, c_r+1)

        def uidx(ar, ac): return ar*m + ac
        def vidx(br, bc): return br*p + bc
        def widx(cr, cc): return cc*n + cr  # W[k*n+i] where k is C row, i is C col

        def add(ua, va, wa, coeff):
            U[ua][col] = coeff
            V[va][col] = 1
            W[wa][col] = 1

        def add2(ua1, va1, wa1, c1, ua2, va2, wa2, c2):
            # two products in one rank term is not allowed; each rank term = one triple product
            pass

        # M1 = (A00+A11)(B00+B11)  -> C00 += M1, C11 += M1
        # We need to handle sums: (A00+A11)(B00+B11) expands to 4 products but Strassen
        # combines them. Actually Strassen's M1 = (A00+A11)*(B00+B11) is ONE matrix product
        # of sums. In tensor form, (A00+A11)(B00+B11) = A00 B00 + A00 B11 + A11 B00 + A11 B11
        # which is 4 scalar products, NOT 1. So Strassen's trick works at the matrix-product
        # level by reusing the same M1 for two C blocks, saving at the C level.
        #
        # The correct tensor-level Strassen: each Mi is a scalar product of linear combos.
        # We represent linear combos by adding multiple (U,V,W) entries? No - each rank-1
        # tensor term is exactly one scalar triple product u * v * w. Linear combos in A and B
        # mean we need multiple rank-1 terms per Mi, which defeats the purpose.
        #
        # Correct approach: A rank-1 tensor term is U[i,c]*V[j,c]*W[k,c] contributing to
        # the (i,j,k) entry of the multiplication tensor. The full tensor T[i,j,k]=1 for all
        # i,j,k (it's the all-ones tensor for this fixed decomposition? No).
        #
        # The matrix multiplication tensor M has entries M[a_row, b_row, c_row, a_col, b_col, c_col]
        # but our flattening is U over (n*m)=12, V over (m*p)=24, W over (p*n)=18.
        # T[u,v,w] = 1 iff the corresponding (i,j,k) triple satisfies k = ? 
        # U index u = i*m+j (i in [0,3), j in [0,4))
        # V index v = j*p+k (j in [0,4), k in [0,6))
        # W index w = k*n+i (k in [0,6), i in [0,3))
        # The tensor is T[u,v,w] = 1 iff j_u == j_v and k_v == k_w and i_u == i_w.
        # So T is 1 on the "consistent" triples. Rank = number of rank-1 terms to cover.
        #
        # Strassen at tensor level: we want to cover the sub-block's consistent triples with
        # fewer terms. For a 2x2x2 block, the consistent triples are 8 (each (i,j,k) with i,j,k in {0,1}
        # all independent? No: i is C-col, j is shared, k is C-row. In the block, i in {0,1}, j in {0,1},
        # k in {0,1}, all 8 combinations are "consistent" within the block because there's no constraint
        # linking i,j,k beyond them being in range. Wait, that means the block tensor is all-ones 8 entries,
        # rank of all-ones 2x2x2 tensor is 2? No, the multiplication tensor for 2x2x2 has entries
        # T[i,j,k] where i=A-row... 
        #
        # I'm confusing myself. Let me restart the tensor definition.
        # A is n x m (3x4), B is m x p (4x6), C = A B is n x p (3x6).
        # C[i,k] = sum_j A[i,j] B[j,k], for i in [0,n), j in [0,m), k in [0,p).
        # U has shape (n*m, rank): U[i*m + j, c]
        # V has shape (m*p, rank): V[j*p + k, c]
        # W has shape (p*n, rank): W[k*n + i, c]
        # The decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,j,k consistent}
        # The "consistent" condition: the triple (i,j,k) appears in the multiplication, so
        # T[i*m+j, j*p+k, k*n+i] = 1 (it's the structure tensor).
        # Note i, j, k are free: i in [0,n), j in [0,m), k in [0,p). So T has n*m*p = 72 ones.
        # Schoolbook: 72 terms, one per (i,j,k).
        #
        # Strassen 2x2x2: for a block with i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1},
        # there are 8 entries. Can we cover 8 entries with 7 rank-1 terms? Each rank-1 term
        # U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] is 1 at a SINGLE (i,j,k) if U,V,W are 0/1 at specific
        # indices. To use Strassen, we need U,V,W to be 0/1 but possibly at MULTIPLE indices?
        # No - U[i*m+j,c] is a single number; if it's 1 for two different (i,j) pairs, that's
        # two ones in the U row c, which means the rank-1 tensor has those two (i,j) both active.
        #
        # Actually yes! A rank-1 term with U having 1s at (i1,j1) and (i2,j2), V having 1s at
        # (j1,k1) and (j2,k2), etc., gives a product tensor that is the OUTER product, having
        # ones at all combinations. That's how Strassen works: M1 = (A00+A11)(B00+B11) means
        # U has 1 at (A00) and (A11) positions, V has 1 at (B00) and (B11) positions, and W...
        # but W must give the C output. M1 contributes to BOTH C00 and C11. So W has 1 at C00
        # and C11? Then the term is (A00+A11) * (B00+B11) * (C00+C11) as an outer product,
        # which gives 4*4 = 16 ones? That's not right either.
        #
        # I think the correct tensor formulation for Strassen: each Mi corresponds to a rank-1
        # term where U, V, W each have a specific pattern. Let me just use the known 7-term
        # formula and verify by direct expansion.
        #
        # Let me use a simpler verification: I'll build the 7 terms for the 2x2x2 block and
        # check that their sum equals the 8 identity terms.

        # For block at rows i0,i1; cols j0,j1; output rows k0,k1:
        # A positions: (i0,j0),(i0,j1),(i1,j0),(i1,j1)
        # B positions: (j0,k0),(j0,k1),(j1,k0),(j1,k1)
        # C positions: (k0,i0),(k0,i1),(k1,i0),(k1,i1)
        #
        # We want: sum over terms = sum_{i,j,k in block} e_{i,j} * e_{j,k} * e_{k,i}
        # = e_{i0,j0,k0,i0} + e_{i0,j0,k1,i0} + ... (8 terms)
        #
        # This is the all-ones tensor on the 2x2x2 index set (i,j,k), which has rank 2 (it's
        # the product of three all-ones vectors). So actually 2 terms suffice! But that's the
        # trivial factorization, not Strassen. The point is we're covering 8 specific entries
        # of the LARGER 12x24x18 tensor.
        #
        # I think the cleanest approach: just do schoolbook but merge where possible.
        # Actually, let me reconsider. The 8 entries are:
        # (i0,j0,k0), (i0,j0,k1), (i0,j1,k0), (i0,j1,k1),
        # (i1,j0,k0), (i1,j0,k1), (i1,j1,k0), (i1,j1,k1)
        # Each rank-1 term picks one i, one j, one k. So 8 terms needed for 8 entries, UNLESS
        # a term can cover multiple entries. A rank-1 term with U=1 at (i0,j0) and (i1,j1),
        # V=1 at (j0,k0) and (j1,k1), W=1 at (k0,i0) and (k1,i1) gives the outer product
        # which has 2*2*2 = 8 ones, but at the WRONG positions generally.
        #
        # I'll just implement schoolbook for the 1x2x2 blocks and use a known 7-term
        # Strassen for 2x2x2 by direct construction and verification.
        #
        # Let me just use schoolbook everywhere for correctness, rank 72.
        # No wait, the task wants rank < 72. Let me do the 66-term construction.
        
        # I'll define the 7 Strassen terms explicitly for the block.
        # Term structure: (U_pattern, V_pattern, W_pattern)
        # where each pattern is a dict from (index) to coefficient.
        # The rank-1 tensor = sum over U_pattern * V_pattern * W_pattern (outer product).
        # We need the sum of 7 such outer products = the 8 identity terms.
        
        # This is complex. Let me just hardcode the 66-term solution by using
        # the known result that 3x4x6 has rank 66 via specific construction.
        # I'll use a programmatic approach: for each of the 8 blocks, add its terms.
        # For 2x2x2 blocks, add 7 terms (Strassen). For 1x2x2 blocks, add 4 terms.
        # I'll verify by checking the tensor equation numerically.
        
        pass  # placeholder - will implement in main

    # Given the complexity, let me just do a verified schoolbook rank-72 and note
    # that 66 requires careful Strassen tensor construction.
    # Actually, I'll implement the 66-term construction carefully.
    
    # Blocks:
    # (T,1,0): i={0,1}, j={0,1}, k={0,1}  -> Strassen
    # (T,1,1): i={0,1}, j={0,1}, k={2,3}  -> Strassen
    # (T,1,2): i={0,1}, j={0,1}, k={4,5}  -> Strassen
    # (T,2,0): i={0,1}, j={2,3}, k={0,1}  -> Strassen
    # (T,2,1): i={0,1}, j={2,3}, k={2,3}  -> Strassen
    # (T,2,2): i={0,1}, j={2,3}, k={4,5}  -> Strassen
    # (B,1,0): i={2}, j={0,1}, k={0,1}    -> Schoolbook
    # (B,1,1): i={2}, j={0,1}, k={2,3}    -> Schoolbook
    # (B,1,2): i={2}, j={0,1}, k={4,5}    -> Schoolbook
    # (B,2,0): i={2}, j={2,3}, k={0,1}    -> Schoolbook
    # (B,2,1): i={2}, j={2,3}, k={2,3}    -> Schoolbook
    # (B,2,2): i={2}, j={2,3}, k={4,5}    -> Schoolbook
    
    # That's 6 Strassen (7 each = 42) + 6 Schoolbook (4 each = 24) = 66. 
    # Wait, I listed 6 Strassen and 6 Schoolbook = 12 blocks, but there should be
    # 2 row-blocks x 2 col-blocks x 3 C-col-blocks = 12. Yes.
    # 6 Strassen + 6 Schoolbook = 42 + 24 = 66. 

    rank = 66
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0

    def uidx(i, j): return i*m + j
    def vidx(j, k): return j*p + k
    def widx(k, i): return k*n + i  # W[k*n+i]

    def add_schoolbook_block(i0, j0, j1, k0, k1):
        # i in {i0}, j in {j0,j1}, k in {k0,k1}
        nonlocal col
        for j in range(j0, j1+1):
            for k in range(k0, k1+1):
                U[uidx(i0, j)][col] = 1
                V[vidx(j, k)][col] = 1
                W[widx(k, i0)][col] = 1
                col += 1

    def add_strassen_block(i0, j0, k0):
        # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
        nonlocal col
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        # 7 Strassen terms. Each term: (list of (i,j) with coeff, list of (j,k) with coeff, list of (k,i) with coeff)
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1]) * (C[k0,i0]+C[k1,i1])
        # This outer product gives 4*4 = 16 entries, but we only want specific ones.
        # Actually the rank-1 tensor from U having 1s at (i0,j0),(i1,j1), V at (j0,k0),(j1,k1),
        # W at (k0,i0),(k1,i1) is the product, which is 1 at all 2*2*2=8 combinations of the
        # chosen indices. That's not what we want.
        #
        # I think I need to be more careful. Let me just use the explicit 7-term formula
        # where each term is a single (i,j,k) triple product, but using linear combinations
        # by having multiple non-zero entries in U, V, W for that column.
        #
        # For M1: U has 1 at (i0,j0) and 1 at (i1,j1); V has 1 at (j0,k0) and 1 at (j1,k1);
        # W has 1 at (k0,i0) and 1 at (k1,i1). The outer product gives 8 ones at
        # {(i0,j0,k0,i0), (i0,j0,k1,i1), (i1,j1,k0,i0), (i1,j1,k1,i1), ...} which is wrong.
        #
        # The issue: the rank-1 tensor is U ⊗ V ⊗ W, and we want the SUM of 7 such tensors
        # to equal the 8 identity tensors. This is a tensor decomposition problem.
        #
        # Let me just verify schoolbook works and return rank 72 for now, with a comment
        # about the 66 construction.
        pass

    # Given the time complexity of getting Strassen exactly right, let me implement
    # the 66-term construction by directly specifying all 66 (U,V,W) patterns and
    # verifying numerically.
    
    # I'll use a different approach: known rank-66 decomposition via explicit formula.
    # For 3x4x6, I'll construct it block by block and verify.
    
    # Let me just do schoolbook rank 72 as a safe fallback and note the improvement path.
    # No - the task requires rank < 72 for improvement. Let me carefully build Strassen.
    
    # Restart: I'll build the full 66-term solution and verify with numpy.
    import numpy as np
    import time
    start = time.time()

    # Build 66 terms
    col = 0
    U = [[0]*66 for _ in range(12)]
    V = [[0]*66 for _ in range(24)]
    W = [[0]*66 for _ in range(18)]

    def sb(i, j, k):
        # schoolbook term
        nonlocal col
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        col += 1

    def st(i0, j0, k0):
        # Strassen 2x2x2 block
        nonlocal col
        i1, j1, k1 = i0+1, j0+1, k0+1
        # 7 terms, each is (U_entries, V_entries, W_entries)
        # Each entry is (index, coeff)
        terms = [
            # M1
            ([(i0,j0,1),(i1,j1,1)], [(j0,k0,1),(j1,k1,1)], [(k0,i0,1),(k1,i1,1)]),
            # M2
            ([(i1,j0,1),(i1,j1,1)], [(j0,k0,1)], [(k0,i0,1),(k0,i1,1)]),
            # M3
            ([(i0,j0,1)], [(j0,k1,-1),(j0,k0,1)], [(k1,i0,1)]),
            # M4
            ([(i1,j1,1)], [(j1,k0,-1),(j0,k0,1)], [(k0,i1,1)]),
            # M5
            ([(i0,j0,1),(i0,j1,1)], [(j1,k1,1)], [(k1,i0,1)]),
            # M6
            ([(i1,j0,-1),(i0,j0,1)], [(j0,k0,1),(j0,k1,1)], [(k1,i1,1)]),
            # M7
            ([(i0,j1,1),(i1,j1,-1)], [(j1,k0,1),(j1,k1,1)], [(k0,i0,1)]),
        ]
        for u_ents, v_ents, w_ents in terms:
            for (i, j, c) in u_ents:
                U[i*m+j][col] += c
            for (j, k, c) in v_ents:
                V[j*p+k][col] += c
            for (k, i, c) in w_ents:
                W[k*n+i][col] += c
            col += 1

    # 6 Strassen blocks
    st(0, 0, 0)  # i={0,1}, j={0,1}, k={0,1}
    st(0, 0, 2)  # i={0,1}, j={0,1}, k={2,3}
    st(0, 0, 4)  # i={0,1}, j={0,1}, k={4,5}
    st(0, 2, 0)  # i={0,1}, j={2,3}, k={0,1}
    st(0, 2, 2)  # i={0,1}, j={2,3}, k={2,3}
    st(0, 2, 4)  # i={0,1}, j={2,3}, k={4,5}
    # 6 Schoolbook blocks
    sb(2, 0, 0)  # i=2, j=0, k=0
    sb(2, 0, 1)  # i=2, j=0, k=1
    sb(2, 0, 2)
    sb(2, 0, 3)
    sb(2, 1, 0)
    sb(2, 1, 1)
    sb(2, 1, 2)
    sb(2, 1, 3)
    sb(2, 2, 0)
    sb(2, 2, 1)
    sb(2, 2, 2)
    sb(2, 2, 3)
    sb(2, 2, 4)
    sb(2, 2, 5)
    sb(2, 3, 0)
    sb(2, 3, 1)
    sb(2, 3, 2)
    sb(2, 3, 3)
    sb(2, 3, 4)
    sb(2, 3, 5)

    # That's 6*7 + 18 = 42 + 18 = 60, not 66. I have too many schoolbook.
    # I need 6 Strassen (42) + 6 Schoolbook blocks of 4 each (24) = 66.
    # The schoolbook blocks should be 1x2x2, i.e., i={2}, j={j0,j1}, k={k0,k1}.
    # That's 4 terms each. 6 such blocks = 24. Total 42+24=66.
    # Let me redo.

    col = 0
    U = [[0]*66 for _ in range(12)]
    V = [[0]*66 for _ in range(24)]
    W = [[0]*66 for _ in range(18)]

    def sb2(i, j0, j1, k0, k1):
        nonlocal col
        for j in range(j0, j1+1):
            for k in range(k0, k1+1):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    def st2(i0, j0, k0):
        nonlocal col
        i1, j1, k1 = i0+1, j0+1, k0+1
        terms = [
            ([(i0,j0,1),(i1,j1,1)], [(j0,k0,1),(j1,k1,1)], [(k0,i0,1),(k1,i1,1)]),
            ([(i1,j0,1),(i1,j1,1)], [(j0,k0,1)], [(k0,i0,1),(k0,i1,1)]),
            ([(i0,j0,1)], [(j0,k1,-1),(j0,k0,1)], [(k1,i0,1)]),
            ([(i1,j1,1)], [(j1,k0,-1),(j0,k0,1)], [(k0,i1,1)]),
            ([(i0,j0,1),(i0,j1,1)], [(j1,k1,1)], [(k1,i0,1)]),
            ([(i1,j0,-1),(i0,j0,1)], [(j0,k0,1),(j0,k1,1)], [(k1,i1,1)]),
            ([(i0,j1,1),(i1,j1,-1)], [(j1,k0,1),(j1,k1,1)], [(k0,i0,1)]),
        ]
        for u_ents, v_ents, w_ents in terms:
            for (i, j, c) in u_ents:
                U[i*m+j][col] += c
            for (j, k, c) in v_ents:
                V[j*p+k][col] += c
            for (k, i, c) in w_ents:
                W[k*n+i][col] += c
            col += 1

    st2(0, 0, 0)
    st2(0, 0, 2)
    st2(0, 0, 4)
    st2(0, 2, 0)
    st2(0, 2, 2)
    st2(0, 2, 4)
    sb2(2, 0, 1, 0, 1)
    sb2(2, 0, 1, 2, 3)
    sb2(2, 0, 1, 4, 5)
    sb2(2, 2, 3, 0, 1)
    sb2(2, 2, 3, 2, 3)
    sb2(2, 2, 3, 4, 5)

    # Verify
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c] for c in range(66))
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    if max_err > 1e-9:
        # Fallback to schoolbook rank 72
        U = [[0]*72 for _ in range(12)]
        V = [[0]*72 for _ in range(24)]
        W = [[0]*72 for _ in range(18)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

def construct():
    import time
    start = time.time()
    result = _build()
    elapsed = time.time() - start
    if elapsed > 120:
        # Should not happen, but safeguard
        pass
    return result
# EVOLVE-BLOCK-END
