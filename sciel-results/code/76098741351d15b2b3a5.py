"""Tensor decomposition of 5x5x5 matrix multiplication, rank 52."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = 5
    m = 5
    p = 5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_dict, v_dict, w_dict):
        U_cols.append(u_dict)
        V_cols.append(v_dict)
        W_cols.append(w_dict)

    # ---- Strassen 4x4 core (indices 0..3) ----
    # Partition 4x4 into 2x2 blocks with local indices (bi, bj) for bi,bj in {0,1}
    # Block (bi, bj) covers rows bi*2, bi*2+1 and cols bj*2, bj*2+1
    # Global index mapping: local (li, lj) in block (bi, bj) -> global (bi*2+li, bj*2+lj)

    # Strassen's 7 products for 2x2 block matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # For each Strassen product Mt, we need to build one rank-1 term.
    # The term has U entries for A-block positions, V entries for B-block positions,
    # W entries for C-block positions.

    # Define block-local to global index maps
    def block_to_global(bi, bj, li, lj):
        """Map block (bi,bj) local (li,lj) to global (i,j)"""
        return bi * 2 + li, bj * 2 + lj

    # For each of the 7 Strassen products, define the A-block coefficients,
    # B-block coefficients, and C-block contributions.
    # Each product Mt is a bilinear form: sum over (i,j,k) of coeff * A_ij * B_jk * C_ki
    # But actually in the tensor decomposition, each rank-1 term corresponds to
    # one bilinear form that contributes to C.

    # Let me think more carefully. The matrix multiplication tensor is:
    # T[i,j,k] = delta_{i,k} (no, that's wrong)
    # Actually: C_{ik} = sum_j A_{ij} B_{jk}
    # So T_{i,j,k} = 1 if the term A_{ij} * B_{jk} contributes to C_{ik}
    # The tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that
    # every (i,j,k) triple appears once).

    # For Strassen, we're decomposing the 4x4x4 tensor.
    # Each rank-1 term is a product of:
    #   u: a linear form in A (coefficients on A_{ij} for i,j in 0..3)
    #   v: a linear form in B (coefficients on B_{jk} for j,k in 0..3)
    #   w: a linear form in C (coefficients on C_{ki} for k,i in 0..3)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

    # For Strassen, the 7 products give 7 rank-1 terms for the 4x4 core.

    # Let me define the 7 Strassen terms explicitly.
    # Each term t has:
    #   U_t: dict mapping (i, j) -> coeff for i,j in 0..3
    #   V_t: dict mapping (j, k) -> coeff for j,k in 0..3
    #   W_t: dict mapping (k, i) -> coeff for k,i in 0..3

    # Strassen product 1: M1 = (A00 + A11)(B00 + B11)
    # A-block: A00 (bi=0,bj=0) + A11 (bi=1,bj=1)
    # B-block: B00 (bi=0,bj=0) + B11 (bi=1,bj=1)
    # C contribution: C00 += M1, C11 += M1
    # So W has +1 on C00 and +1 on C11 (in block terms)

    # Let me enumerate all 7 products:

    strassen_products = [
        # (name, A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_row, block_col, coeff)
        # B_blocks: list of (block_row, block_col, coeff)
        # C_blocks: list of (block_row, block_col, coeff)
        ("M1",
         [(0, 0, 1), (1, 1, 1)],       # A00 + A11
         [(0, 0, 1), (1, 1, 1)],       # B00 + B11
         [(0, 0, 1), (1, 1, 1)]),      # C00 + C11
        ("M2",
         [(1, 0, 1), (1, 1, 1)],       # A10 + A11
         [(0, 0, 1)],                  # B00
         [(1, 0, 1), (1, 1, -1)]),     # C10 - C11
        ("M3",
         [(0, 0, 1)],                  # A00
         [(0, 1, 1), (1, 1, -1)],      # B01 - B11
         [(0, 0, -1), (0, 1, 1)]),     # -C00 + C01
        ("M4",
         [(1, 1, 1)],                  # A11
         [(1, 0, 1), (0, 0, -1)],      # B10 - B00
         [(0, 0, 1), (1, 0, 1)]),      # C00 + C10
        ("M5",
         [(0, 0, 1), (0, 1, 1)],       # A00 + A01
         [(1, 1, 1)],                  # B11
         [(0, 1, 1), (1, 1, -1)]),     # C01 - C11
        ("M6",
         [(1, 0, 1), (0, 0, -1)],      # A10 - A00
         [(0, 0, 1), (0, 1, 1)],       # B00 + B01
         [(1, 0, -1), (1, 1, -1)]),    # -C10 - C11
        ("M7",
         [(0, 1, 1), (1, 1, -1)],      # A01 - A11
         [(1, 0, 1), (1, 1, 1)],       # B10 + B11
         [(0, 0, -1), (1, 1, 1)]),     # -C00 + C11
    ]

    for name, a_blocks, b_blocks, c_blocks in strassen_products:
        u_dict = {}
        v_dict = {}
        w_dict = {}

        for (bi, bj, coeff) in a_blocks:
            for li in range(2):
                for lj in range(2):
                    gi, gj = block_to_global(bi, bj, li, lj)
                    idx = gi * m + gj
                    u_dict[idx] = u_dict.get(idx, 0) + coeff

        for (bi, bj, coeff) in b_blocks:
            for li in range(2):
                for lj in range(2):
                    gi, gj = block_to_global(bi, bj, li, lj)
                    idx = gi * p + gj
                    v_dict[idx] = v_dict.get(idx, 0) + coeff

        for (bi, bj, coeff) in c_blocks:
            for li in range(2):
                for lj in range(2):
                    gi, gj = block_to_global(bi, bj, li, lj)
                    # C block (bi, bj) means C_{row, col} where row is from block row bi, col from block col bj
                    # In our convention, W[k*n+i] where k is the column index of C, i is the row index
                    # C_{ik} is at W[k*n+i]
                    # Block (bi, bj) of C: rows bi*2..bi*2+1, cols bj*2..bj*2+1
                    # So i = bi*2+li, k = bj*2+lj
                    idx = gj * n + gi  # W[k*n+i] with k=gj, i=gi
                    w_dict[idx] = w_dict.get(idx, 0) + coeff

        add_term(u_dict, v_dict, w_dict)

    # ---- Frame terms (involving index 4) ----
    # Case 1: C[i, 4] for i in 0..3: C[i,4] = sum_j A[i,j] B[j,4]
    # Split j into 0..3 (Strassen) and j=4 (schoolbook)
    # Strassen part: C[i,4] = sum_{j=0..3} A[i,j] B[j,4]
    # This is a 4x1 x 1x4 "outer product" style but really it's:
    # For each i in 0..3: C[i,4] = sum_{j=0..3} A[i,j] * B[j,4]
    # We can use Strassen on the 4x4 submatrix A[0..3, 0..3] * B[0..3, 4] (a 4x1 vector)
    # Actually, this is simpler: it's just 4 independent dot products of length 4.
    # But we can group them: think of A[0..3, 0..3] as a 4x4 matrix and B[0..3, 4] as a 4x1 matrix.
    # C[:, 4] = A[0..3, 0..3] * B[0..3, 4]
    # This is a 4x4 x 4x1 multiplication. We can use Strassen on the 4x4 part.
    # But Strassen gives 7 products for 4x4*4x4. Here B is 4x1, so we only need the first column of B.
    # Let's use the Strassen decomposition for the 4x4 core, but with B being a 4x1 vector.
    
    # Actually, let me think differently. The frame terms involve index 4.
    # C[i,4] for i in 0..3: these are 4 entries, each is a dot product of row i of A (length 5) with column 4 of B (length 5).
    # = sum_{j=0..4} A[i,j] * B[j,4]
    # = sum_{j=0..3} A[i,j] * B[j,4] + A[i,4] * B[4,4]
    
    # Similarly C[4,k] for k in 0..3: = sum_{j=0..3} A[4,j] * B[j,k] + A[4,4] * B[4,k]
    # And C[4,4] = sum_{j=0..4} A[4,j] * B[j,4] = sum_{j=0..3} A[4,j] * B[j,4] + A[4,4]*B[4,4]
    
    # The "cross" terms (involving index 4 in A or B but not both) can be handled with Strassen on 4x4 blocks.
    # Let me use Strassen for the 4x4 sub-blocks involving index 4.
    
    # For C[i,4], i in 0..3: 
    #   = sum_{j=0..3} A[i,j] * B[j,4] + A[i,4] * B[4,4]
    # The first part: 4x1 = 4x4 * 4x1. We can use Strassen-like decomposition.
    # Actually, for a 4x1 result, we just need 4 independent sums. But we can share computation.
    # Let's just use schoolbook for now but be smarter:
    # sum_{j=0..3} A[i,j] * B[j,4] for i=0..3
    # This is a 4x4 matrix A[0..3,0..3] times a 4x1 vector B[0..3,4].
    # We can decompose this using Strassen's 7 products applied to 2x2 blocks.
    
    # Let me use a 2x2 block Strassen for the 4x4 x 4x1 case.
    # A is 4x4, B is 4x1 (column vector). C is 4x1.
    # Partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B0], [B1]] where B0, B1 are 2x1
    # C = [[C0], [C1]] where C0, C1 are 2x1
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # Using Strassen-style:
    # M1 = (A00+A11)*B0  -> contributes to C0 and C1
    # M2 = A11*B1         -> contributes to C1
    # M3 = A00*(B1-B0)    -> contributes to C0
    # C0 = M1 + M3 - M2... no wait.
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # Let's use:
    # M1 = (A00 + A11) * B0
    # M2 = A01 * B1  
    # M3 = A10 * B0
    # M4 = A11 * B1
    # C0 = M1 - A11*B0 + M2 = M1 - (A11*B0) + M2
    # This isn't clean. Let me just use schoolbook for the frame terms for simplicity,
    # but optimize by combining where possible.
    
    # Actually, let me reconsider. The current approach uses:
    # Case 1: 4*5 = 20 terms
    # Case 2: 4*5 = 20 terms  
    # Case 3: 5 terms
    # Total: 45 terms for the frame.
    
    # Can we do better? 
    # C[i,4] = sum_{j=0..4} A[i,j] B[j,4] for i=0..3
    # C[4,k] = sum_{j=0..4} A[4,j] B[j,k] for k=0..3
    # C[4,4] = sum_{j=0..4} A[4,j] B[j,4]
    
    # The term A[4,4]*B[4,4] appears in all three (C[i,4] has A[i,4]*B[4,4] not A[4,4]*B[4,4])
    # Wait no: C[i,4] = sum_j A[i,j]B[j,4], so the j=4 term is A[i,4]*B[4,4].
    # C[4,k] = sum_j A[4,j]B[j,k], so the j=4 term is A[4,4]*B[4,k].
    # C[4,4] = sum_j A[4,j]B[j,4], so the j=4 term is A[4,4]*B[4,4].
    
    # So A[4,4]*B[4,4] only appears in C[4,4].
    # A[i,4]*B[4,4] for i=0..3 appears in C[i,4].
    # A[4,4]*B[4,k] for k=0..3 appears in C[4,k].
    
    # These are all distinct bilinear forms, so we can't easily combine them.
    # Let's stick with schoolbook for the frame but verify the count.
    
    # Case 1: C[i,4] for i in 0..3: 4 * 5 = 20 terms
    for i in range(4):
        for j in range(5):
            u = {i * m + j: 1}
            v = {j * p + 4: 1}
            w = {4 * n + i: 1}
            add_term(u, v, w)

    # Case 2: C[4, k] for k in 0..3: 4 * 5 = 20 terms
    for k in range(4):
        for j in range(5):
            u = {4 * m + j: 1}
            v = {j * p + k: 1}
            w = {k * n + 4: 1}
            add_term(u, v, w)

    # Case 3: C[4, 4]: 5 terms
    for j in range(5):
        u = {4 * m + j: 1}
        v = {j * p + 4: 1}
        w = {4 * n + 4: 1}
        add_term(u, v, w)

    rank = len(U_cols)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify with a few random tests
    import random
    rng = random.Random(42)
    for trial in range(20):
        if time.time() > tmax:
            break
        A = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[ii][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)] for ii in range(n)]
        C_test = [[0] * p for _ in range(n)]
        for ii in range(n):
            for kk in range(p):
                s = 0
                for r in range(rank):
                    for jj in range(m):
                        s += U[ii * m + jj][r] * V[jj * p + kk][r] * W[kk * n + ii][r]
                C_test[ii][kk] = s
        if C_test != C_true:
            # Debug: find the mismatch
            for ii in range(n):
                for kk in range(p):
                    if C_test[ii][kk] != C_true[ii][kk]:
                        # This is expected to fail if there's a bug
                        pass
            return _schoolbook()

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
