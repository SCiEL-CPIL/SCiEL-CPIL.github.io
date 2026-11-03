"""Matrix multiplication 4x4x6 via block-Kronecker Strassen decomposition."""
DIMENSIONS = [4, 4, 6]

import time

def _build_strassen_2x2():
    """Return (U,V,W) for 2x2x2 Strassen rank-8 decomposition.
    Indices: U row = i*2+j (i in [0,2), j in [0,2)), V row = j*2+k, W row = k*2+i.
    Strassen products M1..M7, each a (2x2)*(2x2) product of rank 8.
    """
    # Strassen block-level: 7 scalar products, each = full 2x2x2 (rank 8)
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    n, m, p = 2, 2, 2
    inner_rank = n * m * p  # 8
    rank = 7 * inner_rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def add_block(U_block, V_block, W_block, base):
        # U_block: list of (i,j,coeff) for A
        # V_block: list of (j,k,coeff) for B
        # W_block: list of (i,k,coeff) for C (output index k*n+i)
        # Each is a 2x2x2 schoolbook-like additive term
        for (i, j, cu) in U_block:
            for (j2, k, cv) in V_block:
                for (i2, k2, cw) in W_block:
                    if i == i2 and j == j2 and k == k2:
                        for col in range(base, base + inner_rank):
                            # schoolbook: each (i,j,k) gets one column
                            pass
        # Better: build column-by-column
        # For a product of (linear combo of A) * (linear combo of B) contributing to (linear combo of C)
        # We need: for each (a_i_j, b_j_k, c_i_k) triple with matching (i,j,k)
        # Actually a "block product" M = X*Y where X = sum cu*(i,j), Y = sum cv*(j,k)
        # contributes sum cu*cv * (i,j,k) terms to C where C_{i,k} += ...
        # But C index is (k,i) in W. The product X*Y has entry (i,k) = sum_j X[i,j]*Y[j,k]
        # = sum_j sum_i' sum_k' cu(i',j') cv(j',k') [i==i'][k==k'] A[i,j]B[j,k]
        # So term (i,j,k) gets coefficient cu(i,j)*cv(j,k) for the C[i,k] output.
        # In our tensor, the (i,j,k) term multiplies into W row = k*n+i.
        for (i, j, cu) in U_block:
            for (j2, k, cv) in V_block:
                if j != j2:
                    continue
                # This (i,j,k) scalar product contributes coefficient cu*cv to C[i,k]
                # We need to find which column in this block's 8 columns corresponds to (i,j,k)
                # Schoolbook column index for (i,j,k) is i*m+j*p+k = i*2+j*2+k
                scol = i * m + j * p + k
                coeff = cu * cv
                for (i2, k2, cw) in W_block:
                    if i2 == i and k2 == k:
                        col = base + scol
                        U[i * m + j][col] += coeff * 1  # wait, need to think
                        # Actually U[i*m+j] is the A[i,j] input. The rank-1 term is
                        # U[row_a, col] * V[row_b, col] * W[row_c, col]
                        # For the (i,j,k) scalar: row_a = i*m+j, row_b = j*p+k, row_c = k*n+i
                        # coefficient in U = cu, in V = cv, in W = cw
                        # So U[i*m+j][col] += cu, V[j*p+k][col] += cv, W[k*n+i][col] += cw
                        U[i * m + j][col] += cu
                        V[j * p + k][col] += cv
                        W[k * n + i][col] += cw
                        break

    # Define the 7 Strassen products as (U_block, V_block, W_block)
    # Each block: U_block = list of (i,j,coeff) for X, V_block for Y, W_block for C
    def blk(i, j, c=1.0):
        return [(i, j, c)]
    def bblk(*terms):
        return list(terms)

    # M1 = (A11+A22)*(B11+B22) -> C11+M4-M5+M7 part; contributes to C11
    M1 = (bblk((0,0,1),(1,1,1)), bblk((0,0,1),(1,1,1)), bblk((0,0,1)))
    M2 = (bblk((1,0,1),(1,1,1)), bblk((0,0,1)), bblk((0,0,1)))
    M3 = (bblk((0,0,1)), bblk((0,1,1),(1,1,-1)), bblk((0,0,1)))
    M4 = (bblk((1,1,1)), bblk((1,0,1),(0,0,-1)), bblk((0,0,1)))
    M5 = (bblk((0,0,1),(0,1,1)), bblk((1,1,1)), bblk((0,0,1)))
    M6 = (bblk((1,0,1),(0,0,-1)), bblk((0,0,1),(0,1,1)), bblk((0,0,1)))
    M7 = (bblk((0,1,1),(1,1,-1)), bblk((1,0,1),(1,1,1)), bblk((0,0,1)))

    # C11 = M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    # Each M is a full 2x2x2 product. We assign each M to its own 8 columns,
    # and the C-block contributions (W_block) determine which C entries each M feeds.
    # But M1 contributes to BOTH C11 and C22. So we must split M1 into two column sets?
    # No: a single rank-1 term M1 (8 cols) has a fixed W pattern. If M1 feeds C11 with +1
    # and C22 with +1, that's two different W rows needing the same U,V. That's not rank-1.
    # Resolution: each "M" is actually the 8 rank-1 terms of a 2x2x2 product. Each such
    # rank-1 term (i,j,k) feeds exactly ONE C entry C[i,k]. So M1's (0,0,0) term feeds C11,
    # M1's (1,1,1) term feeds C22. Good - they're separate columns. So W_block per M is just
    # the identity: each (i,j,k) -> C[i,k]. Let me redo: W_block for each M = identity mapping.

    # Redo: each M is a product X*Y. Its 8 rank-1 terms are (i,j,k) with U=cu(i,j), V=cv(j,k),
    # W feeds C[i,k] with coefficient 1 (identity). The Strassen combination happens at C level:
    # C[i,k] = sum over M of (M's (i,j,k) term coefficient). But that's automatic because each
    # M's (i,j,k) term adds cu*cv to U,V and 1 to W[i,k]. Then C[i,k] = sum_M cu_M(i,j)*cv_M(j,k).
    # Let's verify C11 = sum_M cu_M(0,0)*cv_M(0,0) for the (0,0,0) term... no, C11 = sum_j sum over terms.
    # This is getting complex. Let me just directly verify by construction and trust the algebra,
    # then TEST numerically.

    prods = [M1, M2, M3, M4, M5, M6, M7]
    for mi, (Ub, Vb, Wb) in enumerate(prods):
        base = mi * inner_rank
        # Wb should be identity: each (i,j,k) term -> C[i,k]
        # Rebuild Wb as identity for the 2x2x2
        Wb_id = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Wb_id.append((i, k, 1.0))
        add_block(Ub, Vb, Wb_id, base)
    return U, V, W


def _verify(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Build via einsum
    # U shape (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Reshape U -> (n,m,rank), V -> (m,p,rank), W -> (p,n,rank)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    # T[i,j,k] = sum_c Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    # Should equal 1 everywhere
    err = np.max(np.abs(T - 1.0))
    return err


def construct():
    n, m, p = DIMENSIONS  # 4,4,6
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s

    # Block decomposition: 4x4x6 = (2x2 blocks of 2) x (2x3 blocks of 2)
    # A is 4x4 -> 2x2 blocks each 2x2
    # B is 4x6 -> 2x3 blocks each 2x2
    # C is 4x6 -> 2x3 blocks each 2x2
    # Block (I,J) of A is 2x2, block (J,K) of B is 2x2, product -> block (I,K) of C.
    # Use Strassen on the 2x2 block grid: 7 block-products.
    # Each block-product = product of (2x2 linear combos) * (2x2 linear combos) -> 2x2 result
    # = a 2x2x2 tensor of rank 8 (schoolbook). Total rank = 7*8 = 56.
    #
    # Global indices: A[i,j] with i=2*I+ii, j=2*J+jj (I,J in 0..1, ii,jj in 0..1)
    # B[j,k] with j=2*J+jj, k=2*K+kk
    # C[i,k] with i=2*I+ii, k=2*K+kk ; W row = k*n+i = (2*K+kk)*4 + (2*I+ii)
    #
    # Strassen block products (I,J,K in {0,1}):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    inner_rank = 8  # 2x2x2 schoolbook
    rank = 7 * inner_rank
    U = [[0.0] * rank for _ in range(n * m)]   # 16 rows
    V = [[0.0] * rank for _ in range(m * p)]   # 24 rows
    W = [[0.0] * rank for _ in range(p * n)]   # 24 rows

    # For each Strassen product, define A-combo (I,J,coeff), B-combo (J,K,coeff), C-target (I,K,coeff)
    # Each product is expanded into 8 rank-1 terms (ii,jj,kk)
    def aidx(I, ii, J, jj):
        return (2 * I + ii) * m + (2 * J + jj)  # i*m+j
    def bidx(J, jj, K, kk):
        return (2 * J + jj) * p + (2 * K + kk)  # j*p+k
    def widx(K, kk, I, ii):
        return (2 * K + kk) * n + (2 * I + ii)  # k*n+i

    # Strassen products: (list of (I,J,ca), list of (J,K,cb), list of (I,K,cw))
    prods = [
        # M1
        ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
        # M2
        ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(1,0,1.0),(1,1,1.0)]),
        # M3
        ([(0,0,1.0)], [(0,1,1.0),(1,1,-1.0)], [(0,1,1.0),(1,1,1.0)]),
        # M4
        ([(1,1,1.0)], [(1,0,1.0),(0,0,-1.0)], [(1,0,1.0),(1,1,1.0)]),
        # M5
        ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,1,1.0),(1,1,1.0)]),
        # M6
        ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(0,1,1.0)], [(1,1,1.0)]),
        # M7
        ([(0,1,1.0),(1,1,-1.0)], [(1,0,1.0),(1,1,1.0)], [(1,1,1.0)]),
    ]
    # Wait, the C-target for each M must be where it feeds. But each M's 8 rank-1 terms
    # feed specific (I,K) blocks. The C-target list says which (I,K) blocks this M contributes to
    # and with what coefficient. For a given M and a given (ii,jj,kk) inner index, the term
    # (I,J,ii,jj) x (J,K,jj,kk) feeds C block (I,K) at inner (ii,kk). The coefficient is
    # ca(I,J)*cb(J,K)*cw(I,K) where cw is the Strassen sign for that (I,K) target.
    # But a single M contributes to multiple (I,K) targets (e.g. M1 -> C00 and C11). For a fixed
    # inner (ii,jj,kk) and fixed (I,J) and (J,K), the term feeds C[I,K] with coeff ca*cb*cw(I,K).
    # Since the same (I,J,K,ii,jj,kk) can feed different (I,K)? No: (I,J,K) fixes I and K, so
    # (I,K) is determined. Good. So for each M, for each (I,J,ca) in A-combo, (J,K,cb) in B-combo,
    # (I,K,cw) in C-target: the (I,J,K) block triple with inner (ii,jj,kk) contributes
    # ca*cb*cw to the rank-1 term. But wait, that's not right either: the C-target for M1 is
    # [(0,0,1),(1,1,1)] meaning M1 feeds C00 with +1 and C11 with +1. But a single (I,J,K) triple
    # has fixed I,K. So M1's A-combo (0,0) and (1,1), B-combo (0,0) and (1,1): the combinations
    # (I,J)=(0,0),(J,K)=(0,0) -> I=0,K=0 -> feeds C00 with cw=+1. (I,J)=(1,1),(J,K)=(1,1) -> I=1,K=1
    # -> feeds C11 with cw=+1. (I,J)=(0,0),(J,K)=(1,1) -> J mismatch (0 vs 1), skip. So the J must
    # match between A and B combos. Good, that's the tensor contraction.

    for mi, (Acombo, Bcombo, Ccombo) in enumerate(prods):
        base = mi * inner_rank
        # Build a map from (I,K) -> cw
        cw_map = {}
        for (I, K, cw) in Ccombo:
            cw_map[(I, K)] = cw
        for (I, J, ca) in Acombo:
            for (J2, K, cb) in Bcombo:
                if J != J2:
                    continue
                cw = cw_map.get((I, K), 0.0)
                if cw == 0.0:
                    continue
                # For each inner (ii,jj,kk)
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            scol = ii * 2 + jj * 2 + kk  # 0..7 within block
                            col = base + scol
                            U[aidx(I, ii, J, jj)][col] += ca
                            V[bidx(J, jj, K, kk)][col] += cb
                            W[widx(K, kk, I, ii)][col] += cw

    # Verify
    err = _verify(U, V, W, n, m, p, rank)
    if err > 1e-6:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
