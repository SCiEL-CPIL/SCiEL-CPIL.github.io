"""Program for the suite task; Strassen-style recursive tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Build a rank-r decomposition of the (n x m) @ (m x p) tensor.

    Strategy: recursively decompose the multiplication using Strassen on the
    2x2 core blocks. Each Strassen step replaces 8 scalar products by 7
    bilinear terms. We emit a list of triplets (u_vec, v_vec, w_vec) over the
    flattened index spaces, then stack them into U, V, W.

    The returned matrices have integer entries and exactly decompose the
    matrix-multiplication tensor under W[k*n + i, c].
    """
    n, m, p = DIMENSIONS

    # terms: list of (u, v, w) where u is dict over (i,j), v over (j,k), w over (k,i)
    def add_terms(terms, u, v, w):
        terms.append((u, v, w))

    def scalar_terms(n, m, p, terms):
        # Schoolbook: one term per (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    add_terms(terms, u, v, w)

    def strassen_core(terms, ioff, joff, koff, n, m, p):
        # Decompose C[i0:i0+n, k0:k0+p] = A[i0:i0+n, j0:j0+m] @ B[j0:j0+m, k0:k0+p]
        # using Strassen on the 2x2 blocks (requires n, m, p to be 2).
        # We assume n == m == p == 2 here.
        A = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}  # identity over the block
        # Build the 7 Strassen terms over the 2x2 block.
        # Indices within block: ai in {0,1}, bj in {0,1}, ck in {0,1}
        def idx(i, j):  # A index
            return (ioff + i, joff + j)
        def idxb(j, k):  # B index
            return (joff + j, koff + k)
        def idxc(k, i):  # C index
            return (koff + k, ioff + i)

        # Strassen 7 products
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        terms7 = [
            # (coef for A entries, coef for B entries)
            ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),  # M1
            ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),              # M2
            ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}),             # M3
            ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}),             # M4
            ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),              # M5
            ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}),  # M6
            ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}),  # M7
        ]
        # C = M1 + M4 - M5 + M7, M3+M5, M2+M4, M1-M2+M3+M6
        Ccombos = [
            ((0, 0), [(0, +1), (3, +1), (4, -1), (6, +1)]),
            ((0, 1), [(2, +1), (4, +1)]),
            ((1, 0), [(1, +1), (3, +1)]),
            ((1, 1), [(0, +1), (1, -1), (2, +1), (5, +1)]),
        ]
        # For each of the 7 Strassen products, create a single rank-1 term.
        # u and v are the linear forms (dicts with coefficients).
        # w is the linear form on C (dicts with coefficients).
        
        # First, determine the w vector for each product t.
        # w_t will have entries at the C positions where M_t contributes.
        w_vectors = [dict() for _ in range(7)]
        
        for (ci, cj), combo in Ccombos:
            for (t, s) in combo:
                c_idx = idxc(cj, ci) # Note: idxc(k, i) -> (k, i). Here ci is i, cj is k? 
                                     # In Ccombos: ((0,0), ...) -> ci=0, cj=0.
                                     # C index is (k, i). So k=cj, i=ci.
                                     # Wait, Ccombos definition: ((0,0), ...) means C[0,0].
                                     # So i=0, k=0.
                                     # idxc(k, i) -> idxc(0, 0).
                                     # Let's check idxc definition: def idxc(k, i): return (koff + k, ioff + i)
                                     # So we need idxc(cj, ci).
                if c_idx in w_vectors[t]:
                    w_vectors[t][c_idx] += s
                else:
                    w_vectors[t][c_idx] = s

        for t in range(7):
            a_coefs, b_coefs = terms7[t]
            # Build u vector
            u = {}
            for (ai, aj), ac in a_coefs.items():
                u[idx(ai, aj)] = ac
            
            # Build v vector
            v = {}
            for (bk, ck), bc in b_coefs.items():
                v[idxb(bk, ck)] = bc
            
            # w vector is already built
            w = w_vectors[t]
            
            # Only add if w is not empty (should not be, but safety)
            if w:
                add_terms(terms, u, v, w)

    def decompose(terms, n, m, p, ioff, joff, koff):
        if n == 2 and m == 2 and p == 2:
            strassen_core(terms, ioff, joff, koff, n, m, p)
        else:
            # Fall back to schoolbook for non-2 blocks (ensures correctness)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = {(ioff + i, joff + j): 1}
                        v = {(joff + j, koff + k): 1}
                        w = {(koff + k, ioff + i): 1}
                        add_terms(terms, u, v, w)

    terms = []
    # For n=2, m=4, p=6: apply Strassen to all 2x2x2 sub-blocks.
    # Split m into 2 blocks of 2, p into 3 blocks of 2.
    # Each 2x2x2 block uses 7 Strassen terms: total 6 * 7 = 42 terms.
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                bn = min(2, n - i0)
                bm = min(2, m - j0)
                bp = min(2, p - k0)
                if bn == 2 and bm == 2 and bp == 2:
                    strassen_core(terms, i0, j0, k0, bn, bm, bp)
                else:
                    decompose(terms, bn, bm, bp, i0, j0, k0)
    
    # Try to merge duplicate or linearly dependent terms to reduce rank.
    # Convert to arrays and check for linear dependence.
    import numpy as np
    
    rank = len(terms)
    if rank > 0:
        U_arr = np.zeros((n * m, rank))
        V_arr = np.zeros((m * p, rank))
        W_arr = np.zeros((p * n, rank))
        
        for c, (u, v, w) in enumerate(terms):
            for (i, j), val in u.items():
                U_arr[i * m + j, c] = val
            for (j, k), val in v.items():
                V_arr[j * p + k, c] = val
            for (k, i), val in w.items():
                W_arr[k * n + i, c] = val
        
        # Check for linearly dependent columns using SVD
        # A term c is redundant if its contribution can be expressed as a combination of others
        # This is complex; for now, just check if any term is all zeros
        valid_terms = []
        for c in range(rank):
            if not (np.allclose(U_arr[:, c], 0) and np.allclose(V_arr[:, c], 0) and np.allclose(W_arr[:, c], 0)):
                valid_terms.append(c)
        
        if len(valid_terms) < rank:
            terms = [terms[c] for c in valid_terms]
            rank = len(terms)

    # Assemble factor matrices.
    # The terms contain dicts with indices. We need to map them to flat indices.
    # u: (i, j) -> i * m + j
    # v: (j, k) -> j * p + k
    # w: (k, i) -> k * n + i
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][c] = val
        for (j, k), val in v.items():
            V[j * p + k][c] = val
        for (k, i), val in w.items():
            W[k * n + i][c] = val
            
    # Verify correctness: check that sum_c U[:,c] * V[:,c] * W[:,c] gives the multiplication tensor
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = 1 if this is a valid (i,j,k) triple for multiplication
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    # where valid means i in [0,n), j in [0,m), k in [0,p)
    
    # Build the expected tensor
    expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, k] = 1.0
    
    # Compute actual tensor from decomposition
    actual = np.zeros((n, m, p))
    for c in range(rank):
        uc = U_np[:, c]
        vc = V_np[:, c]
        wc = W_np[:, c]
        # uc has indices (i*m + j), vc has (j*p + k), wc has (k*n + i)
        for idx_u in range(n * m):
            if abs(uc[idx_u]) < 1e-10:
                continue
            i = idx_u // m
            j = idx_u % m
            for idx_v in range(m * p):
                if abs(vc[idx_v]) < 1e-10:
                    continue
                jv = idx_v // p
                k = idx_v % p
                if jv != j:
                    continue
                for idx_w in range(p * n):
                    if abs(wc[idx_w]) < 1e-10:
                        continue
                    kw = idx_w // n
                    iw = idx_w % n
                    if kw != k or iw != i:
                        continue
                    actual[i, j, k] += uc[idx_u] * vc[idx_v] * wc[idx_w]
    
    # Check if they match
    if not np.allclose(actual, expected, atol=1e-10):
        # Fallback to schoolbook if Strassen decomposition is incorrect
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    add_terms(terms, u, v, w)
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c, (u, v, w) in enumerate(terms):
            for (i, j), val in u.items():
                U[i * m + j][c] = val
            for (j, k), val in v.items():
                V[j * p + k][c] = val
            for (k, i), val in w.items():
                W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
