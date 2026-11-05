import time

DIMENSIONS = [3, 4, 5]

def construct():
    start = time.time()
    deadline = start + 120.0
    n, m, p = DIMENSIONS

    # We will use a partition based on Strassen 2x2x2 blocks where possible.
    # Partition indices:
    # i: [0, 2), [2, 3)
    # j: [0, 2), [2, 4)
    # k: [0, 2), [2, 4), [4, 5)
    
    i_blocks = [(0, 2), (2, 3)]
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4), (4, 5)]

    U = [[0] * (n * m) for _ in range(n * m)]
    V = [[0] * (m * p) for _ in range(m * p)]
    W = [[0] * (p * n) for _ in range(p * n)]
    
    # We need to accumulate columns dynamically.
    # Instead of pre-allocating, we'll store tuples (i, j, k, coeff) for U, V, W
    # but since U, V, W are matrices, we can just build lists of non-zero entries
    # and then convert to dense matrices.
    
    # Let's use a dictionary to accumulate coefficients for each (index, rank) pair
    # But rank is not fixed. We'll just append to lists.
    
    u_entries = [] # (row, col, val)
    v_entries = []
    w_entries = []
    
    col_idx = 0
    
    for i0, i1 in i_blocks:
        ni = i1 - i0
        for j0, j1 in j_blocks:
            nj = j1 - j0
            for k0, k1 in k_blocks:
                nk = k1 - k0
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Strassen 2x2x2
                    # Local indices 0, 1
                    # M1 = (a+d)(e+h)
                    # U: (i0, j0) + (i0+1, j0+1) ? No, Strassen combines inputs.
                    # The rank-1 tensors are not just single (i,j,k) triples.
                    # They are linear combinations.
                    
                    # We need to add 7 rank-1 terms.
                    # Each term is (u_vec, v_vec, w_vec)
                    
                    # M1: u = e0 + e1 (in i), e0 + e1 (in j) ?
                    # Actually, Strassen works on the 2x2 matrices.
                    # The decomposition is:
                    # C = sum_{r=1}^7 u_r \otimes v_r \otimes w_r
                    
                    # Let's define the vectors for each of the 7 multiplications.
                    # Indices are local (0,1).
                    
                    # M1: (A00+A11)(B00+B11)
                    # u1: [1, 1, 1, 1] ? No, u is for A.
                    # A is 2x2. u1 = [1, 0, 1, 0]? No.
                    # A = [[a, b], [c, d]].
                    # M1 uses a, d.
                    # u1 = [1, 0, 0, 1] (a, b, c, d) -> a + d
                    # v1 = [1, 0, 0, 1] (e, f, g, h) -> e + h
                    # w1 = [1, 1, 1, 1] (c00, c01, c10, c11) -> c00 + c01 + c10 + c11?
                    # No, w1 corresponds to the contribution to C.
                    # C00 += M1, C01 += M1, C10 += M1, C11 += M1?
                    # No.
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # So w1 = [1, 0, 0, 1] (C00, C01, C10, C11) -> C00 + C11
                    # Wait, the tensor T maps A, B to C.
                    # The rank-1 term is $u \otimes v \otimes w$.
                    # Contribution to C is $w \times (u^T A) \times (v^T B)$.
                    
                    # M1:
                    # u1 = [1, 0, 0, 1] (a, b, c, d)
                    # v1 = [1, 0, 0, 1] (e, f, g, h)
                    # w1 = [1, 0, 0, 1] (c00, c01, c10, c11) -> c00 + c11
                    # But we need c00 += M1 and c11 += M1.
                    # So w1 = [1, 0, 0, 1] is correct for C00 and C11.
                    
                    # M2: (c+d)e
                    # u2 = [0, 0, 1, 1]
                    # v2 = [1, 0, 0, 0]
                    # w2 = [0, 0, 1, 0] (C10)
                    
                    # M3: a(f-h)
                    # u3 = [1, 0, 0, 0]
                    # v3 = [0, 1, 0, -1]
                    # w3 = [0, 1, 0, 1] (C01, C11)
                    
                    # M4: d(g-e)
                    # u4 = [0, 0, 0, 1]
                    # v4 = [0, 0, 1, -1]
                    # w4 = [1, 0, 1, 0] (C00, C10)
                    
                    # M5: (a+b)h
                    # u5 = [1, 1, 0, 0]
                    # v5 = [0, 0, 0, 1]
                    # w5 = [-1, 1, 0, 0] (C00, C01)
                    
                    # M6: (c-a)(e+f)
                    # u6 = [-1, 0, 1, 0]
                    # v6 = [1, 1, 0, 0]
                    # w6 = [0, 0, 0, 1] (C11)
                    
                    # M7: (b-d)(g+h)
                    # u7 = [0, 1, 0, -1]
                    # v7 = [0, 0, 1, 1]
                    # w7 = [1, 0, 0, -1] (C00, C11)

                    strassen_terms = [
                        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
                        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
                        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
                        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
                        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
                        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])
                    ]
                    
                    for u_loc, v_loc, w_loc in strassen_terms:
                        # Add to global matrices
                        for li in range(2):
                            for lj in range(2):
                                if u_loc[li*2 + lj] != 0:
                                    u_entries.append(((i0+li)*m + (j0+lj), col_idx, u_loc[li*2 + lj]))
                        for lj in range(2):
                            for lk in range(2):
                                if v_loc[lj*2 + lk] != 0:
                                    v_entries.append(((j0+lj)*p + (k0+lk), col_idx, v_loc[lj*2 + lk]))
                        for lk in range(2):
                            for li in range(2):
                                if w_loc[lk*2 + li] != 0:
                                    w_entries.append(((k0+lk)*n + (i0+li), col_idx, w_loc[lk*2 + li]))
                        col_idx += 1

                else:
                    # Standard multiplication
                    for li in range(ni):
                        for lj in range(nj):
                            for lk in range(nk):
                                u_entries.append(((i0+li)*m + (j0+lj), col_idx, 1))
                                v_entries.append(((j0+lj)*p + (k0+lk), col_idx, 1))
                                w_entries.append(((k0+lk)*n + (i0+li), col_idx, 1))
                                col_idx += 1

    rank = col_idx
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, c, val in u_entries:
        U[r][c] = val
    for r, c, val in v_entries:
        V[r][c] = val
    for r, c, val in w_entries:
        W[r][c] = val

    # Verify
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to standard
        U, V, W = construct_fallback(n, m, p)
        rank = n * m * p

    return {"U": U, "V": V, "W": W}

def construct_fallback(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W
