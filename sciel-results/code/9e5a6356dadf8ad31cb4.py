"""Matrix multiplication 6x6x6 via tensor product of Strassen(2x2, rank 7) and schoolbook(3x3, rank 27)."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6,6,6

    # Strassen tensor for 2x2 matrix multiplication: 7 terms.
    # Each term is (u_terms, v_terms, w_terms) where u_terms is a list of (i,j,coef)
    # for A (2x2), v_terms is (j,k,coef) for B (2x2), w_terms is (k,i,coef) for C (2x2).
    # Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # For 2x2: m=2, p=2, n=2.
    strassen = [
        # Term 0: M1 = (A00+A11)(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # Term 1: M2 = (A10+A11)B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,0,0),(1,0,1)]),
        # Term 2: M3 = A00(B10-B11)
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(1,0,1),(0,0,0)]),
        # Term 3: M4 = A11(B01-B00)
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,1,1),(1,1,0)]),
        # Term 4: M5 = (A00+A01)B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,1,1),(0,1,0)]),
        # Term 5: M6 = (A10-A00)(B00+B10)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(1,0,1),(0,0,0)]),
        # Term 6: M7 = (A01-A11)(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,1),(1,1,0)]),
    ]

    # Schoolbook tensor for 3x3: 27 terms, each (i,j,k)
    schoolbook = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]

    # Tensor product: rank = 7 * 27 = 189
    # Block structure: A (6x6) split into 2x2 blocks of 3x3 each.
    # A[I*3+i, J*3+j] = A_block[I,J][i,j]
    # Similarly for B and C.
    # For each Strassen term (su, sv, sw) and each schoolbook term (si, sj, sk):
    #   U term: for each (I,J,cu) in su: for each local (i,j): U[(I*3+i)*6 + (J*3+j)] += cu
    #   V term: for each (J,K,cv) in sv: for each local (j,k): V[(J*3+j)*6 + (K*3+k)] += cv
    #   W term: for each (K,I,cw) in sw: for each local (k,i): W[(K*3+k)*6 + (I*3+i)] += cw

    rank = 7 * 27
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]

    r = 0
    for (su, sv, sw) in strassen:
        for (si, sj, sk) in schoolbook:
            # Build U contributions
            u_dict = {}
            for (I, J, cu) in su:
                for li in range(3):
                    for lj in range(3):
                        gi = I * 3 + li
                        gj = J * 3 + lj
                        idx = gi * 6 + gj
                        u_dict[idx] = u_dict.get(idx, 0.0) + cu
            for idx, val in u_dict.items():
                U[idx][r] = val

            # Build V contributions
            v_dict = {}
            for (J, K, cv) in sv:
                for lj in range(3):
                    for lk in range(3):
                        gi = J * 3 + lj
                        gj = K * 3 + lk
                        idx = gi * 6 + gj
                        v_dict[idx] = v_dict.get(idx, 0.0) + cv
            for idx, val in v_dict.items():
                V[idx][r] = val

            # Build W contributions
            w_dict = {}
            for (K, I, cw) in sw:
                for lk in range(3):
                    for li in range(3):
                        gi = K * 3 + lk
                        gj = I * 3 + li
                        idx = gi * 6 + gj
                        w_dict[idx] = w_dict.get(idx, 0.0) + cw
            for idx, val in w_dict.items():
                W[idx][r] = val

            r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
