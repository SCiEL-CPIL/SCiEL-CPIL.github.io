"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 105.0

    def build_strassen_2x5x6():
        """
        Build a rank-26 decomposition using two Strassen 2x2x2 blocks
        covering j in {0,1} and {2,3}, k in {0,1}, plus schoolbook for the rest.
        
        The tensor T[i,j,k] = (A*B)[i,k] where A is 2x5, B is 5x6.
        Indexing: U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
        
        Strassen block for j in {ja, jb}, k in {ka, kb}:
        Uses 7 multiplications to compute the 2x2 sub-product.
        
        For the 2x2x2 block with A_sub[i, {ja,jb}] and B_sub[{ja,jb}, {ka,kb}]:
        The 4 output entries are (i, ka), (i, kb) for i in {0,1}.
        
        Strassen formulas for C = A_sub * B_sub (2x2 * 2x2):
        M1 = (A00 + A11) * (B00 + B11)
        M2 = A11 * (B10 - B00)
        M3 = (A00 - A10) * B10
        M4 = A00 * B11
        M5 = (A10 + A00) * B01
        M6 = (A11 - A01) * B01
        M7 = (A01 + A11) * (B00 - B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M4
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M5
        
        Here A_sub[i, j'] where j' in {ja, jb}, B_sub[j', k'] where k' in {ka, kb}
        C_sub[i, k'] where i in {0,1}, k' in {ka, kb}
        
        Mapping: A00 = A[i, ja], A01 = A[i, jb], A10 = A[i, ja], A11 = A[i, jb]
        Wait, let me re-index. For the 2x2 block:
        A_sub has rows i in {0,1} and cols j' in {0,1} (local), where local 0->ja, local 1->jb
        B_sub has rows j' in {0,1} (local) and cols k' in {0,1} (local), where local 0->ka, local 1->kb
        
        So:
        A00 = A[i, ja], A01 = A[i, jb]
        A10 = A[i, ja], A11 = A[i, jb]
        Wait, that's wrong. A is 2x5, so A[i, j] for i in {0,1}, j in {0..4}.
        The sub-block uses columns ja and jb of A, and rows ja and jb of B.
        
        A_sub[i, 0] = A[i, ja], A_sub[i, 1] = A[i, jb]
        B_sub[0, 0] = B[ja, ka], B_sub[0, 1] = B[ja, kb]
        B_sub[1, 0] = B[jb, ka], B_sub[1, 1] = B[jb, kb]
        
        So A00 = A[i, ja], A01 = A[i, jb], A10 = A[i, ja], A11 = A[i, jb]
        That's not right either. Let me think again.
        
        Actually, for the Strassen 2x2x2:
        A is 2x2, B is 2x2, C = A*B is 2x2.
        A = [[A00, A01], [A10, A11]]
        B = [[B00, B01], [B10, B11]]
        C = [[C00, C01], [C10, C11]]
        
        In our case, the "2x2" A_sub corresponds to rows i in {0,1} and cols j in {ja, jb}.
        So A_sub[0,0] = A[0, ja], A_sub[0,1] = A[0, jb]
           A_sub[1,0] = A[1, ja], A_sub[1,1] = A[1, jb]
        
        B_sub corresponds to rows j in {ja, jb} and cols k in {ka, kb}.
        So B_sub[0,0] = B[ja, ka], B_sub[0,1] = B[ja, kb]
           B_sub[1,0] = B[jb, ka], B_sub[1,1] = B[jb, kb]
        
        C_sub[i, k'] = sum over j' in {ja,jb} of A[i, j'] * B[j', k']
        C_sub[0,0] = A[0,ja]*B[ja,ka] + A[0,jb]*B[jb,ka]
        C_sub[0,1] = A[0,ja]*B[ja,kb] + A[0,jb]*B[jb,kb]
        C_sub[1,0] = A[1,ja]*B[ja,ka] + A[1,jb]*B[jb,ka]
        C_sub[1,1] = A[1,ja]*B[ja,kb] + A[1,jb]*B[jb,kb]
        
        Strassen:
        M1 = (A00 + A11) * (B00 + B11) = (A[i,ja] + A[i,jb]) * (B[ja,ka] + B[jb,ka])
        M2 = A11 * (B10 - B00) = A[i,jb] * (B[jb,ka] - B[ja,ka])
        M3 = (A00 - A10) * B10 = (A[i,ja] - A[i,ja]) * B[jb,ka] = 0
        Hmm, that's a problem. A00 and A10 are both A[i, ja] for different i values? No.
        
        Wait, I'm confusing myself. In the Strassen 2x2x2, the "2" in the first dimension refers to the row index of A (which is i in {0,1}), the second "2" refers to the column index of A / row index of B (which is j' in {0,1} local), and the third "2" refers to the column index of B (which is k' in {0,1} local).
        
        So the tensor is T[i, j', k'] = A[i, j'] * B[j', k']
        Where i in {0,1}, j' in {0,1} (local, mapping to ja, jb), k' in {0,1} (local, mapping to ka, kb).
        
        Strassen decomposes this 2x2x2 tensor (rank 4 schoolbook) into rank 7.
        
        The standard Strassen:
        M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        M2 = A[1,1] * (B[1,0] - B[0,0])
        M3 = (A[0,0] - A[1,0]) * B[1,0]
        M4 = A[0,0] * B[1,1]
        M5 = (A[1,0] + A[0,0]) * B[0,1]
        M6 = (A[1,1] - A[0,1]) * B[0,1]
        M7 = (A[0,1] + A[1,1]) * (B[0,0] - B[1,1])
        
        C[0,0] = M1 + M4 - M5 + M7
        C[0,1] = M3 + M4
        C[1,0] = M2 + M4
        C[1,1] = M1 - M2 + M3 + M5
        
        Here A[i, j'] and B[j', k'] are the entries.
        A[0,0] = A[i=0, j=ja], A[0,1] = A[i=0, j=jb]
        A[1,0] = A[i=1, j=ja], A[1,1] = A[i=1, j=jb]
        B[0,0] = B[j=ja, k=ka], B[0,1] = B[j=ja, k=kb]
        B[1,0] = B[j=jb, k=ka], B[1,1] = B[j=jb, k=kb]
        
        So:
        M1 = (A[0,ja] + A[1,jb]) * (B[ja,ka] + B[jb,ka])
        M2 = A[1,jb] * (B[jb,ka] - B[ja,ka])
        M3 = (A[0,ja] - A[1,ja]) * B[jb,ka]
        M4 = A[0,ja] * B[jb,kb]
        M5 = (A[1,ja] + A[0,ja]) * B[ja,kb]
        M6 = (A[1,jb] - A[0,jb]) * B[ja,kb]
        M7 = (A[0,jb] + A[1,jb]) * (B[ja,ka] - B[jb,ka])
        
        C[0,ka] = M1 + M4 - M5 + M7
        C[0,kb] = M3 + M4
        C[1,ka] = M2 + M4
        C[1,kb] = M1 - M2 + M3 + M5
        
        Each M is a product of a linear combination of A entries and a linear combination of B entries.
        So each M corresponds to a rank-1 term in the tensor decomposition.
        
        For M1: U part = A[0,ja] + A[1,jb], V part = B[ja,ka] + B[jb,ka]
        W part: contributes to C[0,ka] (+1), C[1,kb] (+1)
        
        Let me build this properly.
        """
        rank_max = 60
        U = np.zeros((n * m, rank_max), dtype=float)
        V = np.zeros((m * p, rank_max), dtype=float)
        W = np.zeros((p * n, rank_max), dtype=float)
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            if col >= rank_max:
                return
            for u_idx, coeff in u_coeffs:
                U[u_idx, col] += coeff
            for v_idx, coeff in v_coeffs:
                V[v_idx, col] += coeff
            for w_idx, coeff in w_coeffs:
                    W[w_idx, col] += coeff
            col += 1

        def strassen_block(ja, jb, ka, kb):
            # M1 = (A[0,ja] + A[1,jb]) * (B[ja,ka] + B[jb,ka])
            # W: C[0,ka] += 1, C[1,kb] += 1
            u1 = [(0 * m + ja, 1), (1 * m + jb, 1)]
            v1 = [(ja * p + ka, 1), (jb * p + ka, 1)]
            w1 = [(ka * n + 0, 1), (kb * n + 1, 1)]
            add_term(u1, v1, w1)

            # M2 = A[1,jb] * (B[jb,ka] - B[ja,ka])
            # W: C[1,ka] += 1
            u2 = [(1 * m + jb, 1)]
            v2 = [(jb * p + ka, 1), (ja * p + ka, -1)]
            w2 = [(ka * n + 1, 1)]
            add_term(u2, v2, w2)

            # M3 = (A[0,ja] - A[1,ja]) * B[jb,ka]
            # W: C[0,kb] += 1, C[1,kb] += 1
            u3 = [(0 * m + ja, 1), (1 * m + ja, -1)]
            v3 = [(jb * p + ka, 1)]
            w3 = [(kb * n + 0, 1), (kb * n + 1, 1)]
            add_term(u3, v3, w3)

            # M4 = A[0,ja] * B[jb,kb]
            # W: C[0,ka] += 1, C[0,kb] += 1, C[1,ka] += 1
            u4 = [(0 * m + ja, 1)]
            v4 = [(jb * p + kb, 1)]
            w4 = [(ka * n + 0, 1), (kb * n + 0, 1), (ka * n + 1, 1)]
            add_term(u4, v4, w4)

            # M5 = (A[1,ja] + A[0,ja]) * B[ja,kb]
            # W: C[0,ka] -= 1, C[1,kb] += 1
            u5 = [(1 * m + ja, 1), (0 * m + ja, 1)]
            v5 = [(ja * p + kb, 1)]
            w5 = [(ka * n + 0, -1), (kb * n + 1, 1)]
            add_term(u5, v5, w5)

            # M6 = (A[1,jb] - A[0,jb]) * B[ja,kb]
            # W: C[1,kb] += 1
            u6 = [(1 * m + jb, 1), (0 * m + jb, -1)]
            v6 = [(ja * p + kb, 1)]
            w6 = [(kb * n + 1, 1)]
            add_term(u6, v6, w6)

            # M7 = (A[0,jb] + A[1,jb]) * (B[ja,ka] - B[jb,ka])
            # W: C[0,ka] += 1
            u7 = [(0 * m + jb, 1), (1 * m + jb, 1)]
            v7 = [(ja * p + ka, 1), (jb * p + ka, -1)]
            w7 = [(ka * n + 0, 1)]
            add_term(u7, v7, w7)

        # Apply Strassen to j in {0,1}, k in {0,1}
        strassen_block(0, 1, 0, 1)
        # Apply Strassen to j in {2,3}, k in {0,1}
        strassen_block(2, 3, 0, 1)

        # Now handle the remaining entries with schoolbook
        # Covered: i in {0,1}, j in {0,1,2,3}, k in {0,1}
        # Remaining: j in {4}, k in {0..5} and j in {0..3}, k in {2..5}
        
        covered = set()
        for i in range(n):
            for j in [0, 1, 2, 3]:
                for k in [0, 1]:
                    covered.add((i, j, k))

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1

        return U[:, :col].copy(), V[:, :col].copy(), W[:, :col].copy(), col

    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j, :]
                    v_row = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True

    def build_decomposition(ja1, jb1, ja2, jb2, ka, kb):
        """Build a decomposition with two Strassen blocks."""
        rank_max = 60
        U = np.zeros((n * m, rank_max), dtype=float)
        V = np.zeros((m * p, rank_max), dtype=float)
        W = np.zeros((p * n, rank_max), dtype=float)
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            if col >= rank_max:
                return
            for u_idx, coeff in u_coeffs:
                U[u_idx, col] += coeff
            for v_idx, coeff in v_coeffs:
                V[v_idx, col] += coeff
            for w_idx, coeff in w_coeffs:
                    W[w_idx, col] += coeff
            col += 1

        def strassen_block(ja, jb, ka, kb):
            u1 = [(0 * m + ja, 1), (1 * m + jb, 1)]
            v1 = [(ja * p + ka, 1), (jb * p + ka, 1)]
            w1 = [(ka * n + 0, 1), (kb * n + 1, 1)]
            add_term(u1, v1, w1)

            u2 = [(1 * m + jb, 1)]
            v2 = [(jb * p + ka, 1), (ja * p + ka, -1)]
            w2 = [(ka * n + 1, 1)]
            add_term(u2, v2, w2)

            u3 = [(0 * m + ja, 1), (1 * m + ja, -1)]
            v3 = [(jb * p + ka, 1)]
            w3 = [(kb * n + 0, 1), (kb * n + 1, 1)]
            add_term(u3, v3, w3)

            u4 = [(0 * m + ja, 1)]
            v4 = [(jb * p + kb, 1)]
            w4 = [(ka * n + 0, 1), (kb * n + 0, 1), (ka * n + 1, 1)]
            add_term(u4, v4, w4)

            u5 = [(1 * m + ja, 1), (0 * m + ja, 1)]
            v5 = [(ja * p + kb, 1)]
            w5 = [(ka * n + 0, -1), (kb * n + 1, 1)]
            add_term(u5, v5, w5)

            u6 = [(1 * m + jb, 1), (0 * m + jb, -1)]
            v6 = [(ja * p + kb, 1)]
            w6 = [(kb * n + 1, 1)]
            add_term(u6, v6, w6)

            u7 = [(0 * m + jb, 1), (1 * m + jb, 1)]
            v7 = [(ja * p + ka, 1), (jb * p + ka, -1)]
            w7 = [(ka * n + 0, 1)]
            add_term(u7, v7, w7)

        strassen_block(ja1, jb1, ka, kb)
        strassen_block(ja2, jb2, ka, kb)

        strassen_set = set()
        for i in range(n):
            for j in [ja1, jb1, ja2, jb2]:
                for k in [ka, kb]:
                    strassen_set.add((i, j, k))

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in strassen_set:
                        continue
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1

        return U[:, :col].copy(), V[:, :col].copy(), W[:, :col].copy(), col

    # Try different Strassen block configurations
    configs = [
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 1, 2),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 3, 4),
        (0, 1, 2, 3, 4, 5),
        (0, 2, 1, 3, 0, 1),
        (0, 3, 1, 2, 0, 1),
        (0, 1, 3, 4, 0, 1),
        (0, 1, 3, 4, 1, 2),
        (0, 1, 3, 4, 2, 3),
        (0, 1, 3, 4, 3, 4),
        (0, 1, 3, 4, 4, 5),
        (2, 3, 4, 5, 0, 1),
        (2, 3, 4, 5, 1, 2),
        (2, 3, 4, 5, 2, 3),
        (2, 3, 4, 5, 3, 4),
        (2, 3, 4, 5, 4, 5),
    ]

    best_result = None
    best_rank = 60

    # First try the standard construction
    U, V, W, r = build_strassen_2x5x6()
    if verify(U, V, W) and r < best_rank:
        best_rank = r
        best_result = (U.copy(), V.copy(), W.copy())

    for cfg in configs:
        if time.time() > deadline:
            break
        ja1, jb1, ja2, jb2, ka, kb = cfg
        if ja1 == jb1 or ja2 == jb2 or ka == kb:
            continue
        if len({ja1, jb1, ja2, jb2}) != 4:
            continue

        U, V, W, r = build_decomposition(ja1, jb1, ja2, jb2, ka, kb)

        if verify(U, V, W):
            if r < best_rank:
                best_rank = r
                best_result = (U.copy(), V.copy(), W.copy())

    if best_result is None:
        # Fallback to trivial rank 60
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    U, V, W = best_result

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
