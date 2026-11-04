"""
Rank-minimal tensor decomposition for 2x3 x 3x7 matrix multiplication.

Strategy:
  1. Construct an explicit rank-39 decomposition using Strassen's 7-multiplication
     algorithm for the 2x2 sub-blocks, combined with straightforward multiplication
     for the remaining terms.
  2. Verify the decomposition.
  3. Attempt a local-search refinement to reduce rank from 39 to 38 (and below).

The interface returns {'U', 'V', 'W'} with integer entries.
"""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140  # safety margin before 150s kill

    # ---------- Schoolbook baseline (rank 42) ----------
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # ---------- Verification ----------
    def verify_fast(U, V, W, rank):
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    # ---------- Strassen-based rank-39 construction ----------
    def strassen_rank39():
        """
        Partition A (2x3) into [A_22, A_21] where A_22 is 2x2 and A_21 is 2x1.
        Partition B (3x7) into [B_1, B_2, B_3, B_4] where B_1,B_2,B_3 are 3x2
        and B_4 is 3x1.

        Each B_i for i=1,2,3 has form [B_i22 (2x2); B_i12 (1x2)].
        B_4 has form [B_421 (2x1); B_411 (1x1)].

        For i in {1,2,3}:
          C_i = A_22 * B_i22 + A_21 * B_i12
          A_22 * B_i22: Strassen, 7 mults
          A_21 * B_i12: 4 mults (2x1 times 1x2)
          Total: 11 per block

        For i=4:
          C_4 = A_22 * B_421 + A_21 * B_411
          A_22 * B_421: 4 mults
          A_21 * B_411: 2 mults
          Total: 6

        Grand total: 3*11 + 6 = 39
        """
        rank = 39
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        col = 0

        # Index helpers
        # U row: i*m + j  (i in 0..1, j in 0..2)
        # V row: j*p + k  (j in 0..2, k in 0..6)
        # W row: k*n + i  (k in 0..6, i in 0..1)

        # A_22: rows i=0,1; cols j=0,1  -> U indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # A_21: rows i=0,1; col j=2     -> U indices: (0,2)=4, (1,2)=5
        # B cols: k=0,1 for B1; k=2,3 for B2; k=4,5 for B3; k=6 for B4
        # B_22 part: rows j=0,1; B_12 part: row j=2

        # Strassen multipliers for 2x2:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-f)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        # C11 = m1+m4-m5+m7, C12 = m3+m5, C21 = m2+m4, C22 = m1-m2+m3+m6

        # A_22 = [[a, b], [c, d]] with:
        #   a = U[0][col], b = U[1][col], c = U[2][col], d = U[3][col]
        # B_22 = [[e, f], [g, h]] with:
        #   e = V[jp+k0][col], f = V[jp+k1][col], g = V[jp+1+k0][col], h = V[jp+1+k1][col]

        # Strassen coefficients on U (for a,b,c,d) and V (for e,f,g,h) and W (for C entries)
        # We'll define the Strassen contribution as a list of (U_coeffs, V_coeffs, W_coeffs)
        # where each term is a rank-1 component.

        # For a 2x2 Strassen multiplication, the 7 rank-1 terms are:
        # Term 1: (a+d) * (e+h) -> U: [1,0,1,0] on (a,b,c,d), V: [1,1,1,1] on (e,f,g,h)
        #         contributes to W: C11 +=1, C22 +=1
        # Term 2: (c+d) * e     -> U: [0,0,1,1], V: [1,0,0,0]
        #         contributes to W: C21 +=1
        # Term 3: a * (f-h)     -> U: [1,0,0,0], V: [0,1,0,-1]
        #         contributes to W: C12 +=1
        # Term 4: d * (g-f)     -> U: [0,0,0,1], V: [0,-1,1,0]
        #         contributes to W: C11 +=1, C21 +=1
        # Term 5: (a+b) * h     -> U: [1,1,0,0], V: [0,0,0,1]
        #         contributes to W: C11 -=1, C12 +=1
        # Term 6: (c-a) * (e+f) -> U: [-1,0,1,0], V: [1,1,0,0]
        #         contributes to W: C22 +=1
        # Term 7: (b-d) * (g+h) -> U: [0,1,0,-1], V: [0,0,1,1]
        #         contributes to W: C11 +=1

        # W indices for 2x2 block at (i_base, k_base):
        # C11 -> W[k_base*n + i_base]
        # C12 -> W[(k_base+1)*n + i_base]
        # C21 -> W[k_base*n + (i_base+1)]
        # C22 -> W[(k_base+1)*n + (i_base+1)]

        def add_strassen_term(col_idx, i_base, k_base, u_coeffs, v_coeffs, w_contribs):
            """Add one Strassen rank-1 term.
            u_coeffs: [a, b, c, d] values for U rows (i_base+j) for j=0,1
            v_coeffs: [e, f, g, h] values for V rows at columns k_base, k_base+1
            w_contribs: dict mapping (dr, dc) -> coeff where dr,dc are offsets into 2x2
            """
            # U: rows (i_base, 0), (i_base, 1), (i_base+1, 0), (i_base+1, 1)
            u_rows = [i_base * m + 0, i_base * m + 1, (i_base + 1) * m + 0, (i_base + 1) * m + 1]
            for idx, coeff in enumerate(u_coeffs):
                if coeff != 0:
                    U[u_rows[idx]][col_idx] += coeff

            # V: rows (0, k_base), (0, k_base+1), (1, k_base), (1, k_base+1)
            v_rows = [0 * p + k_base, 0 * p + k_base + 1, 1 * p + k_base, 1 * p + k_base + 1]
            for idx, coeff in enumerate(v_coeffs):
                if coeff != 0:
                    V[v_rows[idx]][col_idx] += coeff

            # W contributions
            for (dr, dc), coeff in w_contribs.items():
                if coeff != 0:
                    w_row = (k_base + dc) * n + (i_base + dr)
                    W[w_row][col_idx] += coeff

        # Strassen term definitions:
        # (u_coeffs, v_coeffs, w_contribs)
        strassen_terms = [
            ([1, 0, 1, 0], [1, 1, 1, 1], {(0, 0): 1, (1, 1): 1}),       # m1
            ([0, 0, 1, 1], [1, 0, 0, 0], {(1, 0): 1}),                   # m2
            ([1, 0, 0, 0], [0, 1, 0, -1], {(0, 1): 1}),                  # m3
            ([0, 0, 0, 1], [0, -1, 1, 0], {(0, 0): 1, (1, 0): 1}),       # m4
            ([1, 1, 0, 0], [0, 0, 0, 1], {(0, 0): -1, (0, 1): 1}),       # m5
            ([-1, 0, 1, 0], [1, 1, 0, 0], {(1, 1): 1}),                  # m6
            ([0, 1, 0, -1], [0, 0, 1, 1], {(0, 0): 1}),                  # m7
        ]

        # 2x1 * 1x2 multiplication: 4 terms
        # A_21 = [a2; b2], B_12 = [c2, d2]
        # Result: [[a2*c2, a2*d2], [b2*c2, b2*d2]]
        # 4 rank-1 terms:
        # a2*c2 -> (0,0), a2*d2 -> (0,1), b2*c2 -> (1,0), b2*d2 -> (1,1)
        def add_21_12_terms(col_start, i_base, k_base):
            """Add 4 terms for A_21 (2x1) * B_12 (1x2)."""
            # A_21: rows (i_base, 2) and (i_base+1, 2)
            # B_12: row (2, k_base) and (2, k_base+1)
            terms = [
                # (u_row, v_row, w_contribs)
                (i_base * m + 2, 2 * p + k_base, {(0, 0): 1}),
                (i_base * m + 2, 2 * p + k_base + 1, {(0, 1): 1}),
                ((i_base + 1) * m + 2, 2 * p + k_base, {(1, 0): 1}),
                ((i_base + 1) * m + 2, 2 * p + k_base + 1, {(1, 1): 1}),
            ]
            for offset, (u_row, v_row, w_contribs) in enumerate(terms):
                c = col_start + offset
                U[u_row][c] += 1
                V[v_row][c] += 1
                for (dr, dc), coeff in w_contribs.items():
                    w_row = (k_base + dc) * n + (i_base + dr)
                    W[w_row][c] += coeff

        # 2x2 * 2x1 multiplication: 4 terms
        # A_22 (2x2) * B_21 (2x1) = 2x1 result
        # A_22 = [[a,b],[c,d]], B_21 = [e; g]
        # Result: [a*e + b*g; c*e + d*g]
        # 4 rank-1 terms:
        def add_22_21_terms(col_start, i_base, k_base):
            """Add 4 terms for A_22 (2x2) * B_21 (2x1)."""
            # B_21: rows (0, k_base) and (1, k_base)
            terms = [
                # (u_row, v_row, w_row)
                (i_base * m + 0, 0 * p + k_base, (k_base) * n + i_base),
                (i_base * m + 1, 1 * p + k_base, (k_base) * n + i_base),
                ((i_base + 1) * m + 0, 0 * p + k_base, (k_base) * n + (i_base + 1)),
                ((i_base + 1) * m + 1, 1 * p + k_base, (k_base) * n + (i_base + 1)),
            ]
            for offset, (u_row, v_row, w_row) in enumerate(terms):
                c = col_start + offset
                U[u_row][c] += 1
                V[v_row][c] += 1
                W[w_row][c] += 1

        # 2x1 * 1x1 multiplication: 2 terms
        def add_21_11_terms(col_start, i_base, k_base):
            """Add 2 terms for A_21 (2x1) * B_11 (1x1)."""
            terms = [
                (i_base * m + 2, 2 * p + k_base, k_base * n + i_base),
                ((i_base + 1) * m + 2, 2 * p + k_base, k_base * n + (i_base + 1)),
            ]
            for offset, (u_row, v_row, w_row) in enumerate(terms):
                c = col_start + offset
                U[u_row][c] += 1
                V[v_row][c] += 1
                W[w_row][c] += 1

        # Now assemble all 39 terms
        # i_base = 0 for all (since n=2, we use both rows)
        i_base = 0

        # Block 1: B_1 occupies k=0,1 (2 columns)
        k_base = 0
        # Strassen for A_22 * B_122 (7 terms)
        for st in strassen_terms:
            add_strassen_term(col, i_base, k_base, st[0], st[1], st[2])
            col += 1
        # 2x1 * 1x2 for A_21 * B_112 (4 terms)
        add_21_12_terms(col, i_base, k_base)
        col += 4

        # Block 2: B_2 occupies k=2,3
        k_base = 2
        for st in strassen_terms:
            add_strassen_term(col, i_base, k_base, st[0], st[1], st[2])
            col += 1
        add_21_12_terms(col, i_base, k_base)
        col += 4

        # Block 3: B_3 occupies k=4,5
        k_base = 4
        for st in strassen_terms:
            add_strassen_term(col, i_base, k_base, st[0], st[1], st[2])
            col += 1
        add_21_12_terms(col, i_base, k_base)
        col += 4

        # Block 4: B_4 occupies k=6 (1 column)
        k_base = 6
        # 2x2 * 2x1 for A_22 * B_421 (4 terms)
        add_22_21_terms(col, i_base, k_base)
        col += 4
        # 2x1 * 1x1 for A_21 * B_411 (2 terms)
        add_21_11_terms(col, i_base, k_base)
        col += 2

        assert col == 39, f"Expected 39 columns, got {col}"
        return {"U": U, "V": V, "W": W}

    # Verify and use the Strassen construction
    strassen_result = strassen_rank39()
    if verify_fast(strassen_result["U"], strassen_result["V"], strassen_result["W"], 39):
        best = strassen_result
        best_rank = 39

    # ---------- Local search to try reducing rank ----------
    def local_search(U, V, W, target_rank, max_iters=200):
        """Local search to refine a candidate."""
        U_work = [row[:] for row in U]
        V_work = [row[:] for row in V]
        W_work = [row[:] for row in W]

        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if verify_fast(U_work, V_work, W_work, target_rank):
                return True
            # Pick a random position to modify
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, nm - 1)
                col = random.randint(0, target_rank - 1)
                old = U_work[row][col]
                U_work[row][col] = random.randint(-2, 2)
                if not verify_fast(U_work, V_work, W_work, target_rank) and time.time() > deadline - 5:
                    U_work[row][col] = old
            elif mat == 'V':
                row = random.randint(0, mp - 1)
                col = random.randint(0, target_rank - 1)
                old = V_work[row][col]
                V_work[row][col] = random.randint(-2, 2)
                if not verify_fast(U_work, V_work, W_work, target_rank) and time.time() > deadline - 5:
                    V_work[row][col] = old
            else:
                row = random.randint(0, pn - 1)
                col = random.randint(0, target_rank - 1)
                old = W_work[row][col]
                W_work[row][col] = random.randint(-2, 2)
                if not verify_fast(U_work, V_work, W_work, target_rank) and time.time() > deadline - 5:
                    W_work[row][col] = old
        return verify_fast(U_work, V_work, W_work, target_rank)

    def try_rank_with_local_search(target_rank, max_attempts=50):
        """Try to find a decomposition with local search refinement."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None

            # Start from the Strassen rank-39 solution and zero out some columns
            # to get a candidate at target_rank, then refine
            if best_rank > target_rank:
                U0 = [row[:target_rank] for row in best["U"]]
                V0 = [row[:target_rank] for row in best["V"]]
                W0 = [row[:target_rank] for row in best["W"]]
            else:
                U0 = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(nm)]
                V0 = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(mp)]
                W0 = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(pn)]

            if local_search(U0, V0, W0, target_rank, max_iters=30):
                return {"U": U0, "V": V0, "W": W0}
        return None

    # Try to reduce rank from current best
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 15:
            break

        result = try_rank_with_local_search(target_rank, max_attempts=20)
        if result:
            best = result
            best_rank = target_rank
            continue

    return best
# EVOLVE-BLOCK-END
