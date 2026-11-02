"""Program for matrix multiplication tensor decomposition 4x4 x 4x6."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # ---------- Schoolbook baseline: rank 96 ----------
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, rank, n, m, p):
        """T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}

    # ---------- Recursive Strassen for the 4x4 x 4x4 block (rank 49) ----------
    # 2x2 Strassen linear forms.
    # A_coeffs[alpha][u][v] = coefficient of A[u][v] in the alpha-th A-form
    A_coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[1, 0], [-1, 0]],
        [[0, 1], [0, -1]],
    ]
    B_coeffs = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[1, 0], [-1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ]
    # C_coeffs[u][v][alpha] = coefficient of M_alpha in C[u][v]
    C_coeffs = [
        [[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0]],
        [[0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]],
    ]

    # Build the 4x4 x 4x4 x 4x4 decomposition with 49 rank-1 terms.
    # A = [A00 A01; A10 A11], B = [B00 B01; B10 B11], C = [C00 C01; C10 C11]
    # Each block is 2x2.  Outer Strassen: M_beta = L_A_beta . L_B_beta.
    # Inner Strassen on each M_beta (a 2x2x2x2 product) -> 7 rank-1 terms.
    #
    # For outer index beta and inner index alpha, the rank-1 term is:
    #   (sum_{r,c} A_coeffs[alpha][r][c] * L_A_beta[r,c])
    # * (sum_{r,c} B_coeffs[alpha][r][c] * L_B_beta[r,c])
    # * (C_coeffs[u][v][alpha])   where (u,v) is the block position.
    #
    # L_A_beta[r,c] is the (r,c) entry of the 2x2 matrix of linear forms
    #   L_A_beta = sum_{(r,c)} A_coeffs[beta][r][c] * A_{r,c}
    # so L_A_beta[r,c] = sum_{(r',c')} A_coeffs[beta][r][c] * A_{r',c'}.
    #
    # Therefore the alpha-th inner A-form applied to L_A_beta is:
    #   sum_{r,c} A_coeffs[alpha][r][c] * L_A_beta[r,c]
    # = sum_{r,c} A_coeffs[alpha][r][c] * sum_{r',c'} A_coeffs[beta][r][c] * A_{r',c'}
    # = sum_{r',c'} (sum_{r,c} A_coeffs[alpha][r][c] * A_coeffs[beta][r][c]) * A_{r',c'}
    #
    # So the coefficient of A_{r',c'} (the (r',c') block of A) in the
    # (beta, alpha) term is  sum_{r,c} A_coeffs[alpha][r][c] * A_coeffs[beta][r][c].

    # Precompute the combined coefficients.
    # combA[beta][alpha][r'][c'] = sum_{r,c} A_coeffs[alpha][r][c]*A_coeffs[beta][r][c]
    combA = [[[0.0]*2 for _ in range(2)] for _ in range(7) for _ in range(7)]
    combB = [[[0.0]*2 for _ in range(2)] for _ in range(7) for _ in range(7)]
    for beta in range(7):
        for alpha in range(7):
            for rp in range(2):
                for cp in range(2):
                    sa = 0.0
                    sb = 0.0
                    for r in range(2):
                        for c in range(2):
                            sa += A_coeffs[alpha][r][c] * A_coeffs[beta][r][c]
                            sb += B_coeffs[alpha][r][c] * B_coeffs[beta][r][c]
                    combA[beta][alpha][rp][cp] = sa
                    combB[beta][alpha][rp][cp] = sb

    # 4x4 block layout: block (u,v) occupies rows u*2..u*2+1, cols v*2..v*2+1
    # A_{r',c'} block -> A indices: row in {2*u+r'... } actually A is 4x4,
    # block (u,v) has A[2*u + rr, 2*v + cc] for rr,cc in 0..1.
    # The 49 terms fill U444 (16 x 49), V444 (16 x 49), W444 (16 x 49).

    U444 = [[0.0]*49 for _ in range(16)]
    V444 = [[0.0]*49 for _ in range(16)]
    W444 = [[0.0]*49 for _ in range(16)]

    for beta in range(7):
        for alpha in range(7):
            col_idx = beta * 7 + alpha
            # For each block position (u,v) in C:
            for u in range(2):
                for v in range(2):
                    c_coeff = C_coeffs[u][v][alpha]
                    if c_coeff == 0:
                        continue
                    # W entry for block (u,v): the (u,v) block of C is C[2*u+rr, 2*v+cc]
                    for rr in range(2):
                        for cc in range(2):
                            W444[(2*u + rr)*4 + (2*v + cc)][col_idx] = c_coeff
                    # U entry: block (u,v) of A -> A[2*u+rr, 2*v+cc], coefficient combA
                    for rr in range(2):
                        for cc in range(2):
                            U444[(2*u + rr)*4 + (2*v + cc)][col_idx] = combA[beta][alpha][rr][cc]
                    # V entry: block (u,v) of B -> B[2*u+rr, 2*v+cc], coefficient combB
                    for rr in range(2):
                        for cc in range(2):
                            V444[(2*u + rr)*4 + (2*v + cc)][col_idx] = combB[beta][alpha][rr][cc]

    # Verify the 4x4x4 block in isolation (T[i,j,k] for i,j,k in 0..3).
    def verify_block(U444, V444, W444, rank):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0.0
                    for c in range(rank):
                        s += U444[i*4+j][c] * V444[j*4+k][c] * W444[k*4+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    block_ok = verify_block(U444, V444, W444, 49)

    # ---------- Full 4x4 x 4x6 decomposition ----------
    # Split the 4x6 output into k in 0..3 (use 4x4x4 Strassen, rank 49)
    # and k in 4..5 (use schoolbook 4x4x2, rank 32).
    # Total rank = 49 + 32 = 81.
    rank_total = 81
    Uf = [[0.0]*rank_total for _ in range(n*m)]   # 16 x 81
    Vf = [[0.0]*rank_total for _ in range(m*p)]   # 24 x 81
    Wf = [[0.0]*rank_total for _ in range(p*n)]   # 24 x 81

    # Part 1: k in 0..3 (4x4x4 Strassen).  k index maps to W column k*n+i,
    # but W444 is indexed by k*4+i (k in 0..3).  In the full Wf, the row for
    # (k, i) with k in 0..3 is row k*n + i = k*4 + i (since n=4).  So it lines up.
    for i in range(16):
        for c in range(49):
            Uf[i][c] = U444[i][c]
            Vf[i][c] = V444[i][c]
            Wf[i][c] = W444[i][c]

    # Part 2: k in 4..5 (schoolbook 4x4x2, rank 32).
    col = 49
    for i in range(n):
        for j in range(m):
            for k in range(4, 6):
                Uf[i*m + j][col] = 1
                Vf[j*p + k][col] = 1
                Wf[k*n + i][col] = 1
                col += 1

    candidate = {'U': Uf, 'V': Vf, 'W': Wf, 'rank': rank_total}
    if verify(Uf, Vf, Wf, rank_total, n, m, p):
        best = candidate

    # Safety: ensure we always return a valid result.
    if not verify(best['U'], best['V'], best['W'], best['rank'], n, m, p):
        # fall back to schoolbook
        best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}

    return best
# EVOLVE-BLOCK-END
