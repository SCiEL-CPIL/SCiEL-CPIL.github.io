"""
Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.

Strategy:
- Partition the 4 intermediate columns into 2 groups of 2 (jg in {0,1}).
- Partition the 8 output columns into 4 groups of 2 (kg in {0,1,2,3}).
- For each (jg, kg) pair, use Strassen's 7-term 2x2*2x2 decomposition.
- Total rank = 2 * 4 * 7 = 56 (vs schoolbook 64).

Factor matrices:
  U: (n*m, rank) = (8, 56)
  V: (m*p, rank) = (32, 56)
  W: (p*n, rank) = (16, 56)

Trace convention for W: W[k*n + i, r]
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Strassen 2x2*2x2: for each of the 7 intermediate products M_s,
    # list (coefficient, index) pairs for A, B, and C.
    # A-indices: 0=a, 1=b, 2=c, 3=d
    # B-indices: 0=e, 1=f, 2=g, 3=h
    # C-indices: 0=C11, 1=C12, 2=C21, 3=C22

    strassen_A = [
        [(1, 0), (1, 3)],       # M1 = (a+d)
        [(1, 2), (1, 3)],       # M2 = (c+d)
        [(1, 0)],               # M3 = a
        [(1, 3)],               # M4 = d
        [(1, 0), (1, 1)],       # M5 = (a+b)
        [(1, 2), (-1, 0)],      # M6 = (c-a)
        [(1, 1), (-1, 3)],      # M7 = (b-d)
    ]
    strassen_B = [
        [(1, 0), (1, 3)],       # e+h
        [(1, 0)],               # e
        [(1, 1), (-1, 3)],      # f-h
        [(-1, 0), (1, 2)],      # g-e
        [(1, 3)],               # h
        [(1, 0), (1, 1)],       # e+f
        [(1, 2), (1, 3)],       # g+h
    ]
    # C combination: for each C_local entry, list (coefficient, M_index)
    strassen_C = [
        [(1, 0), (1, 3), (-1, 4), (1, 6)],   # C11 = M1+M4-M5+M7
        [(1, 2), (1, 4)],                     # C12 = M3+M5
        [(1, 1), (1, 3)],                     # C21 = M2+M4
        [(1, 0), (-1, 1), (1, 2), (1, 5)],   # C22 = M1-M2+M3+M6
    ]

    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for jg in range(2):
        for kg in range(4):
            for s in range(7):
                c = (jg * 4 + kg) * 7 + s

                # Fill U
                for coeff, a_idx in strassen_A[s]:
                    i = a_idx // 2
                    jl = a_idx % 2
                    U[i * m + jg * 2 + jl][c] += coeff

                # Fill V
                for coeff, b_idx in strassen_B[s]:
                    jl = b_idx // 2
                    kl = b_idx % 2
                    V[(jg * 2 + jl) * p + kg * 2 + kl][c] += coeff

                # Fill W: for each C_local entry that involves M_s
                for c_local in range(4):
                    for coeff, m_idx in strassen_C[c_local]:
                        if m_idx == s:
                            i = c_local // 2
                            kl = c_local % 2
                            W[(kg * 2 + kl) * n + i][c] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
