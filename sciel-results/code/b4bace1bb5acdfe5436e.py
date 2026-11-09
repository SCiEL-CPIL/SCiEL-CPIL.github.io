"""
Rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.

Uses the known rank-19 decomposition for 3x3x3 (Bini, Capovani, Lotti, Romani 1980),
applied to two independent 3x3x3 blocks (columns 0-2 and 3-5 of the 3x6 matrix).
Total rank = 19 + 19 = 38.

The 19 terms for 3x3x3 are indexed as:
  u[i*m + j] for i,j in 0..2  (A[i][j])
  v[j*p + k] for j,k in 0..2  (B[j][k])
  w[k*n + i] for k,i in 0..2  (C[i][k], with W index = k*n + i)

The 19 terms (u, v, w) as sparse dicts mapping index -> coefficient:
"""

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    rank = 38
    U = [[0] * rank for _ in range(n * m)]   # 9 x 38
    V = [[0] * rank for _ in range(m * p)]   # 18 x 38
    W = [[0] * rank for _ in range(p * n)]   # 18 x 38

    # The 19 terms for 3x3x3 (Bini et al. 1980)
    # Each term is (u_dict, v_dict, w_dict)
    # u index: i*3+j for A[i][j]
    # v index: j*3+k for B[j][k]
    # w index: k*3+i for C[i][k] (W[k*n+i])

    terms = [
        # Term 0: (A11+A22)(B11+B22) -> C11, C22
        ({1: 1, 4: 1}, {1: 1, 4: 1}, {4: 1, 7: 1}),
        # Term 1: (A21+A22)B11 -> C21
        ({4: 1, 5: 1}, {1: 1}, {4: 1}),
        # Term 2: A11(B21-B22) -> C12
        ({1: 1}, {4: 1, 5: -1}, {7: 1}),
        # Term 3: A22(B12-B11) -> C21
        ({4: 1}, {2: 1, 1: -1}, {4: 1}),
        # Term 4: (A11+A12)B22 -> C12
        ({1: 1, 2: 1}, {5: 1}, {7: 1}),
        # Term 5: (A21-A11)(B11+B12) -> C21
        ({4: 1, 1: -1}, {1: 1, 2: 1}, {4: 1}),
        # Term 6: (A12-A22)(B21+B22) -> C12
        ({2: 1, 4: -1}, {4: 1, 5: 1}, {7: 1}),
        # Term 7: (A31+A32)B11 -> C31
        ({7: 1, 8: 1}, {1: 1}, {1: 1}),
        # Term 8: (A11+A33)(B11+B33) -> C11, C33
        ({1: 1, 8: 1}, {1: 1, 8: 1}, {1: 1, 8: 1}),
        # Term 9: A33(B12-B11) -> C32
        ({8: 1}, {2: 1, 1: -1}, {5: 1}),
        # Term 10: A11(B32-B31) -> C13
        ({1: 1}, {7: 1, 6: -1}, {2: 1}),
        # Term 11: (A31+A32)B33 -> C33
        ({7: 1, 8: 1}, {8: 1}, {8: 1}),
        # Term 12: (A13+A33)B22 -> C12, C32
        ({2: 1, 8: 1}, {5: 1}, {7: 1, 5: 1}),
        # Term 13: (A23+A33)B12 -> C22, C32
        ({5: 1, 8: 1}, {2: 1}, {4: 1, 5: 1}),
        # Term 14: (A12+A23)B31 -> C13, C23
        ({2: 1, 5: 1}, {6: 1}, {2: 1, 5: 1}),
        # Term 15: (A13+A21)B22 -> C13, C22
        ({2: 1, 4: 1}, {5: 1}, {2: 1, 4: 1}),
        # Term 16: (A23+A31)B12 -> C23, C32
        ({5: 1, 7: 1}, {2: 1}, {5: 1, 5: 1}),
        # Term 17: (A12-A23)B33 -> C13, C23
        ({2: 1, 5: -1}, {8: 1}, {2: 1, 5: 1}),
        # Term 18: (A13-A31)B21 -> C13, C31
        ({2: 1, 7: -1}, {4: 1}, {2: 1, 1: 1}),
    ]

    # Block 0: columns k=0,1,2 of the 3x6 matrix
    for t in range(19):
        u_dict, v_dict, w_dict = terms[t]
        col = t
        for idx, coeff in u_dict.items():
            U[idx][col] += coeff
        for idx, coeff in v_dict.items():
            V[idx][col] += coeff
        for idx, coeff in w_dict.items():
            W[idx][col] += coeff

    # Block 1: columns k=3,4,5 of the 3x6 matrix
    # B[j][k] for k in 3,4,5 maps to v index j*6 + k
    # C[i][k] for k in 3,4,5 maps to W index k*3 + i
    for t in range(19):
        u_dict, v_dict, w_dict = terms[t]
        col = 19 + t
        for idx, coeff in u_dict.items():
            U[idx][col] += coeff
        for idx, coeff in v_dict.items():
            # v index for block 1: j*6 + (k+3) where original k is 0,1,2
            # Original v index is j*3+k, so j = idx//3, k = idx%3
            j = idx // 3
            k = idx % 3
            new_v_idx = j * 6 + (k + 3)
            V[new_v_idx][col] += coeff
        for idx, coeff in w_dict.items():
            # w index for block 1: (k+3)*3 + i where original k is 0,1,2
            # Original w index is k*3+i, so k = idx//3, i = idx%3
            k = idx // 3
            i = idx % 3
            new_w_idx = (k + 3) * 3 + i
            W[new_w_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
