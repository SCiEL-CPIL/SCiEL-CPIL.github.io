import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2 base
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Strassen 4x4x4 specs (P0..P6)
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]
    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]

    # Directly iterate over the 2 half-slices, 7 outer Strassen terms, 7 inner terms
    for h in range(2):
        for k in range(7):
            for s in range(7):
                idx = (h * 7 + k) * 7 + s
                # Build U, V, W columns for this index
                # U: 16 x 1
                u_col = np.zeros(16)
                for ca, bidx in A_specs[k]:
                    bi, bj = bidx // 2, bidx % 2
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            u_col[gr * 4 + gc] += ca * U2[r * 2 + c, s]
                U[:, idx] = u_col

                # V: 32 x 1
                v_col = np.zeros(32)
                for cb, bidx in B_specs[k]:
                    bi, bj = bidx // 2, bidx % 2
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            gk = gc + h * 4
                            v_col[gr * 8 + gk] += cb * V2[r * 2 + c, s]
                V[:, idx] = v_col

                # W: 32 x 1
                w_col = np.zeros(32)
                for cc, cidx in C_specs[k]:
                    bi, bj = cidx // 2, cidx % 2
                    for r in range(2):
                        for c in range(2):
                            gi = bi * 2 + r
                            gk = bj * 2 + c + h * 4
                            w_col[gk * 4 + gi] += cc * W2[r * 2 + c, s]
                W[:, idx] = w_col

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
