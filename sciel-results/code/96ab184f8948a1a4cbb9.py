"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication using Strassen blocks."""
import time

START_TIME = time.time()
TIME_LIMIT = 115.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        pass  # Should not trigger; construction is O(1)

    n, m, p = 2, 4, 6
    rank_total = 42
    U = [[0] * rank_total for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank_total for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank_total for _ in range(p * n)]   # 12 x 42

    # Strassen 2x2@2x2 rank-7 decomposition
    # Products P1..P7:
    # P1=(a11+a22)(b11+b22), P2=(a21+a22)b11, P3=a11(b12-b22),
    # P4=a22(b21-b11), P5=(a11+a12)b22, P6=(a21-a11)(b11+b12), P7=(a12-a22)(b21+b22)
    # Output: c11=P1+P4-P5+P7, c12=P3+P5, c21=P2+P4, c22=P1-P2+P3+P6

    strassen_U = [
        {0: 1, 3: 1},   # P1: a11+a22
        {2: 1, 3: 1},   # P2: a21+a22
        {0: 1},         # P3: a11
        {3: 1},         # P4: a22
        {0: 1, 1: 1},   # P5: a11+a12
        {0: -1, 2: 1},  # P6: a21-a11
        {1: 1, 3: -1},  # P7: a12-a22
    ]
    strassen_V = [
        {0: 1, 3: 1},   # b11+b22
        {0: 1},         # b11
        {1: 1, 3: -1},  # b12-b22
        {2: 1, 0: -1},  # b21-b11
        {3: 1},         # b22
        {0: 1, 1: 1},   # b11+b12
        {2: 1, 3: 1},   # b21+b22
    ]
    strassen_W = [
        {0: 1, 3: 1},   # P1 -> c11, c22
        {2: 1, 3: -1},  # P2 -> c21, -c22
        {1: 1, 3: 1},   # P3 -> c12, c22
        {0: 1, 2: 1},   # P4 -> c11, c21
        {0: -1, 1: 1},  # P5 -> -c11, c12
        {3: 1},         # P6 -> c22
        {0: 1},         # P7 -> c11
    ]

    col_offset = 0
    for j_block in range(3):
        for side in ['L', 'R']:
            u_col_off = 0 if side == 'L' else 2
            v_row_off = 0 if side == 'L' else 2

            for prod_idx in range(7):
                col = col_offset + prod_idx

                # U: local (i, j_local) -> global i*4 + (u_col_off + j_local)
                for local_idx, sign in strassen_U[prod_idx].items():
                    i = local_idx // 2
                    j_loc = local_idx % 2
                    U[i * 4 + u_col_off + j_loc][col] += sign

                # V: local (k_local, c_local) -> global (v_row_off+k_local)*6 + (2*j_block+c_local)
                for local_idx, sign in strassen_V[prod_idx].items():
                    k_loc = local_idx // 2
                    c_loc = local_idx % 2
                    V[(v_row_off + k_loc) * 6 + 2 * j_block + c_loc][col] += sign

                # W: local (c_local, i) -> global (2*j_block+c_local)*2 + i
                for local_idx, sign in strassen_W[prod_idx].items():
                    c_loc = local_idx // 2
                    i = local_idx % 2
                    W[(2 * j_block + c_loc) * 2 + i][col] += sign

            col_offset += 7

    return {"U": U, "V": V, "W": W}
