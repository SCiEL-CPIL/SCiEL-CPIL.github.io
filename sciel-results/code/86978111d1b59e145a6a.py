import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 5

    # Strassen 2x2x2 blocks: 4 non-overlapping blocks covering i in {0,1}
    blocks = [
        (0, 0, 0),  # i=0,1; j=0,1; k=0,1
        (0, 2, 0),  # i=0,1; j=2,3; k=0,1
        (0, 0, 2),  # i=0,1; j=0,1; k=2,3
        (0, 2, 2),  # i=0,1; j=2,3; k=2,3
    ]

    n_strassen = len(blocks)
    n_covered = n_strassen * 8
    n_schoolbook = n * m * p - n_covered
    total_rank = n_strassen * 7 + n_schoolbook  # 28 + 43 = 71

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i0, j0, k0, col_offset):
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7

    # Mark covered triples
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True

    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1

    # Verify with numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)

    if not np.allclose(T_decomp, target, atol=1e-10):
        # Fallback: pure schoolbook
        rank_s = n * m * p
        U_s = [[0] * rank_s for _ in range(n * m)]
        V_s = [[0] * rank_s for _ in range(m * p)]
        W_s = [[0] * rank_s for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c] = 1
                    V_s[j*p+k][c] = 1
                    W_s[k*n+i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    # Attempt limited search for rank reduction by trying alternative block placements
    if time.time() < deadline - 20:
        # Try all valid 4-block combinations (non-overlapping 2x2x2 blocks in 3x5x5)
        valid_positions = []
        for i0 in range(n - 1):
            for j0 in range(m - 1):
                for k0 in range(p - 1):
                    valid_positions.append((i0, j0, k0))

        def blocks_non_overlapping(blocks_list):
            for a in range(len(blocks_list)):
                for b in range(a + 1, len(blocks_list)):
                    i0a, j0a, k0a = blocks_list[a]
                    i0b, j0b, k0b = blocks_list[b]
                    if (abs(i0a - i0b) < 2 and abs(j0a - j0b) < 2 and abs(k0a - k0b) < 2):
                        return False
            return True

        from itertools import combinations
        best_rank = total_rank
        best_result = (U, V, W)

        # Only try if we might find something better
        for combo in combinations(valid_positions, 4):
            if time.time() > deadline - 5:
                break
            if not blocks_non_overlapping(list(combo)):
                continue
            # Build and check
            n_sb = n * m * p - len(combo) * 8
            r = len(combo) * 7 + n_sb
            if r >= best_rank:
                continue
            U_t = [[0] * r for _ in range(n * m)]
            V_t = [[0] * r for _ in range(m * p)]
            W_t = [[0] * r for _ in range(p * n)]
            col_t = 0
            for i0, j0, k0 in combo:
                add_strassen_block_to(U_t, V_t, W_t, i0, j0, k0, col_t, n, m, p)
                col_t += 7
            cov = [[[False]*p for _ in range(m)] for _ in range(n)]
            for i0, j0, k0 in combo:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            cov[i0+di][j0+dj][k0+dk] = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if not cov[i][j][k]:
                            U_t[i*m+j][col_t] = 1
                            V_t[j*p+k][col_t] = 1
                            W_t[k*n+i][col_t] = 1
                            col_t += 1
            # Quick verify
            Ut = np.array(U_t, dtype=float)
            Vt = np.array(V_t, dtype=float)
            Wt = np.array(W_t, dtype=float)
            Td = np.einsum('ar,br,cr->abc', Ut, Vt, Wt)
            if np.allclose(Td, target, atol=1e-10):
                best_rank = r
                best_result = (U_t, V_t, W_t)

        if best_rank < total_rank:
            return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}

    return {"U": U, "V": V, "W": W}


def add_strassen_block_to(U, V, W, i0, j0, k0, col_offset, n, m, p):
    """Add a 2x2x2 Strassen block to existing matrices."""
    # M1 = (A00+A11)(B00+B11)
    c = col_offset + 0
    U[i0*m + j0][c] += 1
    U[(i0+1)*m + (j0+1)][c] += 1
    V[j0*p + k0][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1

    # M2 = (A10+A11)B00
    c = col_offset + 1
    U[(i0+1)*m + j0][c] += 1
    U[(i0+1)*m + (j0+1)][c] += 1
    V[j0*p + k0][c] += 1
    W[k0*n + (i0+1)][c] += 1
    W[(k0+1)*n + (i0+1)][c] += -1

    # M3 = A00(B01-B11)
    c = col_offset + 2
    U[i0*m + j0][c] += 1
    V[j0*p + (k0+1)][c] += 1
    V[(j0+1)*p + (k0+1)][c] += -1
    W[(k0+1)*n + i0][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1

    # M4 = A11(B10-B00)
    c = col_offset + 3
    U[(i0+1)*m + (j0+1)][c] += 1
    V[(j0+1)*p + k0][c] += 1
    V[j0*p + k0][c] += -1
    W[k0*n + i0][c] += 1
    W[k0*n + (i0+1)][c] += 1

    # M5 = (A00+A01)B11
    c = col_offset + 4
    U[i0*m + j0][c] += 1
    U[i0*m + (j0+1)][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += -1
    W[(k0+1)*n + i0][c] += 1

    # M6 = (A10-A00)(B00+B01)
    c = col_offset + 5
    U[(i0+1)*m + j0][c] += 1
    U[i0*m + j0][c] += -1
    V[j0*p + k0][c] += 1
    V[j0*p + (k0+1)][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1

    # M7 = (A01-A11)(B10+B11)
    c = col_offset + 6
    U[i0*m + (j0+1)][c] += 1
    U[(i0+1)*m + (j0+1)][c] += -1
    V[(j0+1)*p + k0][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += 1
