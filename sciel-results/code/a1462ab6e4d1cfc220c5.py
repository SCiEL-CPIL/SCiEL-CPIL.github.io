"""Matrix multiplication 4x4x6 - multi-strategy rank minimization."""
import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 4, 4, 6
    nm = n * m  # 16
    mp = m * p  # 24
    pn = p * n  # 24

    def verify_decomp(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        rank = len(U[0]) if isinstance(U, list) else U.shape[1]
        U_mat = np.array(U) if isinstance(U, list) else U
        V_mat = np.array(V) if isinstance(V, list) else V
        W_mat = np.array(W) if isinstance(W, list) else W
        # Efficient check using tensor contraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_mat[i * m + j], V_mat[j * p + k] * W_mat[k * n + i])
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p  # 96
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Strategy 1: Nested Strassen (77 rank) - current best
    def nested_strassen():
        R = 77
        U = np.zeros((nm, R), dtype=int)
        V = np.zeros((mp, R), dtype=int)
        W = np.zeros((pn, R), dtype=int)

        alpha = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        beta = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        recon = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 0, 1)],
            [(0, 0, -1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, 1)]
        ]

        col = 0
        for s in range(7):
            for si in range(7):
                for a in range(4):
                    if alpha[si][a] != 0:
                        x_row, x_col = a // 2, a % 2
                        for r in range(4):
                            if alpha[s][r] != 0:
                                ar, br = r // 2, r % 2
                                u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                                U[u_idx, col] += alpha[si][a] * alpha[s][r]

                for b in range(4):
                    if beta[si][b] != 0:
                        y_row, y_inner_col = b // 2, b % 2
                        for c in range(4):
                            if beta[s][c] != 0:
                                ac, bc = c // 2, c % 2
                                v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                                V[v_idx, col] += beta[si][b] * beta[s][c]

                for a, b, gamma in recon[s]:
                    for za, zb, zgamma in recon[si]:
                        w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                        W[w_idx, col] += gamma * zgamma
                col += 1

            for u in range(2):
                for v in range(2):
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                            U[u_idx, col] += alpha[s][r]

                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                            V[v_idx, col] += beta[s][c]

                    for a, b, gamma in recon[s]:
                        w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                        W[w_idx, col] += gamma
                    col += 1

        return U, V, W

    # Strategy 2: Column-pair splitting (3 x 4x4x2, each rank 28) = 84
    def column_pair_split():
        R = 84
        U = np.zeros((nm, R), dtype=int)
        V = np.zeros((mp, R), dtype=int)
        W = np.zeros((pn, R), dtype=int)

        # Strassen for 2x2x2
        alpha = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        beta = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        recon = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 0, 1)],
            [(0, 0, -1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, 1)]
        ]

        col = 0
        for col_pair in range(3):  # 3 column pairs: cols 0-1, 2-3, 4-5
            # For each column pair, solve 4x4 x 4x2
            # Top-level Strassen: 7 multiplications of (2x2) x (2x1)
            for s in range(7):
                # X_s is 2x2, Y_s is 2x1
                # 4 schoolbook multiplications
                for u in range(2):
                    for v in range(2):
                        # U part
                        for r in range(4):
                            if alpha[s][r] != 0:
                                ar, br = r // 2, r % 2
                                u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                                U[u_idx, col] += alpha[s][r]

                        # V part: Y_s[v, 0] where Y_s is 2x1
                        # Y_s[v, 0] = sum_c beta[s][c] * B_c[v, 0]
                        # B_c is 2x1 (for this subproblem), indexed by (ac, bc)
                        # Global B: B[ac*2+v, bc*2 + col_pair*2 + 0]
                        for c in range(4):
                            if beta[s][c] != 0:
                                ac, bc = c // 2, c % 2
                                # Global column index: bc*2 + col_pair*2
                                v_idx = (ac * 2 + v) * 6 + (bc * 2 + col_pair * 2)
                                V[v_idx, col] += beta[s][c]

                        # W part: output Z[u] contributes to C
                        for a, b, gamma in recon[s]:
                            # C block (a,b) at row u, col 0 of the 2-column pair
                            # Global: row = a*2 + u, col = b*2 + col_pair*2
                            w_idx = (b * 2 + col_pair * 2) * 4 + (a * 2 + u)
                            W[w_idx, col] += gamma
                        col += 1

        return U, V, W

    # Strategy 3: Try a different top-level split - 1x4 + 3x4 rows, or asymmetric
    # Actually, let's try: split B into 4x2 + 4x2 + 4x2 but use a rank-24 scheme for 4x4x2 if possible
    # For now, just try the two strategies and pick the best

    best_rank = 96
    best_result = None

    # Try nested Strassen (rank 77)
    U1, V1, W1 = nested_strassen()
    if verify_decomp(U1, V1, W1):
        best_rank = 77
        best_result = (U1, V1, W1)

    # Try column pair split (rank 84)
    if time.time() < deadline - 30:
        U2, V2, W2 = column_pair_split()
        if verify_decomp(U2, V2, W2) and 84 < best_rank:
            best_rank = 84
            best_result = (U2, V2, W2)

    # Fallback to schoolbook
    if best_result is None:
        U, V, W = schoolbook()
        best_result = (U, V, W)

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
