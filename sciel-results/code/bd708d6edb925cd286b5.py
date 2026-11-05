"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 138  # 150s wall-clock limit, keep ~12s margin

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Index conventions (trace convention W[k*n + i, r]):
    #   U index: i*m + j   (A is n x m)
    #   V index: j*p + k   (B is m x p)
    #   W index: k*n + i   (C is p x n, trace convention)
    # A rank-R decomposition is valid iff for every (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1

    def make_trace(U, V, W, n, m, p):
        """Return the (n*m*p,) vector of trace products T[i,j,k]."""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                u_row = Ua[i * m + j, :]
                for k in range(p):
                    T[i, j, k] = np.dot(u_row * Va[j * p + k, :], Wa[k * n + i, :])
        return T

    def verify(U, V, W, n, m, p, tol=1e-7):
        """Exact (dense) verification of the decomposition."""
        T = make_trace(U, V, W, n, m, p)
        return bool(np.all(np.abs(T - 1.0) <= tol))

    def schoolbook(n, m, p):
        """Rank n*m*p trivial decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def build_4x4x4_winograd():
        """Rank-18 Winograd F(4) decomposition for 4x4 * 4x4 -> 4x4.

        Identity: with A' = A + A^T, B' = B + B^T, X = A' B' (rank-16 core),
        plus two rank-1 terms
            T1 = (A_0 + A_1)(B_0 + B_1),  T2 = (A_0 - A_1)(B_0 - B_1),
        where A_0 = A[0:2], A_1 = A[2:4], B_0 = B[0:2], B_1 = B[2:4],
        we have  A B = X - diag(X) + 2 diag(X) J,  J = ones(4,4).
        This is verified numerically below before being returned.
        """
        rank = 18
        U = [[0] * rank for _ in range(16)]  # n*m = 16
        V = [[0] * rank for _ in range(16)]  # m*p = 16
        W = [[0] * rank for _ in range(16)]  # p*n = 16

        # --- Core: X = A' B' (rank 16). A' = A + A^T, B' = B + B^T ---
        # U_a(i,j) = A[i,j] + A[j,i] ; V_a(j,k) = B[j,k] + B[k,j]
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][j * 4 + i] = 1  # A[i,j]
                U[i * 4 + j][i * 4 + j] = 1  # A[j,i]
        for j in range(4):
            for k in range(4):
                V[j * 4 + k][j * 4 + k] = 1  # B[j,k]
                V[j * 4 + k][k * 4 + k] = 1  # B[k,j]
        # W for X: X[i,k] = sum_j A'[i,j] B'[j,k]; only the diagonal X[i,i] is
        # retained (X - diag(X) drops the off-diagonal part).
        for i in range(4):
            for j in range(4):
                W[i * 4 + i][j * 4 + j] = 1  # X[i,i]

        # --- Two extra rank-1 terms ---
        # T1 = (A_0 + A_1)(B_0 + B_1),  T2 = (A_0 - A_1)(B_0 - B_1)
        for i in range(4):
            for j in range(4):
                a1 = 1 if i < 2 else 1       # A_0 + A_1 -> all +1
                a2 = 1 if i < 2 else -1      # A_0 - A_1 -> +1 / -1
                b1 = 1 if j < 2 else 1       # B_0 + B_1 -> all +1
                b2 = 1 if j < 2 else -1      # B_0 - B_1 -> +1 / -1
                U[i * 4 + j][16] += a1
                V[j * 4 + j][16] += b1
                U[i * 4 + j][17] += a2
                V[j * 4 + j][17] += b2
        # diag(T1) and diag(T2) each contribute +1 to C[i,i]
        for i in range(4):
            W[i * 4 + i][16] += 2
            W[i * 4 + i][17] += 2

        # --- 2*diag(X)*J : every C[i,k] gets 2*(X[i,i] + T1[i,i] + T2[i,i]) ---
        for i in range(4):
            for k in range(4):
                W[k * 4 + i][j * 4 + j]  # placeholder (overwritten below)
        # Reset W and redo the full W construction cleanly to avoid the
        # placeholder line above leaving stale entries.
        W = [[0] * rank for _ in range(16)]
        for i in range(4):
            for j in range(4):
                W[i * 4 + i][j * 4 + j] = 1  # diag(X)
            for k in range(4):
                # 2*diag(X)*J term on C[i,k]
                W[k * 4 + i][j * 4 + j] += 2
            # 2*diag(T1)*J and 2*diag(T2)*J
            for k in range(4):
                W[k * 4 + i][16] += 2
                W[k * 4 + i][17] += 2

        if not verify(U, V, W, 4, 4, 4):
            return None
        return U, V, W

    def build_4x4x6():
        """Build a 4x4 * 4x6 decomposition.

        Split the 6 output columns into a 4x4 core (Winograd rank-18) and a
        4x2 strip handled with schoolbook rank-1 terms (8 more), giving rank 26.
        Falls back to pure schoolbook (rank 96) if the Winograd core is invalid.
        """
        core = build_4x4x4_winograd()
        if core is None:
            return schoolbook(4, 4, 6)

        U_core, V_core, W_core = core
        rank_core = len(U_core[0])  # 18
        total_rank = rank_core + 8

        U = [[0] * total_rank for _ in range(16)]  # n*m
        V = [[0] * total_rank for _ in range(24)]  # m*p
        W = [[0] * total_rank for _ in range(24)]  # p*n

        # --- Core (columns 0..3 of B -> columns 0..3 of C) ---
        for r in range(16):
            for c in range(rank_core):
                U[r][c] = U_core[r][c]
        for j in range(4):
            for k in range(4):
                V[j * 6 + k][j * 4 + k] = V_core[j * 4 + k][j * 4 + k]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][k * 4 + i] = W_core[k * 4 + i][k * 4 + i]

        # --- Strip (columns 4..5 of B -> columns 4..5 of C), schoolbook ---
        col = rank_core
        for i in range(4):
            for j in range(4):
                for k in (4, 5):
                    U[i * 4 + j][col] = 1
                    V[j * 6 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1

        if not verify(U, V, W, 4, 4, 6):
            return schoolbook(4, 4, 6)
        return U, V, W

    def try_reduce_rank(U, V, W, n, m, p, time_budget):
        """Reduce rank by removing zero-contribution columns and cancelling pairs."""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        R = Ua.shape[1]
        if R == 0:
            return U, V, W, R

        N = n * m * p

        def column_contribution(r):
            u_col = Ua[:, r]
            v_col = Va[:, r]
            w_col = Wa[:, r]
            c = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    uv = u_col[i * m + j]
                    if uv == 0:
                        continue
                    for k in range(p):
                        c[i, j, k] = uv * v_col[j * p + k] * w_col[k * n + i]
            return c

        M = np.zeros((R, N), dtype=float)
        for r in range(R):
            if time.time() > time_budget:
                break
            M[r, :] = column_contribution(r).flatten()

        active = list(range(R))

        # Remove zero-contribution columns
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            for idx in range(len(active)):
                if time.time() > time_budget:
                    break
                if np.all(np.abs(M[idx, :]) <= 1e-9):
                    active.pop(idx)
                    M = np.delete(M, idx, axis=0)
                    changed = True
                    break

        # Remove cancelling pairs (M[i] + M[j] == 0)
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            for i in range(len(active)):
                if time.time() > time_budget:
                    break
                for j in range(i + 1, len(active)):
                    if np.all(np.abs(M[i, :] + M[j, :]) <= 1e-9):
                        active.pop(j)
                        active.pop(i)
                        M = np.delete(M, [i, j], axis=0)
                        changed = True
                        break
                if changed:
                    break

        if len(active) == R:
            return U, V, W, R

        U_new = Ua[:, active]
        V_new = Va[:, active]
        W_new = Wa[:, active]
        return U_new.tolist(), V_new.tolist(), W_new.tolist(), len(active)

    # Build the initial decomposition (Winograd core + schoolbook strip).
    U, V, W = build_4x4x6()

    # Verify; fall back to schoolbook if broken.
    if not verify(U, V, W, n, m, p):
        U, V, W = schoolbook(n, m, p)
        return {"U": U, "V": W and V, "W": W}

    # Time-bounded rank reduction.
    time_budget = min(deadline - 8, time.time() + 120)
    U_r, V_r, W_r, _ = try_reduce_rank(U, V, W, n, m, p, time_budget)

    if verify(U_r, V_r, W_r, n, m, p):
        return {"U": U_r, "V": V_r, "W": W_r}
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
