import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0


def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7

    def elapsed():
        return time.time() - start_time

    def check_valid(U, V, W, tol=1e-8):
        """Check if the decomposition is valid using vectorized operations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        U_t = U_arr.reshape(n, m, r)
        V_t = V_arr.reshape(m, p, r)
        W_t = W_arr.reshape(p, n, r)
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        except Exception:
            return False
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)

    def residual_vec(U, V, W):
        """Compute the residual vector."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        U_t = U_arr.reshape(n, m, r)
        V_t = V_arr.reshape(m, p, r)
        W_t = W_arr.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        target = np.ones((n, m, p))
        res = T - target
        return res.flatten()

    def build_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2_2x2():
        """Strassen's rank-7 decomposition for 2x2x2."""
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        # p1 = (a11+a22)(b11+b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # p2 = (a21+a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # p3 = a11(b12-b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # p4 = a22(b21-b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # p5 = (a11+a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # p6 = (a21-a11)(b11+b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # p7 = (a12-a22)(b21+b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def build_schoolbook_2x2x2x1():
        """Schoolbook decomposition for 2x2 * 2x1 -> 2x1, rank 4."""
        U = [[0] * 4 for _ in range(4)]
        V = [[0] * 4 for _ in range(2)]
        W = [[0] * 4 for _ in range(2)]
        col = 0
        for i in range(2):
            for j in range(2):
                U[i * 2 + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W

    def kron_decompose(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2, n, m, p):
        """Build Kronecker product decomposition."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 * r2
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        for r1_idx in range(r1):
            for r2_idx in range(r2):
                r = r1_idx * r2 + r2_idx
                for i1 in range(n1):
                    for j1 in range(m1):
                        for i2 in range(n2):
                            for j2 in range(m2):
                                i = i1 * n2 + i2
                                j = j1 * m2 + j2
                                val = U1[i1 * m1 + j1][r1_idx] * U2[i2 * m2 + j2][r2_idx]
                                if val != 0:
                                    U[i * m + j][r] += val
                for j1 in range(m1):
                    for k1 in range(p1):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                j = j1 * m2 + j2
                                k = k1 * p2 + k2
                                val = V1[j1 * p1 + k1][r1_idx] * V2[j2 * p2 + k2][r2_idx]
                                if val != 0:
                                    V[j * p + k][r] += val
                for k1 in range(p1):
                    for i1 in range(n1):
                        for k2 in range(p2):
                            for i2 in range(n2):
                                k = k1 * p2 + k2
                                i = i1 * n2 + i2
                                val = W1[k1 * n1 + i1][r1_idx] * W2[k2 * n2 + i2][r2_idx]
                                if val != 0:
                                    W[k * n + i][r] += val
        return U, V, W

    def build_4x4x4_strassen():
        """Build rank-49 decomposition for 4x4 * 4x4 using Kronecker of two Strassen."""
        n = m = p = 4
        U2, V2, W2 = build_strassen_2x2_2x2()
        U, V, W = kron_decompose(U2, V2, W2, U2, V2, W2, 2, 2, 2, 2, 2, 2, n, m, p)
        return U, V, W

    def build_4x4x2():
        """Build decomposition for 4x4 * 4x2 using Kronecker (Strassen_2x2 ⊗ schoolbook_2x2x2x1)."""
        n, m, p = 4, 4, 2
        U1, V1, W1 = build_strassen_2x2_2x2()
        U2, V2, W2 = build_schoolbook_2x2x2x1()
        U, V, W = kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 2, 2, 2, 1, n, m, p)
        return U, V, W

    def build_4x4x1():
        """Build decomposition for 4x4 * 4x1 using Kronecker (schoolbook_2x2x2x1 ⊗ schoolbook_2x2x2x1)."""
        n, m, p = 4, 4, 1
        U1, V1, W1 = build_schoolbook_2x2x2x1()
        U2, V2, W2 = build_schoolbook_2x2x2x1()
        U, V, W = kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 1, 2, 2, 1, n, m, p)
        return U, V, W

    def build_4x4x7_rank93():
        """
        Build 4x4x7 decomposition with rank 93 by concatenating:
        Part 1: 4x4 * 4x4 (B cols 0-3) -> rank 49 (Strassen Kronecker)
        Part 2: 4x4 * 4x2 (B cols 4-5) -> rank 28 (Strassen ⊗ schoolbook)
        Part 3: 4x4 * 4x1 (B col 6)    -> rank 16 (schoolbook Kronecker)
        Total: 93
        """
        n, m, p = 4, 4, 7

        part1 = build_4x4x4_strassen()
        part2 = build_4x4x2()
        part3 = build_4x4x1()

        r1 = len(part1[0])  # 49
        r2 = len(part2[0])  # 28
        r3 = len(part3[0])  # 16
        total_rank = r1 + r2 + r3  # 93

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        # Part 1: cols 0-3 of B
        for i in range(n * m):
            for r in range(r1):
                U[i][r] = part1[0][i][r]
        for j in range(4):
            for k in range(4):
                for r in range(r1):
                    V[j * p + k][r] = part1[1][j * 4 + k][r]
        for k in range(4):
            for i in range(n):
                for r in range(r1):
                    W[k * n + i][r] = part1[2][k * 4 + i][r]

        # Part 2: cols 4-5 of B
        off = r1
        for i in range(n * m):
            for r in range(r2):
                U[i][off + r] = part2[0][i][r]
        for j in range(4):
            for k in range(2):
                for r in range(r2):
                    V[j * p + (4 + k)][off + r] = part2[1][j * 2 + k][r]
        for k in range(2):
            for i in range(n):
                for r in range(r2):
                    W[(4 + k) * n + i][off + r] = part2[2][k * 2 + i][r]

        # Part 3: col 6 of B
        off = r1 + r2
        for i in range(n * m):
            for r in range(r3):
                U[i][off + r] = part3[0][i][r]
        for j in range(4):
            for r in range(r3):
                V[j * p + 6][off + r] = part3[1][j][r]
        for i in range(n):
            for r in range(r3):
                W[6 * n + i][off + r] = part3[2][i][r]

        return U, V, W

    def tensors_to_vec(U, V, W):
        """Flatten the three factor matrices into one vector."""
        r = len(U[0])
        u = np.array(U, dtype=float).flatten()
        v = np.array(V, dtype=float).flatten()
        w = np.array(W, dtype=float).flatten()
        return np.concatenate([u, v, w])

    def vec_to_tensors(x, r):
        """Unflatten a vector into three factor matrices."""
        u_len = n * m * r
        v_len = m * p * r
        U = x[:u_len].reshape(n * m, r)
        V = x[u_len:u_len + v_len].reshape(m * p, r)
        W = x[u_len + v_len:].reshape(p * n, r)
        return U, V, W

    def objective(x, r):
        """Objective function for minimization: squared residual."""
        U, V, W = vec_to_tensors(x, r)
        res = residual_vec(U, V, W)
        return np.dot(res, res)

    def rank_sparse_search(U0, V0, W0, r0, target_rank, max_iter=500):
        """
        Rank-sparse interpolation: try to find a rank-target_rank decomposition
        by projecting the rank-r0 solution and then refining.
        Uses multiple random projections and L-BFGS-B refinement.
        """
        if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
            return None

        cur_U = np.array(U0, dtype=float)
        cur_V = np.array(V0, dtype=float)
        cur_W = np.array(W0, dtype=float)

        for attempt in range(12):
            if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                break

            if attempt == 0:
                # Random linear combination
                M = np.random.randn(target_rank, r0)
                M /= np.sqrt(r0)
                U_new = cur_U @ M.T
                V_new = cur_V @ M.T
                W_new = cur_W @ M.T
            elif attempt == 1:
                # Keep first target_rank columns
                U_new = cur_U[:, :target_rank].copy()
                V_new = cur_V[:, :target_rank].copy()
                W_new = cur_W[:, :target_rank].copy()
            elif attempt == 2:
                # Random projection with structure preservation
                M = np.zeros((target_rank, r0))
                keep_count = min(target_rank // 2, r0)
                M[:keep_count, :keep_count] = np.eye(keep_count)
                if target_rank > keep_count:
                    M[keep_count:, :] = np.random.randn(target_rank - keep_count, r0)
                    M[keep_count:, :] /= np.sqrt(r0)
                U_new = cur_U @ M.T
                V_new = cur_V @ M.T
                W_new = cur_W @ M.T
            elif attempt == 3:
                # Gaussian perturbation of truncated
                U_new = cur_U[:, :target_rank].copy()
                V_new = cur_V[:, :target_rank].copy()
                W_new = cur_W[:, :target_rank].copy()
                U_new += 0.1 * np.random.randn(*U_new.shape)
                V_new += 0.1 * np.random.randn(*V_new.shape)
                W_new += 0.1 * np.random.randn(*W_new.shape)
            elif attempt == 4:
                # Keep last target_rank columns
                U_new = cur_U[:, r0 - target_rank:].copy()
                V_new = cur_V[:, r0 - target_rank:].copy()
                W_new = cur_W[:, r0 - target_rank:].copy()
            elif attempt == 5:
                # Mix: half from start, half from end
                half = target_rank // 2
                U_new = np.hstack([cur_U[:, :half], cur_U[:, r0 - (target_rank - half):]])
                V_new = np.hstack([cur_V[:, :half], cur_V[:, r0 - (target_rank - half):]])
                W_new = np.hstack([cur_W[:, :half], cur_W[:, r0 - (target_rank - half):]])
            elif attempt == 6:
                # Kronecker-biased projection
                M = np.zeros((target_rank, r0))
                kron_keep = min(49, target_rank)
                M[:kron_keep, :kron_keep] = np.eye(kron_keep) * 0.9
                if target_rank > kron_keep:
                    M[kron_keep:, :] = np.random.randn(target_rank - kron_keep, r0)
                    M[kron_keep:, :] /= np.sqrt(r0)
                U_new = cur_U @ M.T
                V_new = cur_V @ M.T
                W_new = cur_W @ M.T
            elif attempt == 7:
                # SVD-based projection
                # Treat the tensor as a matrix and project
                U_mat = cur_U.flatten()
                V_mat = cur_V.flatten()
                W_mat = cur_W.flatten()
                full = np.concatenate([U_mat, V_mat, W_mat])
                try:
                    U_svd, s, Vt = np.linalg.svd(full.reshape(-1, r0), full_matrices=False)
                    U_new = (cur_U.flatten() @ Vt[:target_rank].T).reshape(n*m, target_rank)
                    V_new = (cur_V.flatten() @ Vt[:target_rank].T).reshape(m*p, target_rank)
                    W_new = (cur_W.flatten() @ Vt[:target_rank].T).reshape(p*n, target_rank)
                except:
                    continue
            elif attempt == 8:
                # Sparse random: keep random subset of columns
                cols = np.random.choice(r0, size=target_rank, replace=False)
                U_new = cur_U[:, cols].copy()
                V_new = cur_V[:, cols].copy()
                W_new = cur_W[:, cols].copy()
            elif attempt == 9:
                # Double random: two random projections composed
                M1 = np.random.randn(target_rank, r0) / np.sqrt(r0)
                M2 = np.random.randn(target_rank, target_rank) / np.sqrt(target_rank)
                U_new = cur_U @ M1.T @ M2.T
                V_new = cur_V @ M1.T @ M2.T
                W_new = cur_W @ M1.T @ M2.T
            elif attempt == 10:
                # Perturbed SVD
                U_mat = cur_U.flatten()
                V_mat = cur_V.flatten()
                W_mat = cur_W.flatten()
                full = np.concatenate([U_mat, V_mat, W_mat])
                try:
                    U_svd, s, Vt = np.linalg.svd(full.reshape(-1, r0), full_matrices=False)
                    U_new = (cur_U.flatten() @ Vt[:target_rank].T).reshape(n*m, target_rank)
                    V_new = (cur_V.flatten() @ Vt[:target_rank].T).reshape(m*p, target_rank)
                    W_new = (cur_W.flatten() @ Vt[:target_rank].T).reshape(p*n, target_rank)
                    U_new += 0.05 * np.random.randn(*U_new.shape)
                    V_new += 0.05 * np.random.randn(*V_new.shape)
                    W_new += 0.05 * np.random.randn(*W_new.shape)
                except:
                    continue
            else:
                # Block-aware: keep Kronecker part + random for rest
                M = np.zeros((target_rank, r0))
                keep_49 = min(49, target_rank)
                M[:keep_49, :keep_49] = np.eye(keep_49)
                if target_rank > keep_49:
                    M[keep_49:, 49:] = np.random.randn(target_rank - keep_49, r0 - 49)
                    M[keep_49:, 49:] /= np.sqrt(max(1, r0 - 49))
                U_new = cur_U @ M.T
                V_new = cur_V @ M.T
                W_new = cur_W @ M.T

            x0 = tensors_to_vec(U_new.tolist(), V_new.tolist(), W_new.tolist())

            try:
                res = minimize(objective, x0, args=(target_rank,), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
                if res.fun < 1e-8:
                    U_out, V_out, W_out = vec_to_tensors(res.x, target_rank)
                    if check_valid(U_out.tolist(), V_out.tolist(), W_out.tolist(), tol=1e-8):
                        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
            except Exception:
                pass

        return None

    # ---- Main flow ----
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p

    np.random.seed(12345)

    # Step 1: Build rank-93 construction (49 + 28 + 16)
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_rank93()
            r_kron = len(U_kron[0])
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception:
            pass

    # Step 2: Try to reduce rank using rank-sparse interpolation
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN and best_rank > 60:
        cur_U = np.array(best_result["U"], dtype=float)
        cur_V = np.array(best_result["V"], dtype=float)
        cur_W = np.array(best_result["W"], dtype=float)
        cur_rank = len(cur_U[0])

        # Try reducing rank step by step
        for target_rank in range(cur_rank - 1, max(70, cur_rank - 15), -1):
            if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                break

            success = False
            result = rank_sparse_search(cur_U.tolist(), cur_V.tolist(), cur_W.tolist(),
                                        cur_rank, target_rank)
            if result is not None:
                best_result = result
                best_rank = target_rank
                cur_U = np.array(result["U"], dtype=float)
                cur_V = np.array(result["V"], dtype=float)
                cur_W = np.array(result["W"], dtype=float)
                success = True

            if not success and elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                break

    # Step 3: Final polish - snap to integers/half-integers
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            cur_U = np.array(best_result["U"], dtype=float)
            cur_V = np.array(best_result["V"], dtype=float)
            cur_W = np.array(best_result["W"], dtype=float)

            # Try snapping to nearest half-integer
            U_snap = np.round(cur_U * 2) / 2
            V_snap = np.round(cur_V * 2) / 2
            W_snap = np.round(cur_W * 2) / 2

            if check_valid(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), tol=1e-6):
                best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            else:
                U_int = np.round(cur_U)
                V_int = np.round(cur_V)
                W_int = np.round(cur_W)
                if check_valid(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=1e-6):
                    best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        except Exception:
            pass

    return best_result
