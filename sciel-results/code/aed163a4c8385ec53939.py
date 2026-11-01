import time
import numpy as np

DIMENSIONS = [2, 4, 8]


def _verify_fast(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    rank = len(U[0]) if U else 0
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)

    U_r = U_mat.reshape(n, m, rank)
    V_r = V_mat.reshape(m, p, rank)
    W_r = W_mat.reshape(p, n, rank)

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        factor_A = np.einsum('ij,ijc->ic', A, U_r)
        factor_B = np.einsum('jk,jkc->kc', B, V_r)
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_r)
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True


def _build_block_strassen():
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U_s = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[0,0,1,-1],[0,1,0,1]]
    V_s = [[1,0,0,1],[0,0,0,1],[1,0,-1,0],[0,-1,0,1],[0,0,1,0],[1,1,0,0],[0,0,1,1]]
    W_s = [[1,0,0,1,-1,0,1],[0,0,1,0,1,0,0],[0,1,0,0,0,1,0],[1,-1,1,0,0,1,0]]

    col_off = 0
    for block in range(4):
        for f in range(7):
            col = col_off + f
            for r in range(4):
                U[r, col] = U_s[f][r]
                j = r // 2
                k = 2 * block + r % 2
                V[j * 8 + k, col] = V_s[f][r]
                i = r // 2
                W[k * 2 + i, col] = W_s[r][f]
        col_off += 7

    for block in range(4):
        for f in range(7):
            col = col_off + f
            for r in range(4):
                U[2 + r, col] = U_s[f][r]
                j = 2 + r // 2
                k = 2 * block + r % 2
                V[j * 8 + k, col] = V_s[f][r]
                i = r // 2
                W[k * 2 + i, col] = W_s[r][f]
        col_off += 7

    return U.tolist(), V.tolist(), W.tolist()


def _tensor_image(U, V, W, n, m, p):
    """Return 64 x rank matrix where col c is the flattened tensor vector for that rank-1 component."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # Tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # For each c, the rank-1 tensor is U[:,:,c] * V[:,:,c] * W[:,:,c] (outer product)
    # Flatten: index = (i*p + k)*m + j  or we just need consistency for linear dependence.
    # Use flat index: i, j, k -> i*(m*p) + j*p + k
    T = np.zeros((n * m * p, rank), dtype=float)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m * p) + j * p + k
                    T[idx, c] = U_r[i, j, c] * V_r[j, k, c] * W_r[k, i, c]
    return T


def _greedy_nullspace_reduction(U, V, W, T, n, m, p, start_time, deadline):
    """
    Greedily try to remove columns of T that are linear combinations of others.
    A column c is removable if T[:, c] is in span of T[:, c' for c' != c].
    We check by seeing if nullspace of [T] has a vector with support only on c (i.e., removing c doesn't change the column space).
    Simpler: for each column c, check if rank(T) == rank(T without c). If yes, remove it.
    """
    rank = T.shape[1]
    keep = list(range(rank))
    U_cur = U[:, keep].copy()
    V_cur = V[:, keep].copy()
    W_cur = W[:, keep].copy()
    T_cur = T[:, keep].copy()

    changed = True
    while changed and (time.time() < deadline - 15):
        changed = False
        r = T_cur.shape[1]
        for c in range(r):
            if time.time() > deadline - 15:
                break
            # Check if column c is in span of others
            others = [x for x in range(r) if x != c]
            if len(others) == 0:
                continue
            T_others = T_cur[:, others]
            r_full = np.linalg.matrix_rank(T_cur)
            r_other = np.linalg.matrix_rank(T_others)
            if r_full == r_other:
                # Remove column c
                keep.pop(c)
                U_cur = U[:, keep].copy()
                V_cur = V[:, keep].copy()
                W_cur = W[:, keep].copy()
                T_cur = T[:, keep].copy()
                changed = True
                break  # restart scan after removal
    return U_cur, V_cur, W_cur, keep


def _randomized_rank_search(U, V, W, T, n, m, p, start_time, deadline):
    """
    Try to find a lower rank by projecting onto a random r-dimensional subspace.
    For a given r, pick r random columns, compute SVD of T[:, cols], project onto top-r right singular vectors.
    Then reconstruct U_proj = U @ P, V_proj = V @ P, W_proj = W @ P where P is the projection matrix.
    Verify if the projected factors still work.
    """
    best_U, best_V, best_W, best_r = U, V, W, T.shape[1]
    
    for r in range(T.shape[1] - 1, max(20, T.shape[1] - 15), -1):
        if time.time() > deadline - 10:
            break
        found = False
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            cols = np.random.choice(T.shape[1], size=r, replace=False)
            T_sub = T[:, cols]
            # SVD to get orthonormal basis for column space
            try:
                U_svd, s, Vt = np.linalg.svd(T_sub, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            # The column space of T_sub is spanned by U_svd (64 x r)
            # We want to find a rank-r decomposition: T ≈ U_svd @ diag(s) @ Vt
            # The factors in the rank-r decomposition are:
            # U_new = U @ Vt.T @ diag(sqrt(s))  ... this is getting complex.
            
            # Alternative: use the fact that if T has rank r, we can factor T = X @ Y
            # where X is 64 x r and Y is r x rank.
            # Then U_new = U @ Y, V_new = V @ Y, W_new = W @ Y? No.
            
            # Simpler approach: just try random subsets of columns and verify.
            U_test = U[:, cols]
            V_test = V[:, cols]
            W_test = W[:, cols]
            if _verify_fast(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p, trials=20, tol=1e-9):
                best_U, best_V, best_W, best_r = U_test, V_test, W_test, r
                found = True
                break
        if found:
            continue  # try lower r
        else:
            break  # can't go lower with this method
    
    return best_U, best_V, best_W, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin

    n, m, p = DIMENSIONS

    # Build initial rank-56 decomposition
    U_list, V_list, W_list = _build_block_strassen()
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)

    # Verify initial
    if not _verify_fast(U_list, V_list, W_list, n, m, p, trials=30, tol=1e-9):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Build tensor image
    T = _tensor_image(U, V, W, n, m, p)

    # Step 1: Greedy null-space reduction
    U_red, V_red, W_red, keep = _greedy_nullspace_reduction(U, V, W, T, n, m, p, start_time, deadline)
    rank_after_greedy = len(keep)

    # Verify reduced
    if _verify_fast(U_red.tolist(), V_red.tolist(), W_red.tolist(), n, m, p, trials=30, tol=1e-9):
        U, V, W = U_red, V_red, W_red
    else:
        # Fallback to original
        U, V, W = np.array(U_list, dtype=float), np.array(V_list, dtype=float), np.array(W_list, dtype=float)
        rank_after_greedy = 56

    # Step 2: Randomized rank search
    if time.time() < deadline - 10:
        T2 = _tensor_image(U, V, W, n, m, p)
        U_best, V_best, W_best, best_r = _randomized_rank_search(U, V, W, T2, n, m, p, start_time, deadline)
        if best_r < rank_after_greedy:
            U, V, W = U_best, V_best, W_best

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
