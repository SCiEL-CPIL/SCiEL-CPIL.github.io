"""Matrix multiplication 3x5x6 rank-minimal decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 6]
HALF_INT_GRID = np.arange(-20, 21, 0.5)  # -20, -19.5, ..., 19.5, 20

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if j == k else 0.0
                    actual = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    err = abs(actual - expected)
                    if err > max_err:
                        max_err = err
        return max_err < 1e-9, max_err

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U.tolist(), V.tolist(), W.tolist(), rank

    def als_decompose(target_rank, max_iters=200, tol=1e-12, seed=0):
        if time_left() < 5:
            return None, 0.0

        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, target_rank) * 0.5
        V = rng.randn(m*p, target_rank) * 0.5
        W = rng.randn(p*n, target_rank) * 0.5

        best_err = float('inf')

        for iteration in range(max_iters):
            if time_left() < 3:
                break

            for i in range(n):
                for j in range(m):
                    row_u = i*m + j
                    M = np.empty((p, target_rank))
                    for k in range(p):
                        M[k] = V[j*p+k] * W[k*n+i]
                    tgt = np.array([1.0 if j == k else 0.0 for k in range(p)])
                    try:
                        U[row_u] = np.linalg.lstsq(M.T, tgt, rcond=None)[0]
                    except:
                        pass

            for j in range(m):
                for k in range(p):
                    row_v = j*p + k
                    M = np.empty((n, target_rank))
                    for i in range(n):
                        M[i] = U[i*m+j] * W[k*n+i]
                    tgt = np.array([1.0 if j == k else 0.0 for i in range(n)])
                    try:
                        V[row_v] = np.linalg.lstsq(M.T, tgt, rcond=None)[0]
                    except:
                        pass

            for k in range(p):
                for i in range(n):
                    row_w = k*n + i
                    M = np.empty((m, target_rank))
                    for j in range(m):
                        M[j] = U[i*m+j] * V[j*p+k]
                    tgt = np.array([1.0 if j == k else 0.0 for j in range(m)])
                    try:
                        W[row_w] = np.linalg.lstsq(M.T, tgt, rcond=None)[0]
                    except:
                        pass

            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0 if j == k else 0.0
                        actual = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err = abs(actual - expected)
                        if err > max_err:
                            max_err = err

            if max_err < best_err:
                best_err = max_err
                if max_err < tol:
                    break

        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        ok, final_err = verify(U_list, V_list, W_list, target_rank)
        if ok:
            return (U_list, V_list, W_list), target_rank
        return None, best_err

    def polish(U, V, W, target_rank):
        """Use scipy least_squares to polish the solution."""
        if time_left() < 5:
            return None

        U0 = np.array(U, dtype=np.float64).flatten()
        V0 = np.array(V, dtype=np.float64).flatten()
        W0 = np.array(W, dtype=np.float64).flatten()
        x0 = np.concatenate([U0, V0, W0])

        def residuals(x):
            U_flat = x[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            W_flat = x[2*n*m*target_rank:].reshape(p*n, target_rank)
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0 if j == k else 0.0
                        actual = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                        res.append(actual - expected)
            return np.array(res)

        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            U_polished = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_polished = result.x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            W_polished = result.x[2*n*m*target_rank:].reshape(p*n, target_rank)
            U_list = U_polished.tolist()
            V_list = V_polished.tolist()
            W_list = W_polished.tolist()
            ok, err = verify(U_list, V_list, W_list, target_rank)
            if ok:
                return (U_list, V_list, W_list)
        except:
            pass
        return None

    def try_rank_als(target_rank, max_restarts=10):
        for seed in range(max_restarts):
            if time_left() < 5:
                break
            result, err = als_decompose(target_rank, seed=seed)
            if result is not None:
                return result
            # If ALS got close, try polishing
            if err < 1e-4 and time_left() > 10:
                U_tmp, V_tmp, W_tmp = als_decompose(target_rank, seed=seed, max_iters=50)
                if U_tmp is not None:
                    polished = polish(U_tmp, V_tmp, W_tmp, target_rank)
                    if polished is not None:
                        return polished
        return None

    def greedy_rank1_projection(target_rank, max_restarts=5):
        """
        Greedy algorithm using rank-1 projection.
        At each step, find the best rank-1 tensor (with half-integer entries)
        that maximally reduces the residual Frobenius norm.
        """
        if time_left() < 5:
            return None

        # Build the target tensor T[i,j,k] = 1 if j==k else 0
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j == k:
                        T[i, j, k] = 1.0

        # Residual starts as the full tensor
        R = T.copy()

        best_result = None
        best_fro = np.linalg.norm(R)

        for restart in range(max_restarts):
            if time_left() < 3:
                break

            # Initialize with random perturbation for diversity
            R_cur = R.copy()
            if restart > 0:
                rng = np.random.RandomState(restart)
                # Small random perturbation to escape local optima
                R_cur += rng.randn(*R.shape) * 0.01

            U_cols = []
            V_cols = []
            W_cols = []

            for step in range(target_rank):
                if time_left() < 2:
                    break

                # Find best rank-1 tensor u ⊙ v ⊙ w (elementwise) that
                # minimizes ||R - u⊙v⊙w||_F
                # This is equivalent to maximizing <R, u⊙v⊙w> - 0.5||u⊙v⊙w||_F^2
                # For fixed v,w: u_opt = R_{:,j,k} * v_j * w_k / ||v⊙w||^2 ... 
                # Actually, the optimal u for fixed v,w is:
                # u_i = sum_{j,k} R[i,j,k] * v[j] * w[k] / (sum_{j,k} v[j]^2 * w[k]^2)
                # Similarly for v and w.
                # We use alternating minimization to find the best rank-1 approx.

                # Initialize u, v, w randomly or from residual
                rng2 = np.random.RandomState(step + restart * 100)
                u = rng2.randn(n) * 0.5
                v = rng2.randn(m) * 0.5
                w = rng2.randn(p) * 0.5

                # Alternating minimization for best rank-1 approximation
                for am_iter in range(20):
                    # Update u: u_i = sum_{j,k} R[i,j,k] * v[j] * w[k] / ||v⊙w||^2
                    vw = np.outer(v, w)  # shape (m, p)
                    denom_u = np.sum(vw * vw)
                    if denom_u > 1e-12:
                        u_new = np.zeros(n)
                        for i in range(n):
                            u_new[i] = np.sum(R_cur[i] * vw) / denom_u
                        u = u_new

                    # Update v: v_j = sum_{i,k} R[i,j,k] * u[i] * w[k] / ||u⊙w||^2
                    uw = np.outer(u, w)  # shape (n, p)
                    denom_v = np.sum(uw * uw)
                    if denom_v > 1e-12:
                        v_new = np.zeros(m)
                        for j in range(m):
                            v_new[j] = np.sum(R_cur[:, j, :] * uw) / denom_v
                        v = v_new

                    # Update w: w_k = sum_{i,j} R[i,j,k] * u[i] * v[j] / ||u⊙v||^2
                    uv = np.outer(u, v)  # shape (n, m)
                    denom_w = np.sum(uv * uv)
                    if denom_w > 1e-12:
                        w_new = np.zeros(p)
                        for k in range(p):
                            w_new[k] = np.sum(R_cur[:, :, k] * uv) / denom_w
                        w = w_new

                # Snap to half-integers
                u_half = np.round(u * 2) / 2
                v_half = np.round(v * 2) / 2
                w_half = np.round(w * 2) / 2

                # Compute the rank-1 tensor
                rank1_tensor = np.einsum('i,j,k->ijk', u_half, v_half, w_half)

                # Check improvement
                new_R = R_cur - rank1_tensor
                new_fro = np.linalg.norm(new_R)
                old_fro = np.linalg.norm(R_cur)

                if new_fro < old_fro - 1e-10:
                    R_cur = new_R
                    U_cols.append(u_half)
                    V_cols.append(v_half)
                    W_cols.append(w_half)
                else:
                    # Try with small perturbations of the snapped values
                    found = False
                    for du in [-0.5, 0, 0.5]:
                        for dv in [-0.5, 0, 0.5]:
                            for dw in [-0.5, 0, 0.5]:
                                u_t = u_half + du
                                v_t = v_half + dv
                                w_t = w_half + dw
                                r1_t = np.einsum('i,j,k->ijk', u_t, v_t, w_t)
                                nr_t = R_cur - r1_t
                                nf_t = np.linalg.norm(nr_t)
                                if nf_t < old_fro - 1e-10:
                                    R_cur = nr_t
                                    U_cols.append(u_t)
                                    V_cols.append(v_t)
                                    W_cols.append(w_t)
                                    found = True
                                    break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        # No improvement possible, break
                        break

            # Verify the decomposition
            if len(U_cols) >= target_rank:
                U_mat = np.column_stack(U_cols)
                V_mat = np.column_stack(V_cols)
                W_mat = np.column_stack(W_cols)

                # Verify
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            expected = 1.0 if j == k else 0.0
                            actual = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                            err = abs(actual - expected)
                            if err > max_err:
                                max_err = err

                if max_err < 1e-9:
                    return (U_mat.tolist(), V_mat.tolist(), W_mat.tolist())

            # Track best Frobenius norm for diagnostics
            cur_fro = np.linalg.norm(R_cur)
            if cur_fro < best_fro:
                best_fro = cur_fro

        return None

    def try_rank(target_rank, max_restarts=10):
        # First try greedy rank-1 projection
        result = greedy_rank1_projection(target_rank, max_restarts=3)
        if result is not None:
            return result

        # Then try ALS
        result = try_rank_als(target_rank, max_restarts=max_restarts)
        if result is not None:
            return result

        return None

    U, V, W, rank = schoolbook()

    for target_rank in range(78, 90):
        if time_left() < 10:
            break
        result = try_rank(target_rank, max_restarts=8)
        if result is not None:
            U, V, W = result
            rank = target_rank
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
