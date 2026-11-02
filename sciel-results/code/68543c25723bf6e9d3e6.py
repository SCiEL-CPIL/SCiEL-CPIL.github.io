import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 22
    w_dim = p * n  # 22

    # Build the multiplication tensor entries
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    tensor_entries = []
    nz_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
                nz_set.add((u_idx, v_idx, w_idx))

    # Pre-compute indices for fast ALS
    entries_by_u = {i: [] for i in range(u_dim)}
    entries_by_v = {i: [] for i in range(v_dim)}
    entries_by_w = {i: [] for i in range(w_dim)}
    for (u_idx, v_idx, w_idx) in tensor_entries:
        entries_by_u[u_idx].append((v_idx, w_idx))
        entries_by_v[v_idx].append((u_idx, w_idx))
        entries_by_w[w_idx].append((u_idx, v_idx))

    def make_rank39():
        """
        Returns a known rank-39 decomposition.
        5 Strassen blocks (rank 7) + 1 Schoolbook block (rank 4).
        """
        rank = 39
        U = np.zeros((u_dim, rank), dtype=float)
        V = np.zeros((v_dim, rank), dtype=float)
        W = np.zeros((w_dim, rank), dtype=float)

        r = 0
        # 5 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9)
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1

            # V indices for B (j*p + k)
            v_e = k0       # B[0, k0]
            v_f = k1       # B[0, k1]
            v_g = 11 + k0  # B[1, k0]
            v_h = 11 + k1  # B[1, k1]

            # W indices for C (k*n + i)
            w_c00 = 2 * k0     # C[0, k0]
            w_c10 = 2 * k0 + 1 # C[1, k0]
            w_c01 = 2 * k1     # C[0, k1]
            w_c11 = 2 * k1 + 1 # C[1, k1]

            # m1 = (a+d)(e+h)
            U[0, r] = 1; U[3, r] = 1
            V[v_e, r] = 1; V[v_h, r] = 1
            W[w_c00, r] = 1; W[w_c11, r] = 1
            r += 1

            # m2 = (c+d)e
            U[2, r] = 1; U[3, r] = 1
            V[v_e, r] = 1
            W[w_c10, r] = 1; W[w_c11, r] = -1
            r += 1

            # m3 = a(f-h)
            U[0, r] = 1
            V[v_f, r] = 1; V[v_h, r] = -1
            W[w_c01, r] = 1; W[w_c11, r] = 1
            r += 1

            # m4 = d(g-e)
            U[3, r] = 1
            V[v_g, r] = 1; V[v_e, r] = -1
            W[w_c00, r] = 1; W[w_c10, r] = 1
            r += 1

            # m5 = (a+b)h
            U[0, r] = 1; U[1, r] = 1
            V[v_h, r] = 1
            W[w_c00, r] = -1; W[w_c01, r] = 1
            r += 1

            # m6 = (c-a)(e+f)
            U[2, r] = 1; U[0, r] = -1
            V[v_e, r] = 1; V[v_f, r] = 1
            W[w_c11, r] = 1
            r += 1

            # m7 = (b-d)(g+h)
            U[1, r] = 1; U[3, r] = -1
            V[v_g, r] = 1; V[v_h, r] = 1
            W[w_c00, r] = 1
            r += 1

        # Last column k=10: Schoolbook
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = 11 + k2 # B[1, 10]
        w_c02 = 2 * k2      # C[0, 10]
        w_c12 = 2 * k2 + 1  # C[1, 10]

        # a * B[0, 10]
        U[0, r] = 1
        V[v_e2, r] = 1
        W[w_c02, r] = 1
        r += 1

        # b * B[1, 10]
        U[1, r] = 1
        V[v_g2, r] = 1
        W[w_c02, r] = 1
        r += 1

        # c * B[0, 10]
        U[2, r] = 1
        V[v_e2, r] = 1
        W[w_c12, r] = 1
        r += 1

        # d * B[1, 10]
        U[3, r] = 1
        V[v_g2, r] = 1
        W[w_c12, r] = 1
        r += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_full(U, V, W, tol=1e-9):
        """
        Verifies if U, V, W exactly decompose the tensor.
        U, V, W are numpy arrays.
        """
        # Check non-zero entries
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = np.dot(U[u_idx] * V[v_idx], W[w_idx])
            if abs(total - 1) > tol:
                return False

        # Check zero entries
        # Optimization: Only check a subset if time is tight? 
        # But for correctness, we must check all.
        # There are 4*22*22 - 44 = 1936 zero entries.
        # 1936 dots of size rank (<=39) is fast.
        for u_idx in range(u_dim):
            for v_idx in range(v_dim):
                for w_idx in range(w_dim):
                    if (u_idx, v_idx, w_idx) not in nz_set:
                        total = np.dot(U[u_idx] * V[v_idx], W[w_idx])
                        if abs(total) > tol:
                            return False
        return True

    def compute_error(U, V, W, zero_weight=1.0):
        """
        Computes the squared error for non-zero and zero entries.
        """
        err = 0.0
        # Non-zero entries
        for (u_idx, v_idx, w_idx) in tensor_entries:
            val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
            err += (val - 1.0) ** 2
        
        # Zero entries
        for u_idx in range(u_dim):
            for v_idx in range(v_dim):
                for w_idx in range(w_dim):
                    if (u_idx, v_idx, w_idx) not in nz_set:
                        val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
                        err += zero_weight * val * val
        return err

    def local_search_polish(U, V, W, max_iters=100, tol=1e-9):
        """
        Attempts to polish a near-exact solution to exact integers/half-integers.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)

        # Round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2

        candidates = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        best_err = compute_error(U, V, W)

        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break

            improved = False
            # Try changing one element at a time
            for mat_name in ['U', 'V', 'W']:
                if mat_name == 'U':
                    mat = U
                    dims = (u_dim, U.shape[1])
                elif mat_name == 'V':
                    mat = V
                    dims = (v_dim, V.shape[1])
                else:
                    mat = W
                    dims = (w_dim, W.shape[1])

                for idx in range(dims[0]):
                    for col in range(dims[1]):
                        old_val = mat[idx, col]
                        for new_val in candidates:
                            if abs(new_val - old_val) < 1e-12:
                                continue
                            mat[idx, col] = new_val
                            new_err = compute_error(U, V, W)
                            if new_err < best_err - 1e-12:
                                best_err = new_err
                                improved = True
                                break
                            else:
                                mat[idx, col] = old_val
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

            if verify_full(U, V, W, tol):
                return U.tolist(), V.tolist(), W.tolist()

        if verify_full(U, V, W, tol):
            return U.tolist(), V.tolist(), W.tolist()
        return None

    def try_als_rank(target_rank, max_iters=50, seed=42, init_type='random'):
        """
        Attempts to find a decomposition of target_rank using ALS.
        """
        if time.time() > deadline - 15:
            return None

        np.random.seed(seed)

        if init_type == 'random':
            U = np.random.randn(u_dim, target_rank) * 0.5
            V = np.random.randn(v_dim, target_rank) * 0.5
            W = np.random.randn(w_dim, target_rank) * 0.5
        elif init_type == 'strassen_perturb':
            # Use rank 39 as base, truncate
            r39 = make_rank39()
            U_full = np.array(r39['U'])
            V_full = np.array(r39['V'])
            W_full = np.array(r39['W'])
            U = U_full[:, :target_rank].copy() + np.random.randn(u_dim, target_rank) * 0.1
            V = V_full[:, :target_rank].copy() + np.random.randn(v_dim, target_rank) * 0.1
            W = W_full[:, :target_rank].copy() + np.random.randn(w_dim, target_rank) * 0.1
        else:
            U = np.random.randn(u_dim, target_rank) * 0.5
            V = np.random.randn(v_dim, target_rank) * 0.5
            W = np.random.randn(w_dim, target_rank) * 0.5

        best_err = float('inf')
        best_U, best_V, best_W = None, None, None

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Update U
            for u_idx in range(u_dim):
                entries_u = entries_by_u[u_idx]
                if not entries_u:
                    continue
                A_mat = np.array([V[v_idx] * W[w_idx] for (v_idx, w_idx) in entries_u])
                b_vec = np.ones(len(entries_u))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[u_idx] = x
                except:
                    pass

            # Update V
            for v_idx in range(v_dim):
                entries_v = entries_by_v[v_idx]
                if not entries_v:
                    continue
                A_mat = np.array([U[u_idx] * W[w_idx] for (u_idx, w_idx) in entries_v])
                b_vec = np.ones(len(entries_v))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[v_idx] = x
                except:
                    pass

            # Update W
            for w_idx in range(w_dim):
                entries_w = entries_by_w[w_idx]
                if not entries_w:
                    continue
                A_mat = np.array([U[u_idx] * V[v_idx] for (u_idx, v_idx) in entries_w])
                b_vec = np.ones(len(entries_w))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[w_idx] = x
                except:
                    pass

            err = compute_error(U, V, W, zero_weight=1.0)
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()

            if err < 1e-14:
                break

        if best_U is None:
            return None

        # Polish
        polished = local_search_polish(best_U, best_V, best_W, max_iters=50)
        if polished is not None:
            U_list, V_list, W_list = polished
            if verify_full(np.array(U_list), np.array(V_list), np.array(W_list)):
                return {"U": U_list, "V": V_list, "W": W_list}

        # Check direct rounding
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if verify_full(U_rounded, V_rounded, W_rounded):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}

        return None

    # Try to find lower rank
    # We know 39 works. Try 38, 37.
    for target_rank in [38, 37]:
        if time.time() > deadline - 20:
            break
        
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            init_type = 'strassen_perturb' if seed < 2 else 'random'
            result = try_als_rank(target_rank, max_iters=50, seed=seed, init_type=init_type)
            if result is not None:
                return result

    # Fallback to rank 39
    return make_rank39()
