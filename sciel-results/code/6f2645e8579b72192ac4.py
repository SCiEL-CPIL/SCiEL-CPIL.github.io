import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 118

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 22, 22

    # Build the multiplication tensor
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))

    tensor_set = set(tensor_entries)

    # Verify a decomposition is exact
    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = 0
            for r in range(rank):
                total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            if abs(total - 1) > tol:
                return False
        # Check zero constraints - sample for large sets
        all_u = range(nm)
        all_v = range(mp)
        all_w = range(pn)
        checked = 0
        for u_idx in all_u:
            for v_idx in all_v:
                for w_idx in all_w:
                    if (u_idx, v_idx, w_idx) in tensor_set:
                        continue
                    total = 0
                    for r in range(rank):
                        total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(total) > tol:
                        return False
                    checked += 1
                    if checked > 5000:
                        break
                if checked > 5000:
                    break
            if checked > 5000:
                break
        return True

    # Build the known rank-39 Strassen-based decomposition
    def make_rank39():
        rank = 39
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1

            v_e = k0
            v_f = k1
            v_g = 11 + k0
            v_h = 11 + k1

            w_c00 = 2 * k0
            w_c10 = 2 * k0 + 1
            w_c01 = 2 * k1
            w_c11 = 2 * k1 + 1

            # m1 = (a+d)(e+h)
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1

            # m2 = (c+d)e
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1

            # m3 = a(f-h)
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1

            # m4 = d(g-e)
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1

            # m5 = (a+b)h
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1

            # m6 = (c-a)(e+f)
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1

            # m7 = (b-d)(g+h)
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1

        k2 = 10
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1

        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1

        return {"U": U, "V": V, "W": W}

    # Try to reduce rank by finding linear dependencies among rank-1 terms
    # and attempting to merge them
    def try_reduce_rank(decomp, target_rank, max_attempts=50):
        """Try to reduce rank by finding and merging linearly dependent rank-1 terms."""
        if time.time() > deadline - 15:
            return None

        U = [row[:] for row in decomp['U']]
        V = [row[:] for row in decomp['V']]
        W = [row[:] for row in decomp['W']]
        current_rank = len(U[0])

        if current_rank <= target_rank:
            return decomp

        # Try to find pairs of rank-1 terms that can be merged
        # Two rank-1 terms (u1,v1,w1) and (u2,v2,w2) can be merged into one
        # if there exists alpha such that u1+alpha*u2, v1+alpha*v2, w1+alpha*w2
        # still satisfies the constraints, OR if one term is a scalar multiple of another
        # in a way that allows cancellation.

        # More practically: try to find if any two terms are linearly dependent
        # in the combined space, allowing us to eliminate one.

        # Strategy: For each pair of terms, check if removing one and adjusting
        # the other maintains the decomposition. This is expensive, so we limit attempts.

        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break

            # Pick a random pair of terms to try merging
            import random
            random.seed(attempt + int(time.time() * 1000) % 10000)
            i, j = random.sample(range(current_rank), 2)

            # Try to eliminate term j by absorbing it into term i
            # We need: for all (u,v,w) in tensor_entries:
            #   U[u][i]*V[v][i]*W[w][i] + U[u][j]*V[v][j]*W[w][j] = new_U[u]*new_V[v]*new_W[w]
            # This is generally not possible for a single rank-1 term.
            # Instead, try: can we set U[u][j]=0 for all u (or V or W) and adjust others?

            # Simpler approach: try to find if term j is redundant (its contribution
            # is already captured by other terms)
            # Check if removing term j still gives a valid decomposition
            U_test = [row[:current_rank-1] if idx < j else row[j+1:current_rank] for idx, row in enumerate(U)]
            V_test = [row[:current_rank-1] if idx < j else row[j+1:current_rank] for idx, row in enumerate(V)]
            W_test = [row[:current_rank-1] if idx < j else row[j+1:current_rank] for idx, row in enumerate(W)]

            # Quick check: is term j all zeros?
            if all(U[u][j] == 0 for u in range(nm)) or \
               all(V[v][j] == 0 for v in range(mp)) or \
               all(W[w][j] == 0 for w in range(pn)):
                # Term j is zero, remove it
                U = U_test
                V = V_test
                W = W_test
                current_rank -= 1
                if current_rank <= target_rank:
                    return {"U": U, "V": V, "W": W}
                continue

            # Try a more aggressive approach: use ALS to find a lower-rank approximation
            # starting from the current solution
            break

        # Use ALS with the current solution as initialization
        return _als_from_init(U, V, W, target_rank)

    def _als_from_init(U_init, V_init, W_init, target_rank, max_iters=200):
        """ALS optimization starting from an initial decomposition."""
        if time.time() > deadline - 10:
            return None

        current_rank = len(U_init[0])
        if current_rank < target_rank:
            return None

        # Truncate or pad to target_rank
        if current_rank > target_rank:
            # Keep the first target_rank terms
            U = [row[:target_rank] for row in U_init]
            V = [row[:target_rank] for row in V_init]
            W = [row[:target_rank] for row in W_init]
        else:
            U = [row[:] for row in U_init]
            V = [row[:] for row in V_init]
            W = [row[:] for row in W_init]

        # Build constraint matrices for ALS
        entries_by_u = [[] for _ in range(nm)]
        entries_by_v = [[] for _ in range(mp)]
        entries_by_w = [[] for _ in range(pn)]
        for (u_idx, v_idx, w_idx) in tensor_entries:
            entries_by_u[u_idx].append((v_idx, w_idx))
            entries_by_v[v_idx].append((u_idx, w_idx))
            entries_by_w[w_idx].append((u_idx, v_idx))

        # Zero constraints (sample)
        zero_entries = []
        for u_idx in range(nm):
            for v_idx in range(mp):
                for w_idx in range(pn):
                    if (u_idx, v_idx, w_idx) not in tensor_set:
                        zero_entries.append((u_idx, v_idx, w_idx))

        import random
        random.seed(42)
        np.random.seed(42)

        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        best_err = float('inf')
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Update U
            for u_idx in range(nm):
                e_list = entries_by_u[u_idx]
                if not e_list:
                    continue
                A_mat = np.array([V_arr[v_idx] * W_arr[w_idx] for (v_idx, w_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U_arr[u_idx] = x
                except:
                    pass

            # Update V
            for v_idx in range(mp):
                e_list = entries_by_v[v_idx]
                if not e_list:
                    continue
                A_mat = np.array([U_arr[u_idx] * W_arr[w_idx] for (u_idx, w_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V_arr[v_idx] = x
                except:
                    pass

            # Update W
            for w_idx in range(pn):
                e_list = entries_by_w[w_idx]
                if not e_list:
                    continue
                A_mat = np.array([U_arr[u_idx] * V_arr[v_idx] for (u_idx, v_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_arr[w_idx] = x
                except:
                    pass

            # Compute error
            err = 0.0
            for (u_idx, v_idx, w_idx) in tensor_entries:
                val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                err += (val - 1) ** 2

            # Sample zero constraints
            if zero_entries:
                sample_size = min(50, len(zero_entries))
                sample_idx = np.random.choice(len(zero_entries), sample_size, replace=False)
                zero_err = 0.0
                for idx in sample_idx:
                    u_idx, v_idx, w_idx = zero_entries[idx]
                    val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                    zero_err += val ** 2
                zero_err /= sample_size
            else:
                zero_err = 0.0

            total_err = err + 0.1 * zero_err

            if total_err < best_err:
                best_err = total_err
                best_U = U_arr.copy()
                best_V = V_arr.copy()
                best_W = W_arr.copy()

            if err < 1e-14 and zero_err < 1e-14:
                break

        # Round to half-integers and verify
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2

        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()

        if verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Start with rank-39 and try to reduce
    base_decomp = make_rank39()

    # Try direct rank reduction from 39 down to 35
    for target_rank in [38, 37, 36, 35]:
        if time.time() > deadline - 20:
            break

        result = try_reduce_rank(base_decomp, target_rank, max_attempts=20)
        if result is not None:
            return result

    # If that didn't work, try multiple random initializations with ALS
    for target_rank in [38, 37, 36, 35]:
        if time.time() > deadline - 15:
            break

        for seed in range(10):
            if time.time() > deadline - 15:
                break

            np.random.seed(seed)
            U = np.random.randn(nm, target_rank) * 0.3
            V = np.random.randn(mp, target_rank) * 0.3
            W = np.random.randn(pn, target_rank) * 0.3

            result = _als_from_init(U.tolist(), V.tolist(), W.tolist(), target_rank, max_iters=150)
            if result is not None:
                return result

    # Fall back to known rank-39
    return base_decomp
