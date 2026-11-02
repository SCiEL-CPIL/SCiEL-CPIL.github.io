"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 28 baseline with rank-27 search."""

import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def build_strassen_28():
        """Build the rank-28 Strassen decomposition for 2x4 x 4x4."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
        V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
        W = np.zeros((p * n, rank), dtype=float)  # 8 x 28

        # Strassen 7 terms for 2x2 matrix multiplication
        strassen_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        col = 0
        for b_col_block in range(2):
            for a_col_block in range(2):
                a_base = 2 * a_col_block
                b_row_base = 2 * a_col_block
                b_col_base = 2 * b_col_block
                c_col_base = 2 * b_col_block

                for u_t, v_t, w_t in strassen_terms:
                    for loc, coeff in u_t:
                        r = loc // 2
                        c = loc % 2
                        U[r * 4 + a_base + c, col] += coeff
                    for loc, coeff in v_t:
                        r = loc // 2
                        c = loc % 2
                        V[(b_row_base + r) * 4 + b_col_base + c, col] += coeff
                    for loc, coeff in w_t:
                        r = loc // 2
                        c = loc % 2
                        W[(c_col_base + c) * 2 + r, col] += coeff
                    col += 1

        return {"U": U, "V": V, "W": W}

    def validate_numpy(U, V, W):
        """Validate decomposition using numpy. Returns (valid, max_error)."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    s = np.dot(u_row, v_row * w_row)
                    max_err = max(max_err, abs(s - 1.0))
        for a in range(n * m):
            i1 = a // m
            j1 = a % m
            for b in range(m * p):
                j2 = b // p
                k1 = b % p
                for c in range(p * n):
                    i2 = c // n
                    k2 = c % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    u_row = U[a]
                    v_row = V[b]
                    w_row = W[c]
                    s = np.dot(u_row, v_row * w_row)
                    max_err = max(max_err, abs(s))
        return max_err < 1e-9, max_err

    # Build baseline rank-28
    base = build_strassen_28()
    best_result = {
        "U": base["U"].tolist(),
        "V": base["V"].tolist(),
        "W": base["W"].tolist()
    }
    best_rank = 28

    # Verify baseline
    valid, err = validate_numpy(base["U"], base["V"], base["W"])
    if not valid:
        return best_result

    target_rank = 27
    U28 = base["U"].copy()
    V28 = base["V"].copy()
    W28 = base["W"].copy()

    # Precompute index maps
    a_to_bc = {}
    for a in range(n*m):
        i1 = a // m
        j1 = a % m
        bc_list = []
        for b in range(m*p):
            j2 = b // p
            k1 = b % p
            if j2 != j1:
                continue
            for c in range(p*n):
                i2 = c // n
                k2 = c % n
                if i2 == i1 and k2 == k1:
                    bc_list.append((b, c))
        a_to_bc[a] = bc_list

    b_to_ac = {}
    for b in range(m*p):
        j2 = b // p
        k1 = b % p
        ac_list = []
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            if j1 != j2:
                continue
            for c in range(p*n):
                i2 = c // n
                k2 = c % n
                if i2 == i1 and k2 == k1:
                    ac_list.append((a, c))
        b_to_ac[b] = ac_list

    c_to_ab = {}
    for c in range(p*n):
        i2 = c // n
        k2 = c // n
        ab_list = []
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            if i1 != i2:
                continue
            for b in range(m*p):
                j2 = b // p
                k1 = b % p
                if j2 == j1 and k1 == k2:
                    ab_list.append((a, b))
        c_to_ab[c] = ab_list

    # Vectorized ALS for rank r
    def als_step(U, V, W, r, max_iter=60, tol=1e-13):
        err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            prev_err = err
            for factor in range(3):
                if factor == 0:
                    for a in range(n*m):
                        bc_list = a_to_bc[a]
                        if not bc_list:
                            U[a, :] = 0
                            continue
                        M_mat = np.zeros((len(bc_list), r))
                        rhs = np.ones(len(bc_list))
                        for idx, (b, c) in enumerate(bc_list):
                            M_mat[idx, :] = V[b, :] * W[c, :]
                        if np.any(M_mat != 0):
                            U[a, :], _, _, _ = np.linalg.lstsq(M_mat, rhs, rcond=None)
                        else:
                            U[a, :] = 0
                elif factor == 1:
                    for b in range(m*p):
                        ac_list = b_to_ac[b]
                        if not ac_list:
                            V[b, :] = 0
                            continue
                        M_mat = np.zeros((len(ac_list), r))
                        rhs = np.ones(len(ac_list))
                        for idx, (a, c) in enumerate(ac_list):
                            M_mat[idx, :] = U[a, :] * W[c, :]
                        if np.any(M_mat != 0):
                            V[b, :], _, _, _ = np.linalg.lstsq(M_mat, rhs, rcond=None)
                        else:
                            V[b, :] = 0
                else:
                    for c in range(p*n):
                        ab_list = c_to_ab[c]
                        if not ab_list:
                            W[c, :] = 0
                            continue
                        M_mat = np.zeros((len(ab_list), r))
                        rhs = np.ones(len(ab_list))
                        for idx, (a, b) in enumerate(ab_list):
                            M_mat[idx, :] = U[a, :] * V[b, :]
                        if np.any(M_mat != 0):
                            W[c, :], _, _, _ = np.linalg.lstsq(M_mat, rhs, rcond=None)
                        else:
                            W[c, :] = 0
            valid, err = validate_numpy(U, V, W)
            if valid:
                return U, V, W, err
            if abs(prev_err - err) < tol:
                break
        return U, V, W, err

    # Gradient descent with L1 penalty on squared Frobenius reconstruction error
    def grad_descent(U, V, W, r, max_iter=2000, lr=0.01, l1=0.01):
        """Gradient descent on reconstruction error + L1 penalty to promote sparsity."""
        # Precompute index structure for fast error computation
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We need T[a,b,c] = 1 for (a,b,c) with matching indices, 0 otherwise
        # Target: for each a (i,j), b (j,k), c (k,i) with matching j and k
        targets = {}
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b in range(m*p):
                j2 = b // p
                k1 = b % p
                if j1 != j2:
                    continue
                for c in range(p*n):
                    i2 = c // n
                    k2 = c % n
                    if i1 == i2 and k1 == k2:
                        targets[(a,b,c)] = 1.0

        # Flatten index lists for vectorized ops
        keys = list(targets.keys())
        vals = np.array([targets[k] for k in keys])
        ks_a = np.array([k[0] for k in keys])
        ks_b = np.array([k[1] for k in keys])
        ks_c = np.array([k[2] for k in keys])

        # Also need off-diagonal constraints (T should be 0)
        # Total number of (a,b,c) triples is 8*16*8 = 1024, on-diagonal is 2*4*4=32
        # Off-diagonal: 1024 - 32 = 992
        off_keys = []
        on_set = set(keys)
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a,b,c) not in on_set:
                        off_keys.append((a,b,c))
        off_a = np.array([k[0] for k in off_keys]) if off_keys else np.array([])
        off_b = np.array([k[1] for k in off_keys]) if off_keys else np.array([])
        off_c = np.array([k[2] for k in off_keys]) if off_keys else np.array([])

        def compute_error():
            # On-diagonal error
            on_err = 0.0
            if len(ks_a) > 0:
                for r in range(r):
                    contrib = U[ks_a, r] * V[ks_b, r] * W[ks_c, r]
                    on_err += np.sum((contrib - vals)**2)
            # Off-diagonal error
            off_err = 0.0
            if len(off_a) > 0:
                for r in range(r):
                    contrib = U[off_a, r] * V[off_b, r] * W[off_c, r]
                    off_err += np.sum(contrib**2)
            return on_err + off_err

        err = compute_error()
        best_err = err
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break

            # Compute gradients
            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)

            # On-diagonal gradient: d/dU[a,r] = sum over (b,c) for this a: 2*(T-1)*V[b,r]*W[c,r]
            for r in range(r):
                # On-diagonal
                if len(ks_a) > 0:
                    contrib = U[ks_a, r] * V[ks_b, r] * W[ks_c, r]
                    diff = contrib - vals
                    # For each a, sum over matching (b,c)
                    for idx in range(len(ks_a)):
                        a_idx = ks_a[idx]
                        b_idx = ks_b[idx]
                        c_idx = ks_c[idx]
                        gU[a_idx, r] += 2 * diff[idx] * V[b_idx, r] * W[c_idx, r]
                        gV[b_idx, r] += 2 * diff[idx] * U[a_idx, r] * W[c_idx, r]
                        gW[c_idx, r] += 2 * diff[idx] * U[a_idx, r] * V[b_idx, r]
                # Off-diagonal
                if len(off_a) > 0:
                    for idx in range(len(off_a)):
                        a_idx = off_a[idx]
                        b_idx = off_b[idx]
                        c_idx = off_c[idx]
                        contrib_val = 0.0
                        for rr in range(r):
                            contrib_val += U[a_idx, rr] * V[b_idx, rr] * W[c_idx, rr]
                        gU[a_idx, r] += 2 * contrib_val * V[b_idx, r] * W[c_idx, r]
                        gV[b_idx, r] += 2 * contrib_val * U[a_idx, r] * W[c_idx, r]
                        gW[c_idx, r] += 2 * contrib_val * U[a_idx, r] * V[b_idx, r]

            # L1 penalty gradient
            if l1 > 0:
                gU += l1 * np.sign(U)
                gV += l1 * np.sign(V)
                gW += l1 * np.sign(W)

            # Update
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW

            new_err = compute_error()
            if new_err < best_err:
                best_err = new_err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()

            if best_err < 1e-12:
                break
            # Adaptive learning rate
            if it % 100 == 99 and it > 0:
                lr *= 0.8

        return best_U, best_V, best_W, best_err

    # Search for rank 27
    found = False

    # Strategy 1: ALS with multiple initializations (remove one column + perturb)
    for remove_col in range(28):
        if time.time() > deadline - 40:
            break
        keep_cols = [c for c in range(28) if c != remove_col]
        for attempt in range(15):
            if time.time() > deadline - 5:
                break
            seed = 12345 + remove_col * 1000 + attempt * 7919
            np.random.seed(seed)
            scale = 0.05 + 0.02 * attempt
            U = U28[:, keep_cols].copy() + np.random.randn(n*m, target_rank) * scale
            V = V28[:, keep_cols].copy() + np.random.randn(m*p, target_rank) * scale
            W = W28[:, keep_cols].copy() + np.random.randn(p*n, target_rank) * scale
            U, V, W, err = als_step(U, V, W, target_rank, max_iter=80)
            if err < 1e-9:
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
        if found:
            break

    # Strategy 2: Gradient descent with L1 from structured init
    if not found and time.time() < deadline - 30:
        for attempt in range(8):
            if time.time() > deadline - 5:
                break
            seed = 55555 + attempt * 31337
            np.random.seed(seed)
            # Start from Strassen with one column removed, add moderate noise
            keep_cols = [c for c in range(28) if c != attempt % 28]
            U = U28[:, keep_cols].copy() + np.random.randn(n*m, target_rank) * 0.1
            V = V28[:, keep_cols].copy() + np.random.randn(m*p, target_rank) * 0.1
            W = W28[:, keep_cols].copy() + np.random.randn(p*n, target_rank) * 0.1
            U, V, W, err = grad_descent(U, V, W, target_rank, max_iter=1500, lr=0.005, l1=0.005)
            if err < 1e-9:
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break

    # Strategy 3: Random initialization + ALS
    if not found and time.time() < deadline - 15:
        for attempt in range(8):
            if time.time() > deadline - 5:
                break
            np.random.seed(77777 + attempt * 9999)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            U, V, W, err = als_step(U, V, W, target_rank, max_iter=60)
            if err < 1e-9:
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break

    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    rank = U.shape[1]
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(rank):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", rank)
