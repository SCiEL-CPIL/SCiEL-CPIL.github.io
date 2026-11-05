"""Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DEADLINE = 140  # safety margin before 150s kill


def _strassen_222():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # M1..M7
    # U[i,j], V[j,k], W[k,i]
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B01+B10) -> C01, C10
    # U: A00, A11 ; V: B01, B10 ; W: C01, C10
    U[0, 0] = 1; U[3, 0] = 1
    V[2, 0] = 1; V[3, 0] = 1
    W[1, 0] = 1; W[2, 0] = 1

    # M2 = A10*(B00+B11) -> C10
    U[2, 1] = 1
    V[0, 1] = 1; V[3, 1] = 1
    W[2, 1] = 1

    # M3 = A01*B11 -> C01
    U[1, 2] = 1
    V[3, 2] = 1
    W[1, 2] = 1

    # M4 = A00*B00 -> C00
    U[0, 3] = 1
    V[0, 3] = 1
    W[0, 3] = 1

    # M5 = (A10+A11)*B11 -> C11
    U[2, 4] = 1; U[3, 4] = 1
    V[3, 4] = 1
    W[3, 4] = 1

    # M6 = (A00-A10)*(B10+B11) -> C10, C11
    U[0, 5] = 1; U[2, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[2, 5] = 1; W[3, 5] = -1

    # M7 = (A01+A11)*(B00-B01) -> C00, C01
    U[1, 6] = 1; U[3, 6] = 1
    V[0, 6] = 1; V[2, 6] = -1
    W[0, 6] = 1; W[1, 6] = -1

    return U, V, W


def _schoolbook_block(n, m, p):
    """Return schoolbook decomposition for n x m * m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    R = n * m * p
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _embed_block(U, V, W, i0, j0, k0, i_size, j_size, k_size, col_offset, n_total, m_total, p_total):
    """Embed a block decomposition into the full 4x5x6 tensor.
    U_block: (i_size*j_size, r), V_block: (j_size*k_size, r), W_block: (k_size*i_size, r)
    Maps local indices to global indices.
    """
    r = U.shape[1]
    U_full = np.zeros((n_total * m_total, r), dtype=float)
    V_full = np.zeros((m_total * p_total, r), dtype=float)
    W_full = np.zeros((p_total * n_total, r), dtype=float)

    for li in range(i_size):
        for lj in range(j_size):
            gi = i0 + li
            gj = j0 + lj
            U_full[gi * m_total + lj + j0, :] = U[li * j_size + lj, :]

    for lj in range(j_size):
        for lk in range(k_size):
            gj = j0 + lj
            gk = k0 + lk
            V_full[gj * p_total + gk, :] = V[lj * k_size + lk, :]

    for lk in range(k_size):
        for li in range(i_size):
            gk = k0 + lk
            gi = i0 + li
            W_full[gk * n_total + gi, :] = W[lk * i_size + li, :]

    return U_full, V_full, W_full


def construct():
    start_time = time.time()
    n, m, p = 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    def schoolbook():
        R = n * m * p
        U = [[0.0] * R for _ in range(nm)]
        V = [[0.0] * R for _ in range(mp)]
        W = [[0.0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    def verify(U_arr, V_arr, W_arr, r):
        """Fast verification using numpy einsum."""
        if time.time() > DEADLINE - 3:
            return False
        U_r = np.array(U_arr).reshape(n, m, r)
        V_r = np.array(V_arr).reshape(m, p, r)
        W_r = np.array(W_arr).reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    def block_decomposition():
        """
        Block diagonal embedding strategy:
        Split 4x5*5x6 into blocks and use Strassen for 2x2*2x2 blocks,
        schoolbook for smaller blocks.
        
        Partition:
        i: [0,1], [2,3]  (sizes 2, 2)
        j: [0,1], [2,3], [4]  (sizes 2, 2, 1)
        k: [0,1], [2,3], [4,5]  (sizes 2, 2, 2)
        
        Blocks:
        (2,2,2): 2*3 = 6 blocks -> Strassen rank 7 each = 42
        (2,2,1): 2*1 = 2 blocks -> schoolbook rank 4 each = 8
        (2,1,2): 2*3 = 6 blocks -> schoolbook rank 4 each = 24
        (2,1,1): 2*1 = 2 blocks -> schoolbook rank 2 each = 4
        Total: 42 + 8 + 24 + 4 = 78
        """
        if time.time() > DEADLINE - 10:
            return None

        i_offsets = [0, 2]
        i_sizes = [2, 2]
        j_offsets = [0, 2, 4]
        j_sizes = [2, 2, 1]
        k_offsets = [0, 2, 4]
        k_sizes = [2, 2, 2]

        # Precompute Strassen and schoolbook decompositions
        strassen_U, strassen_V, strassen_W = _strassen_222()

        # Precompute schoolbook for various block sizes
        sb_221_U, sb_221_V, sb_221_W = _schoolbook_block(2, 2, 1)  # rank 4
        sb_212_U, sb_212_V, sb_212_W = _schoolbook_block(2, 1, 2)  # rank 4
        sb_211_U, sb_211_V, sb_211_W = _schoolbook_block(2, 1, 1)  # rank 2

        # Estimate total rank
        total_rank = 0
        for i_idx in range(2):
            for j_idx in range(3):
                for k_idx in range(3):
                    ns, ms, ps = i_sizes[i_idx], j_sizes[j_idx], k_sizes[k_idx]
                    if ns == 2 and ms == 2 and ps == 2:
                        total_rank += 7
                    elif ns == 2 and ms == 2 and ps == 1:
                        total_rank += 4
                    elif ns == 2 and ms == 1 and ps == 2:
                        total_rank += 4
                    elif ns == 2 and ms == 1 and ps == 1:
                        total_rank += 2

        U_full = np.zeros((nm, total_rank), dtype=float)
        V_full = np.zeros((mp, total_rank), dtype=float)
        W_full = np.zeros((pn, total_rank), dtype=float)

        col_offset = 0
        for i_idx, i0 in enumerate(i_offsets):
            ns = i_sizes[i_idx]
            for j_idx, j0 in enumerate(j_offsets):
                ms = j_sizes[j_idx]
                for k_idx, k0 in enumerate(k_offsets):
                    ps = k_sizes[k_idx]

                    if ns == 2 and ms == 2 and ps == 2:
                        U_b, V_b, W_b = strassen_U, strassen_V, strassen_W
                    elif ns == 2 and ms == 2 and ps == 1:
                        U_b, V_b, W_b = sb_221_U, sb_221_V, sb_221_W
                    elif ns == 2 and ms == 1 and ps == 2:
                        U_b, V_b, W_b = sb_212_U, sb_212_V, sb_212_W
                    elif ns == 2 and ms == 1 and ps == 1:
                        U_b, V_b, W_b = sb_211_U, sb_211_V, sb_211_W
                    else:
                        U_b, V_b, W_b = _schoolbook_block(ns, ms, ps)

                    r_b = U_b.shape[1]
                    U_emb, V_emb, W_emb = _embed_block(
                        U_b, V_b, W_b, i0, j0, k0, ns, ms, ps,
                        col_offset, n, m, p
                    )
                    U_full[:, col_offset:col_offset + r_b] = U_emb
                    V_full[:, col_offset:col_offset + r_b] = V_emb
                    W_full[:, col_offset:col_offset + r_b] = W_emb
                    col_offset += r_b

        if col_offset != total_rank:
            return None

        # Verify
        U_list = [[float(U_full[i, r]) for r in range(total_rank)] for i in range(nm)]
        V_list = [[float(V_full[i, r]) for r in range(total_rank)] for i in range(mp)]
        W_list = [[float(W_full[i, r]) for r in range(total_rank)] for i in range(pn)]

        if verify(U_list, V_list, W_list, total_rank):
            return {"U": U_list, "V": V_list, "W": W_list}
        return None

    def try_rank_optimized(R, time_budget):
        """Try to find exact rank-R decomposition via optimization."""
        deadline = time.time() + time_budget
        best_obj = float('inf')
        best_sol = None

        n_restarts = 0
        while time.time() < deadline - 2 and n_restarts < 25:
            n_restarts += 1

            # Structured initialization: ensure coverage of all (i,j,k) triples
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))

            for c in range(R):
                # Cycle through (i,j,k) with some randomization
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p

                # Primary entries with random scaling
                U[i * m + j, c] = random.choice([-1, 1]) * random.uniform(0.8, 1.5)
                V[j * p + k, c] = random.choice([-1, 1]) * random.uniform(0.8, 1.5)
                W[k * n + i, c] = random.choice([-1, 1]) * random.uniform(0.8, 1.5)

                # Add off-diagonal for flexibility
                if random.random() < 0.5:
                    i2 = random.randint(0, n - 1)
                    j2 = random.randint(0, m - 1)
                    U[i2 * m + j2, c] = random.choice([-1, 1]) * random.uniform(0.1, 0.4)
                if random.random() < 0.5:
                    j2 = random.randint(0, m - 1)
                    k2 = random.randint(0, p - 1)
                    V[j2 * p + k2, c] = random.choice([-1, 1]) * random.uniform(0.1, 0.4)
                if random.random() < 0.5:
                    k2 = random.randint(0, p - 1)
                    i2 = random.randint(0, n - 1)
                    W[k2 * n + i2, c] = random.choice([-1, 1]) * random.uniform(0.1, 0.4)

            def compute_violations(Ux, Vx, Wx):
                """Compute violations using vectorized operations."""
                U_r = Ux.reshape(n, m, R)
                V_r = Vx.reshape(m, p, R)
                W_r = Wx.reshape(p, n, R)
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                return T - np.ones((n, m, p))

            def obj_and_grad(x):
                Ux = x[:nm * R].reshape(nm, R)
                Vx = x[nm * R:nm * R + mp * R].reshape(mp, R)
                Wx = x[nm * R + mp * R:nm * R + mp * R + pn * R].reshape(pn, R)

                viol = compute_violations(Ux, Vx, Wx)
                fval = float(np.sum(viol * viol))

                # Compute gradients
                U_r = Ux.reshape(n, m, R)
                V_r = Vx.reshape(m, p, R)
                W_r = Wx.reshape(p, n, R)

                dU = np.zeros_like(Ux)
                dV = np.zeros_like(Vx)
                dW = np.zeros_like(Wx)

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = 2.0 * viol[i, j, k]
                            for r in range(R):
                                dU[i * m + j, r] += diff * Vx[j * p + k, r] * Wx[k * n + i, r]
                                dV[j * p + k, r] += diff * Ux[i * m + j, r] * Wx[k * n + i, r]
                                dW[k * n + i, r] += diff * Ux[i * m + j, r] * Vx[j * p + k, r]

                gx = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return fval, gx

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                fval, gval = obj_and_grad(x0)

                # Coarse optimization
                res = minimize(lambda x: obj_and_grad(x)[0], x0,
                               jac=lambda x: obj_and_grad(x)[1],
                               method='L-BFGS-B',
                               options={'maxiter': 100, 'maxfun': 3000,
                                        'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30})
                fval = res.fun

                if fval < 1e-8:
                    # Polish pass
                    res2 = minimize(lambda x: obj_and_grad(x)[0], res.x,
                                    jac=lambda x: obj_and_grad(x)[1],
                                    method='L-BFGS-B',
                                    options={'maxiter': 200, 'maxfun': 6000,
                                             'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50})
                    fval = res2.fun
                    x_opt = res2.x
                else:
                    x_opt = res.x

                if fval < best_obj:
                    best_obj = fval
                    U_opt = x_opt[:nm * R].reshape(nm, R)
                    V_opt = x_opt[nm * R:nm * R + mp * R].reshape(mp, R)
                    W_opt = x_opt[nm * R + mp * R:nm * R + mp * R + pn * R].reshape(pn, R)
                    best_sol = (U_opt, V_opt, W_opt)

            except Exception:
                pass

            if best_obj < 1e-14:
                break

        if best_sol is not None and best_obj < 1e-10:
            U_opt, V_opt, W_opt = best_sol
            # Verify exactness
            U_list = [[float(U_opt[i, r]) for r in range(R)] for i in range(nm)]
            V_list = [[float(V_opt[i, r]) for r in range(R)] for i in range(mp)]
            W_list = [[float(W_opt[i, r]) for r in range(R)] for i in range(pn)]

            if verify(U_list, V_list, W_list, R):
                return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # First, try the block decomposition (deterministic, fast)
    block_result = block_decomposition()
    if block_result is not None:
        best = block_result
        # Try to improve with optimization at the block rank
        block_rank = len(block_result['U'][0])
        if time.time() < DEADLINE - 20:
            opt_result = try_rank_optimized(block_rank, min(30, DEADLINE - time.time() - 10))
            if opt_result is not None:
                best = opt_result
        return best

    # Fallback: try optimization at various ranks
    ranks_to_try = [60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 105, 110, 115, 119]

    for R in ranks_to_try:
        if time.time() > DEADLINE - 10:
            break
        time_left = DEADLINE - time.time() - 5
        if time_left < 8:
            break

        # Allocate more time to lower ranks (harder)
        if R <= 72:
            budget = min(time_left, 30)
        elif R <= 90:
            budget = min(time_left, 20)
        else:
            budget = min(time_left, 15)

        result = try_rank_optimized(R, budget)

        if result is not None:
            best = result
            break

    return best
