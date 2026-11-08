import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8

    # Precompute tensor index triples
    idx_ui = []
    idx_vj = []
    idx_wk = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_ui.append(i * m + j)
                idx_vj.append(j * p + k)
                idx_wk.append(k * n + i)
    idx_ui = np.array(idx_ui)
    idx_vj = np.array(idx_vj)
    idx_wk = np.array(idx_wk)
    num_entries = len(idx_ui)  # 32

    def compute_values(U, V, W):
        Uv = U[idx_ui, :]
        Vv = V[idx_vj, :]
        Wv = W[idx_wk, :]
        return np.sum(Uv * Vv * Wv, axis=1)

    def verify_decomposition(U, V, W, tol=1e-9):
        vals = compute_values(U, V, W)
        return bool(np.all(np.abs(vals - 1.0) < tol))

    def compute_gradients(U, V, W, vals):
        residuals = vals - 1.0
        Uv = U[idx_ui, :]
        Vv = V[idx_vj, :]
        Wv = W[idx_wk, :]
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        for t in range(num_entries):
            r = residuals[t]
            ui = idx_ui[t]
            vj = idx_vj[t]
            wk = idx_wk[t]
            grad_U[ui, :] += r * Vv[t] * Wv[t]
            grad_V[vj, :] += r * Uv[t] * Wv[t]
            grad_W[wk, :] += r * Uv[t] * Vv[t]
        return grad_U, grad_V, grad_W

    def try_snap(U, V, W):
        for scale in [1, 2, 4, 8, 16]:
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if verify_decomposition(U_s, V_s, W_s):
                return U_s, V_s, W_s
        return None

    def optimize(U, V, W, max_steps, deadline, lr_init=0.1):
        """Adam optimizer with adaptive learning rate."""
        lr = lr_init
        m_U = np.zeros_like(U); v_U = np.zeros_like(U)
        m_V = np.zeros_like(V); v_V = np.zeros_like(V)
        m_W = np.zeros_like(W); v_W = np.zeros_like(W)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        best_loss = np.inf
        best_U = U.copy(); best_V = V.copy(); best_W = W.copy()

        for step in range(max_steps):
            if time.time() > deadline:
                break

            vals = compute_values(U, V, W)
            residuals = vals - 1.0
            loss = 0.5 * np.sum(residuals ** 2)

            if loss < 1e-16:
                return U, V, W, True

            if loss < best_loss:
                best_loss = loss
                best_U = U.copy(); best_V = V.copy(); best_W = W.copy()

            grad_U, grad_V, grad_W = compute_gradients(U, V, W, vals)

            t = step + 1
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
            U -= lr * (m_U / (1 - beta1**t)) / (np.sqrt(v_U / (1 - beta2**t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
            V -= lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)

            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
            W -= lr * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)

            # Adaptive LR: decrease if stuck
            if step > 50 and loss > 0.001:
                lr = max(0.005, lr * 0.995)

        vals = compute_values(best_U, best_V, best_W)
        loss = 0.5 * np.sum((vals - 1.0) ** 2)
        return best_U, best_V, best_W, loss < 1e-14

    def random_init(target_rank, seed):
        np.random.seed(seed)
        U = np.random.randn(nm, target_rank) * 0.3
        V = np.random.randn(mp, target_rank) * 0.3
        W = np.random.randn(pn, target_rank) * 0.3
        for col in range(min(4, target_rank)):
            U[:, col] += np.random.choice([-1, 0, 1], size=nm) * 0.5
            V[:, col] += np.random.choice([-1, 0, 1], size=mp) * 0.5
            W[:, col] += np.random.choice([-1, 0, 1], size=pn) * 0.5
        return U, V, W

    # Build rank-28 Strassen fallback
    rank_28 = 28
    U_28 = np.zeros((nm, rank_28), dtype=float)
    V_28 = np.zeros((mp, rank_28), dtype=float)
    W_28 = np.zeros((pn, rank_28), dtype=float)

    def strassen_terms():
        terms = []
        terms.append(({0:1,3:1}, {0:1,3:1}, {0:1,3:1}))
        terms.append(({2:1,3:1}, {0:1}, {2:1,3:-1}))
        terms.append(({0:1}, {1:1,3:-1}, {1:1,3:1}))
        terms.append(({3:1}, {2:1,0:-1}, {0:1,2:1}))
        terms.append(({0:1,1:1}, {3:1}, {0:-1,1:1}))
        terms.append(({2:1,0:-1}, {0:1,1:1}, {3:1}))
        terms.append(({1:1,3:-1}, {2:1,3:1}, {0:1}))
        return terms

    base_terms = strassen_terms()

    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col, U, V, W):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2; c = loc_idx % 2
                g_idx = r * 4 + (a_offset * 2 + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2; c = loc_idx % 2
                g_idx = (b_row_offset * 2 + r) * 4 + (b_col_offset * 2 + c)
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2; c = loc_idx % 2
                g_idx = (c_col_offset * 2 + c) * 2 + r
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0, U_28, V_28, W_28)
    apply_strassen(1, 1, 0, 0, 7, U_28, V_28, W_28)
    apply_strassen(0, 0, 1, 1, 14, U_28, V_28, W_28)
    apply_strassen(1, 1, 1, 1, 21, U_28, V_28, W_28)

    best_rank = rank_28
    best_U = U_28.copy()
    best_V = V_28.copy()
    best_W = W_28.copy()

    start_time = time.time()
    deadline = start_time + 135

    # Strategy 1: Try to reduce rank-28 via optimization
    if time.time() < deadline - 5:
        U_r, V_r, W_r, success = optimize(U_28.copy(), V_28.copy(), W_28.copy(), 3000, deadline, lr_init=0.05)
        if success:
            result = try_snap(U_r, V_r, W_r)
            if result is not None:
                # Try to reduce rank further
                for target_rank in range(15, best_rank):
                    if time.time() > deadline - 5:
                        break
                    U_sub = U_r[:, :target_rank]
                    V_sub = V_r[:, :target_rank]
                    W_sub = W_r[:, :target_rank]
                    if verify_decomposition(U_sub, V_sub, W_sub):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                        break

    # Strategy 2: Direct random search for low ranks
    for target_rank in range(15, min(25, best_rank)):
        if time.time() > deadline - 10:
            break

        num_trials = 300 if target_rank <= 17 else 200 if target_rank <= 19 else 100

        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break

            seed = 1000 + target_rank * 10000 + trial
            U, V, W = random_init(target_rank, seed)
            U, V, W, success = optimize(U, V, W, 5000, deadline - 5, lr_init=0.08)

            if success:
                result = try_snap(U, V, W)
                if result is not None:
                    best_rank = target_rank
                    best_U, best_V, best_W = result
                    break
                if verify_decomposition(U, V, W):
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break

    # Final polish: try to snap and verify
    final_result = try_snap(best_U, best_V, best_W)
    if final_result is not None:
        best_U, best_V, best_W = final_result

    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                elif abs(x * 2 - round(x * 2)) < 1e-9:
                    new_row.append(round(x * 2) / 2)
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
