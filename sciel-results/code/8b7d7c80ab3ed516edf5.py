import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8

    def verify_decomposition(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    acc = 0.0
                    for r in range(U.shape[1]):
                        acc += U[ui, r] * V[vj, r] * W[wk, r]
                    if abs(acc - 1.0) > tol:
                        return False
        return True

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

    start_time = time.time()
    deadline = start_time + 145

    # Strategy 1: Direct rank-16 construction (two independent 2x2 x 2x4 multiplications)
    # A is 2x4, B is 4x4
    # Split B into B_top (rows 0,1) and B_bot (rows 2,3)
    # Split A into A_left (cols 0,1) and A_right (cols 2,3)
    # C = A_left @ B_top + A_right @ B_bot
    # Each 2x2 @ 2x4 requires 8 scalar multiplications (direct)
    # Total = 16 rank-1 terms

    rank_16 = 16
    U_16 = np.zeros((nm, rank_16), dtype=float)
    V_16 = np.zeros((mp, rank_16), dtype=float)
    W_16 = np.zeros((pn, rank_16), dtype=float)

    # First 8 terms: A_left (cols 0,1) @ B_top (rows 0,1)
    # For i in {0,1}, j in {0,1}, k in {0,1,2,3}:
    #   C[i,k] += A[i,j] * B[j,k]
    term_idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                U_16[ui, term_idx] = 1
                V_16[vj, term_idx] = 1
                W_16[wk, term_idx] = 1
                term_idx += 1

    # Next 8 terms: A_right (cols 2,3) @ B_bot (rows 2,3)
    # For i in {0,1}, j in {2,3}, k in {0,1,2,3}:
    #   C[i,k] += A[i,j] * B[j,k]
    for i in range(2):
        for j in range(2, 4):
            for k in range(4):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                U_16[ui, term_idx] = 1
                V_16[vj, term_idx] = 1
                W_16[wk, term_idx] = 1
                term_idx += 1

    # Verify rank-16 construction
    assert verify_decomposition(U_16, V_16, W_16), "Rank-16 construction failed verification"

    best_rank = 16
    best_U = U_16.copy()
    best_V = V_16.copy()
    best_W = W_16.copy()

    # Strategy 2: Try to find a lower rank via random search + optimization
    # The theoretical lower bound for 2x4x4 is at least 12 (by various bounds)
    # Let's try ranks 12-15 with random initialization and gradient descent

    def compute_values(U, V, W):
        vals = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    vals[idx] = np.dot(U[ui, :], V[vj, :] * W[wk, :])
                    idx += 1
        return vals

    def compute_gradients(U, V, W, vals):
        residuals = vals - 1.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    r = residuals[idx]
                    grad_U[ui, :] += r * (V[vj, :] * W[wk, :])
                    grad_V[vj, :] += r * (U[ui, :] * W[wk, :])
                    grad_W[wk, :] += r * (U[ui, :] * V[vj, :])
                    idx += 1
        return grad_U, grad_V, grad_W

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

            if step > 50 and loss > 0.001:
                lr = max(0.005, lr * 0.995)

        vals = compute_values(best_U, best_V, best_W)
        loss = 0.5 * np.sum((vals - 1.0) ** 2)
        return best_U, best_V, best_W, loss < 1e-14

    def try_snap(U, V, W):
        for scale in [1, 2, 4, 8, 16]:
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if verify_decomposition(U_s, V_s, W_s):
                return U_s, V_s, W_s
        return None

    # Try to find rank 12-15
    for target_rank in range(12, 16):
        if time.time() > deadline - 10:
            break

        num_trials = 20 if target_rank <= 13 else 15
        found = False

        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break

            np.random.seed(10000 + target_rank * 1000 + trial)
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5

            # Add some structured initialization
            for col in range(min(3, target_rank)):
                U[:, col] += np.random.choice([-1, 0, 1], size=nm) * 0.3
                V[:, col] += np.random.choice([-1, 0, 1], size=mp) * 0.3
                W[:, col] += np.random.choice([-1, 0, 1], size=pn) * 0.3

            U_r, V_r, W_r, success = optimize(U, V, W, 3000, deadline - 5, lr_init=0.08)

            if success:
                result = try_snap(U_r, V_r, W_r)
                if result is not None:
                    best_rank = target_rank
                    best_U, best_V, best_W = result
                    found = True
                    break
                if verify_decomposition(U_r, V_r, W_r):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    found = True
                    break

        if found:
            break

    # Final verification
    final_result = try_snap(best_U, best_V, best_W)
    if final_result is not None:
        best_U, best_V, best_W = final_result

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
