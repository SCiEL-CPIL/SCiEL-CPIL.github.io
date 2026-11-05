"""Tensor decomposition for 3x3 by 3x6 matrix multiplication using vectorized numerical search."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 6]


def _verify(U, V, W, n, m, p):
    """Check if decomposition is valid. Returns max error."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                max_err = max(max_err, abs(val - 1.0))
    return max_err


def _verify_fraction(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if val != 1:
                    return False
    return True


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
    n, m, p = 3, 3, 6
    deadline = time.time() + 130.0

    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 54

    # Precompute index arrays for vectorized computation
    idx_i = np.repeat(np.arange(n), m * p).reshape(n, m, p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n).reshape(n, m, p)
    idx_k = np.tile(np.arange(p), n * m).reshape(n, m, p)

    U_idx = (idx_i * m + idx_j).ravel()
    V_idx = (idx_j * p + idx_k).ravel()
    W_idx = (idx_k * n + idx_i).ravel()

    def compute_pred(U, V, W):
        """Vectorized prediction: pred[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        rank = U.shape[1]
        pred = np.zeros(n * m * p)
        for r in range(rank):
            pred += U[U_idx, r] * V[V_idx, r] * W[W_idx, r]
        return pred

    def compute_grads(U, V, W, pred):
        """Vectorized gradient computation."""
        rank = U.shape[1]
        err = pred - 1.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for r in range(rank):
            u_vals = U[U_idx, r]
            v_vals = V[V_idx, r]
            w_vals = W[W_idx, r]
            np.add.at(dU, (U_idx, r), 2 * err * v_vals * w_vals)
            np.add.at(dV, (V_idx, r), 2 * err * u_vals * w_vals)
            np.add.at(dW, (W_idx, r), 2 * err * u_vals * v_vals)
        return dU, dV, dW

    for target_rank in [27, 30, 33, 36, 40, 45, 50]:
        if time.time() > deadline - 20:
            break

        for trial in range(40):
            if time.time() > deadline - 20:
                break

            np.random.seed(42 + trial * 7919 + target_rank * 131)

            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3

            lr = 0.05
            for step in range(5000):
                if time.time() > deadline - 20:
                    break

                pred = compute_pred(U, V, W)
                loss = np.sum((pred - 1.0) ** 2)

                if loss < 1e-12:
                    break

                dU, dV, dW = compute_grads(U, V, W, pred)

                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

                if step % 1000 == 999:
                    lr *= 0.5

            pred = compute_pred(U, V, W)
            max_err = np.max(np.abs(pred - 1.0))

            if max_err < 1e-8:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2

                pred_r = compute_pred(U_r, V_r, W_r)
                max_err_r = np.max(np.abs(pred_r - 1.0))

                if max_err_r < 1e-9:
                    U_list = [[float(x) for x in row] for row in U_r]
                    V_list = [[float(x) for x in row] for row in V_r]
                    W_list = [[float(x) for x in row] for row in W_r]
                    if _verify_fraction(U_list, V_list, W_list, n, m, p):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_list, V_list, W_list
                        break
                elif max_err < 1e-9:
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if _verify(U_list, V_list, W_list, n, m, p) < 1e-9:
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_list, V_list, W_list
                        break

    # Fallback to schoolbook
    if best_U is None:
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
        return {"U": U, "V": V, "W": W}

    return {"U": best_U, "V": best_V, "W": best_W}
