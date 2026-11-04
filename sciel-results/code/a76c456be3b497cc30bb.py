import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 150s kill

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try to find rank-19 decomposition via gradient descent + rounding
    target_rank = 19
    np.random.seed(42)

    for attempt in range(300):
        if time.time() > deadline - 10:
            break

        # Random initialization
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5

        # Gradient descent
        lr = 0.02
        for step in range(3000):
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            loss = 0.0

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        uv = U[i * m + j] * V[j * p + k]
                        val = np.dot(uv, W[k * n + i])
                        err = val - 1.0
                        loss += err * err
                        dU[i * m + j] += err * V[j * p + k] * W[k * n + i]
                        dV[j * p + k] += err * U[i * m + j] * W[k * n + i]
                        dW[k * n + i] += err * U[i * m + j] * V[j * p + k]

            U -= lr * dU
            V -= lr * dV
            W -= lr * dW

            if loss < 1e-14:
                break

        if loss > 1e-10:
            continue

        # Round to half-integers
        U_h = np.round(U * 2) / 2.0
        V_h = np.round(V * 2) / 2.0
        W_h = np.round(W * 2) / 2.0

        # Verify exact solution
        exact = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_h[i * m + j] * V_h[j * p + k], W_h[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        exact = False
                        break
                if not exact:
                    break
            if not exact:
                break

        if exact:
            return {
                "U": U_h.tolist(),
                "V": V_h.tolist(),
                "W": W_h.tolist()
            }

    return {"U": U_school, "V": V_school, "W": W_school}
