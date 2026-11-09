import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomp(U, V, W, tol=1e-10):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def trivial_decomp():
        U = [[0] * 27 for _ in range(9)]
        V = [[0] * 27 for _ in range(9)]
        W = [[0] * 27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find rank-23 decomposition using ALS with multiple restarts
    target_rank = 23
    best_result = None

    for restart in range(30):
        if time.time() > deadline - 15:
            break

        np.random.seed(42 + restart)
        U = np.random.randn(9, target_rank) * 0.5
        V = np.random.randn(9, target_rank) * 0.5
        W = np.random.randn(9, target_rank) * 0.5

        for iteration in range(800):
            if time.time() > deadline - 10:
                break

            # Update U
            for i in range(3):
                for j in range(3):
                    A = np.zeros((target_rank, 3))
                    for r in range(target_rank):
                        for k in range(3):
                            A[r, k] = V[j*3+k, r] * W[k*3+i, r]
                    try:
                        ATA = A @ A.T + 1e-8 * np.eye(target_rank)
                        U[i*3+j, :] = np.linalg.solve(ATA, A @ np.ones(3))
                    except:
                        pass

            # Update V
            for j in range(3):
                for k in range(3):
                    B = np.zeros((target_rank, 3))
                    for r in range(target_rank):
                        for i in range(3):
                            B[r, i] = U[i*3+j, r] * W[k*3+i, r]
                    try:
                        BTB = B @ B.T + 1e-8 * np.eye(target_rank)
                        V[j*3+k, :] = np.linalg.solve(BTB, B @ np.ones(3))
                    except:
                        pass

            # Update W
            for k in range(3):
                for i in range(3):
                    C = np.zeros((target_rank, 3))
                    for r in range(target_rank):
                        for j in range(3):
                            C[r, j] = U[i*3+j, r] * V[j*3+k, r]
                    try:
                        CTC = C @ C.T + 1e-8 * np.eye(target_rank)
                        W[k*3+i, :] = np.linalg.solve(CTC, C @ np.ones(3))
                    except:
                        pass

            if iteration % 50 == 0:
                max_err = 0.0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                            max_err = max(max_err, abs(s - 1.0))
                if max_err < 1e-9:
                    break

        if verify_decomp(U.tolist(), V.tolist(), W.tolist()):
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break

    if best_result is not None:
        return best_result

    return trivial_decomp()
