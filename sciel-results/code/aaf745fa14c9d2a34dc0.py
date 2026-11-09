import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    def compute_error(U, V, W):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(s - 1.0))
        return max_err

    def verify_decomp(U, V, W, tol=1e-10):
        return compute_error(U, V, W) < tol

    def polish(U, V, W, max_iters=200, tol=1e-12):
        """High-precision polish pass with tighter tolerance."""
        rank = len(U[0])
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            prev_err = compute_error(U, V, W)
            if prev_err < tol:
                break

            # Update U with tight regularization
            for i in range(n):
                for j in range(m):
                    A = np.zeros((rank, p))
                    for r in range(rank):
                        for k in range(p):
                            A[r, k] = V[j*p+k, r] * W[k*n+i, r]
                    try:
                        reg = 1e-10 * np.eye(rank)
                        U[i*m+j, :] = np.linalg.solve(A @ A.T + reg, A @ np.ones(p))
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((rank, n))
                    for r in range(rank):
                        for i in range(n):
                            B[r, i] = U[i*m+j, r] * W[k*n+i, r]
                    try:
                        reg = 1e-10 * np.eye(rank)
                        V[j*p+k, :] = np.linalg.solve(B @ B.T + reg, B @ np.ones(n))
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for r in range(rank):
                        for j in range(m):
                            C[r, j] = U[i*m+j, r] * V[j*p+k, r]
                    try:
                        reg = 1e-10 * np.eye(rank)
                        W[k*n+i, :] = np.linalg.solve(C @ C.T + reg, C @ np.ones(m))
                    except:
                        pass

            cur_err = compute_error(U, V, W)
            if cur_err >= prev_err * 0.999 and it > 10:
                # Convergence plateau
                if cur_err < tol * 10:
                    break

        return U, V, W

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

    target_rank = 23
    best_result = None
    best_err = float('inf')

    for restart in range(50):
        if time.time() > deadline - 20:
            break

        np.random.seed(42 + restart)
        U = np.random.randn(9, target_rank) * 0.5
        V = np.random.randn(9, target_rank) * 0.5
        W = np.random.randn(9, target_rank) * 0.5

        # Coarse ALS phase
        for iteration in range(500):
            if time.time() > deadline - 15:
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

            if iteration % 25 == 0:
                err = compute_error(U, V, W)
                if err < 1e-9:
                    break
                if err < best_err:
                    best_err = err

        # Fine polish phase
        U, V, W = polish(U, V, W, max_iters=300, tol=1e-12)
        err = compute_error(U, V, W)

        if err < 1e-10:
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break

        if err < best_err:
            best_err = err

    if best_result is not None:
        return best_result

    return trivial_decomp()
