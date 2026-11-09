import numpy as np
import time

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook decomposition (rank 90)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Try to find lower rank using alternating minimization
    for target_rank in range(85, 69, -1):
        if time.time() > deadline - 30:
            break

        found = False
        for trial in range(3):
            if time.time() > deadline - 25:
                break

            np.random.seed(42 + target_rank * 1000 + trial)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)

            prev_err = float('inf')
            for it in range(300):
                if time.time() > deadline - 20:
                    break

                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :], *_ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :], *_ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :], *_ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        except:
                            pass

                # Compute error
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = abs(val - 1)
                            if err > max_err:
                                max_err = err

                if max_err < 1e-9:
                    found = True
                    break

                if max_err > prev_err * 1.1:
                    break
                prev_err = max_err

            if found:
                break

        if found:
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break

    U_list = [[float(best_U[i, j]) for j in range(best_U.shape[1])] for i in range(best_U.shape[0])]
    V_list = [[float(best_V[i, j]) for j in range(best_V.shape[1])] for i in range(best_V.shape[0])]
    W_list = [[float(best_W[i, j]) for j in range(best_W.shape[1])] for i in range(best_W.shape[0])]

    return {"U": U_list, "V": V_list, "W": W_list}
