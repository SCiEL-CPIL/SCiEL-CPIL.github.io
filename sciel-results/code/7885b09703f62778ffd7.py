"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def try_rank_direct(rank, max_restarts=20, max_iters=500):
        """Try to find a half-integer decomposition of given rank using structured initialization."""
        if time.time() > deadline - 10:
            return None

        np.random.seed(42)

        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return None

            # Initialize with random small integers
            U = np.random.choice([-1, 0, 1], size=(n*m, rank)).astype(float)
            V = np.random.choice([-1, 0, 1], size=(m*p, rank)).astype(float)
            W = np.random.choice([-1, 0, 1], size=(p*n, rank)).astype(float)

            # Gradient descent with adaptive learning rate
            lr = 0.05
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    return None

                # Compute gradients for U
                grad_U = np.zeros((n*m, rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            prod = V[v_idx, :] * W[w_idx, :]
                            val = np.dot(U[u_idx, :] * prod, np.ones(rank))
                            grad_U[u_idx, :] += (val - 1.0) * prod

                # Compute gradients for V
                grad_V = np.zeros((m*p, rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            prod = U[u_idx, :] * W[w_idx, :]
                            val = np.dot(V[v_idx, :] * prod, np.ones(rank))
                            grad_V[v_idx, :] += (val - 1.0) * prod

                # Compute gradients for W
                grad_W = np.zeros((p*n, rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            prod = U[u_idx, :] * V[v_idx, :]
                            val = np.dot(W[w_idx, :] * prod, np.ones(rank))
                            grad_W[w_idx, :] += (val - 1.0) * prod

                # Update with adaptive learning rate
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W

                # Compute loss
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                            loss += (val - 1.0) ** 2

                if loss < 1e-8:
                    U_snap = snap_to_half_int(U)
                    V_snap = snap_to_half_int(V)
                    W_snap = snap_to_half_int(W)
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}

                # Decay learning rate
                lr *= 0.99

        return None

    def try_rank_kronecker(rank_total):
        """Try to find a decomposition using Kronecker product structure."""
        if time.time() > deadline - 10:
            return None

        # M_{2,3,9} = M_{2,3,3} ⊗ M_{1,1,3}
        # Find rank r solution for M_{2,3,3}, then construct rank 3r solution for M_{2,3,9}
        best_r = 3 * 18  # schoolbook for M_{2,3,3} is 18
        for r in range(10, 19):
            if time.time() > deadline - 10:
                return None

            # Search for M_{2,3,3} rank r
            n2, m2, p2 = 2, 3, 3
            found = False
            for restart in range(15):
                if time.time() > deadline - 10:
                    return None

                U2 = np.random.choice([-1, 0, 1], size=(n2*m2, r)).astype(float)
                V2 = np.random.choice([-1, 0, 1], size=(m2*p2, r)).astype(float)
                W2 = np.random.choice([-1, 0, 1], size=(p2*n2, r)).astype(float)

                lr2 = 0.05
                for it in range(300):
                    if time.time() > deadline - 10:
                        return None

                    grad_U2 = np.zeros((n2*m2, r))
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                u_idx = i * m2 + j
                                v_idx = j * p2 + k
                                w_idx = k * n2 + i
                                prod = V2[v_idx, :] * W2[w_idx, :]
                                val = np.dot(U2[u_idx, :] * prod, np.ones(r))
                                grad_U2[u_idx, :] += (val - 1.0) * prod

                    grad_V2 = np.zeros((m2*p2, r))
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                u_idx = i * m2 + j
                                v_idx = j * p2 + k
                                w_idx = k * n2 + i
                                prod = U2[u_idx, :] * W2[w_idx, :]
                                val = np.dot(V2[v_idx, :] * prod, np.ones(r))
                                grad_V2[v_idx, :] += (val - 1.0) * prod

                    grad_W2 = np.zeros((p2*n2, r))
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                u_idx = i * m2 + j
                                v_idx = j * p2 + k
                                w_idx = k * n2 + i
                                prod = U2[u_idx, :] * V2[v_idx, :]
                                val = np.dot(W2[w_idx, :] * prod, np.ones(r))
                                grad_W2[w_idx, :] += (val - 1.0) * prod

                    U2 -= lr2 * grad_U2
                    V2 -= lr2 * grad_V2
                    W2 -= lr2 * grad_W2

                    loss2 = 0.0
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                u_idx = i * m2 + j
                                v_idx = j * p2 + k
                                w_idx = k * n2 + i
                                val = np.dot(U2[u_idx, :] * V2[v_idx, :], W2[w_idx, :])
                                loss2 += (val - 1.0) ** 2

                    if loss2 < 1e-8:
                        U2_snap = snap_to_half_int(U2)
                        V2_snap = snap_to_half_int(V2)
                        W2_snap = snap_to_half_int(W2)
                        U2_list = U2_snap.tolist()
                        V2_list = V2_snap.tolist()
                        W2_list = W2_snap.tolist()

                        # Verify M_{2,3,3}
                        ok = True
                        for i in range(n2):
                            for j in range(m2):
                                for k in range(p2):
                                    u = np.array(U2_list[i*m2+j])
                                    v = np.array(V2_list[j*p2+k])
                                    w = np.array(W2_list[k*n2+i])
                                    val = np.dot(u * v, w)
                                    if abs(val - 1.0) > 1e-9:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                break
                        if ok:
                            best_r = r
                            found = True
                            break

                    lr2 *= 0.99

                if found:
                    break

            if found and best_r < 3 * 18:
                # Construct M_{2,3,9} from M_{2,3,3}
                r_total = 3 * best_r
                U = [[0] * r_total for _ in range(n*m)]
                V = [[0] * r_total for _ in range(m*p)]
                W = [[0] * r_total for _ in range(p*n)]

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            k1 = k % 3
                            k2 = k // 3
                            for c in range(best_r):
                                col = c * 3 + k2
                                U[i*m+j][col] = U2_list[i*m2+j][c]
                                V[j*p+k][col] = V2_list[j*p2+k1][c]
                                W[k*n+i][col] = W2_list[k1*n2+i][c]

                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}

        return None

    # Try Kronecker product approach first
    res = try_rank_kronecker(45)
    if res is not None:
        return res

    # Try direct search for ranks less than 45
    for rank in range(30, 45):
        if time.time() > deadline - 10:
            break
        res = try_rank_direct(rank, max_restarts=10, max_iters=300)
        if res is not None:
            return res

    return schoolbook()
# EVOLVE-BLOCK-END
