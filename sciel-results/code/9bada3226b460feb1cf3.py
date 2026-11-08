"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    nm = n * m   # 6
    mp = m * p   # 9
    pn = p * n   # 6
    R = n * m * p  # 18

    def check(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    # Schoolbook solution
    U = [[0.0]*R for _ in range(nm)]
    V = [[0.0]*R for _ in range(mp)]
    W = [[0.0]*R for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1

    best_rank = R
    best_U, best_V, best_W = U, V, W

    # Search for lower-rank solutions by removing columns and re-optimizing
    for target_rank in range(14, 18):
        if time.time() > deadline - 20:
            break

        num_remove = R - target_rank
        found = False

        for trial in range(150):
            if time.time() > deadline - 15:
                break

            remove = set(random.sample(range(R), num_remove))

            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)

            for c in remove:
                U_arr[:, c] = 0.0
                V_arr[:, c] = 0.0
                W_arr[:, c] = 0.0

            # Gradient-based optimization
            for it in range(2500):
                if time.time() > deadline - 5:
                    break

                # Compute residuals and check convergence
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(R):
                                s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                            res = s - 1.0
                            if abs(res) > max_res:
                                max_res = abs(res)

                if max_res < 1e-10:
                    break

                lr = 0.004 / (1.0 + it * 0.0003)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(R):
                                s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                            res = s - 1.0
                            if abs(res) < 1e-10:
                                continue
                            for c in range(R):
                                if c in remove:
                                    continue
                                U_arr[i*m+j, c] -= lr * res * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                                V_arr[j*p+k, c] -= lr * res * U_arr[i*m+j, c] * W_arr[k*n+i, c]
                                W_arr[k*n+i, c] -= lr * res * U_arr[i*m+j, c] * V_arr[j*p+k, c]

            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(R):
                            s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                        if abs(s - 1.0) > 1e-8:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break

            if ok:
                active = [c for c in range(R) if c not in remove]
                U_p = U_arr[:, active].tolist()
                V_p = V_arr[:, active].tolist()
                W_p = W_arr[:, active].tolist()
                if len(active) < best_rank and check(U_p, V_p, W_p, len(active)):
                    best_rank = len(active)
                    best_U, best_V, best_W = U_p, V_p, W_p
                    found = True
                    break

        if found and best_rank <= target_rank:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
