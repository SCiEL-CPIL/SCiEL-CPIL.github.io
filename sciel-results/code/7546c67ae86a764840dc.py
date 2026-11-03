import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor T[i,j,k] = 1 at position (i*m+j, j*p+k, k*n+i)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify(U, V, W, r, tol=1e-10):
        """Compute max absolute error of decomposition."""
        T_check = np.einsum('ir,jr,kr->ijk', U[:, :r], V[:, :r], W[:, :r])
        return np.max(np.abs(T_check - T))

    def rank55_decomposition():
        """Hand-constructed rank-55: j=0 naive + Strassen on 5 blocks of 2x2 for j=1,2."""
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        col = 0

        # Part A: j=0 (2 x 10 = 20 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2

            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1

            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1

            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1

            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1

            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1

            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1

            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1

            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1

        return U, V, W, rank

    def als_step(U, V, W, r):
        """One vectorized ALS step using pseudoinverse."""
        # Update U: for each row i of U, solve min ||V*W - T_i||
        # T.reshape(nm, mp*pn) @ pinv((V⊗W).T)
        VW = np.einsum('jr,kr->jkr', V[:, :r], W[:, :r]).reshape(mp * pn, r)
        try:
            U_new = T.reshape(nm, mp * pn) @ np.linalg.pinv(VW.T)
        except:
            U_new = U.copy()

        # Update V
        UW = np.einsum('ir,kr->ikr', U_new[:, :r], W[:, :r]).reshape(nm * pn, r)
        try:
            V_new = T.reshape(mp, nm * pn) @ np.linalg.pinv(UW.T)
        except:
            V_new = V.copy()

        # Update W
        UV = np.einsum('ir,jr->ijr', U_new[:, :r], V_new[:, :r]).reshape(nm * mp, r)
        try:
            W_new = T.reshape(pn, nm * mp) @ np.linalg.pinv(UV.T)
        except:
            W_new = W.copy()

        return U_new, V_new, W_new

    # Start with known rank-55 solution
    U_best, V_best, W_best, rank_best = rank55_decomposition()
    err = verify(U_best, V_best, W_best, rank_best)

    # Try to reduce rank using vectorized ALS
    if time.time() < deadline - 10:
        for target_rank in range(rank_best - 1, 40, -1):
            if time.time() > deadline - 10:
                break

            found = False
            max_restarts = 15 if target_rank >= 50 else 10

            for trial in range(max_restarts):
                if time.time() > deadline - 10:
                    break

                np.random.seed(trial * 7919 + target_rank * 13)

                # Initialize near the rank-55 solution with noise
                if target_rank <= rank_best:
                    U_c = U_best[:, :target_rank].copy()
                    V_c = V_best[:, :target_rank].copy()
                    W_c = W_best[:, :target_rank].copy()

                    noise_scale = 0.05 + 0.02 * trial
                    U_c += noise_scale * np.random.randn(nm, target_rank)
                    V_c += noise_scale * np.random.randn(mp, target_rank)
                    W_c += noise_scale * np.random.randn(pn, target_rank)
                else:
                    U_c = np.random.randn(nm, target_rank) * 0.1
                    V_c = np.random.randn(mp, target_rank) * 0.1
                    W_c = np.random.randn(pn, target_rank) * 0.1

                best_err = verify(U_c, V_c, W_c, target_rank)
                best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()

                # Vectorized ALS iterations
                max_iters = 200
                for iteration in range(max_iters):
                    if time.time() > deadline - 10:
                        break

                    U_c, V_c, W_c = als_step(U_c, V_c, W_c, target_rank)

                    err_val = verify(U_c, V_c, W_c, target_rank)

                    if err_val < best_err:
                        best_err = err_val
                        best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()

                    if best_err < 1e-10:
                        # Found a valid decomposition at this rank
                        U_best, V_best, W_best, rank_best = best_U, best_V, best_W, target_rank
                        found = True
                        break
                    elif iteration > 50 and best_err > 0.5:
                        # Stuck, break early
                        break

                if found:
                    break

            if found:
                break

    # Final verification
    final_err = verify(U_best, V_best, W_best, rank_best)
    if final_err > 1e-8:
        # Fall back to rank-55
        U_best, V_best, W_best, rank_best = rank55_decomposition()

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
