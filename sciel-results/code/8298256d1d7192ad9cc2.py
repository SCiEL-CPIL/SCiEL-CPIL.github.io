import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 21, 21

    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Trivial decomposition (rank 63)
    rank_trivial = nm * mp * pn  # 63
    U_triv = np.zeros((nm, rank_trivial))
    V_triv = np.zeros((mp, rank_trivial))
    W_triv = np.zeros((pn, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, col] = 1
                V_triv[j * p + k, col] = 1
                W_triv[k * n + i, col] = 1
                col += 1

    best_result = (U_triv, V_triv, W_triv)
    best_rank = rank_trivial

    # Try to find lower-rank decomposition using ALS
    for target_rank in range(50, 63):
        if time.time() > deadline - 30:
            break

        max_trials = 3
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                break

            # Initialize from trivial decomposition with random column selection + noise
            selected_cols = np.random.choice(rank_trivial, target_rank, replace=False)
            U_curr = U_triv[:, selected_cols] + 0.1 * np.random.randn(nm, target_rank)
            V_curr = V_triv[:, selected_cols] + 0.1 * np.random.randn(mp, target_rank)
            W_curr = W_triv[:, selected_cols] + 0.1 * np.random.randn(pn, target_rank)

            # ALS iterations
            for iteration in range(30):
                if time.time() > deadline - 15:
                    break

                # Update U
                for a in range(nm):
                    i, j = divmod(a, m)
                    M_a = np.zeros((p, target_rank))
                    for k in range(p):
                        M_a[k, :] = V_curr[j * p + k, :] * W_curr[k * n + i, :]
                    try:
                        U_curr[a, :] = np.linalg.lstsq(M_a.T, np.ones(p), rcond=None)[0]
                    except:
                        pass

                # Update V
                for b in range(mp):
                    j, k = divmod(b, p)
                    M_b = np.zeros((n, target_rank))
                    for i in range(n):
                        M_b[i, :] = U_curr[i * m + j, :] * W_curr[k * n + i, :]
                    try:
                        V_curr[b, :] = np.linalg.lstsq(M_b.T, np.ones(n), rcond=None)[0]
                    except:
                        pass

                # Update W
                for c in range(pn):
                    k, i = divmod(c, n)
                    M_c = np.zeros((m, target_rank))
                    for j in range(m):
                        M_c[j, :] = U_curr[i * m + j, :] * V_curr[j * p + k, :]
                    try:
                        W_curr[c, :] = np.linalg.lstsq(M_c.T, np.ones(m), rcond=None)[0]
                    except:
                        pass

            # Verify
            T_comp = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
            if np.allclose(T_comp, T_target, atol=1e-6):
                best_rank = target_rank
                best_result = (U_curr, V_curr, W_curr)
                break

        if best_rank < rank_trivial:
            break

    U_final, V_final, W_final = best_result
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
