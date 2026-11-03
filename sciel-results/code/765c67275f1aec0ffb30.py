import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 110  # safety margin before 150s kill

    # Build the full target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Strategy 1: Block decomposition - 2x3x9 = 3 independent 2x3x3 blocks
    # Search for rank-r decomposition of 2x3x3, then replicate 3x
    n2, m2, p2 = 2, 3, 3
    T2 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i*m2+j, j*p2+k, k*n2+i] = 1.0

    # Try ranks for 2x3x3: known lower bound is around 7-8, schoolbook is 18
    for r_block in range(7, 19):
        if time.time() > deadline - 20:
            break
        r_full = r_block * 3
        if r_full >= best_rank:
            break

        # Try multiple random initializations for 2x3x3
        for trial in range(30):
            if time.time() > deadline - 20:
                break

            np.random.seed(42 + r_block * 1000 + trial)
            U2 = np.random.randn(n2*m2, r_block)
            V2 = np.random.randn(m2*p2, r_block)
            W2 = np.random.randn(p2*n2, r_block)

            # ALS iterations for 2x3x3
            success = False
            for iteration in range(300):
                if time.time() > deadline - 20:
                    break

                # Update U2
                for a in range(n2*m2):
                    M_a = np.zeros((m2*p2*p2*n2, r_block))
                    for b in range(m2*p2):
                        for c in range(p2*n2):
                            M_a[b*(p2*n2)+c, :] = V2[b, :] * W2[c, :]
                    target = T2[a, :, :].flatten()
                    try:
                        U2[a, :], _, _, _ = np.linalg.lstsq(M_a.T, target, rcond=None)
                    except:
                        pass

                # Update V2
                for b in range(m2*p2):
                    M_b = np.zeros((n2*m2*p2*n2, r_block))
                    for a in range(n2*m2):
                        for c in range(p2*n2):
                            M_b[a*(p2*n2)+c, :] = U2[a, :] * W2[c, :]
                    target = T2[:, b, :].reshape(n2*m2, p2*n2).flatten()
                    try:
                        V2[b, :], _, _, _ = np.linalg.lstsq(M_b.T, target, rcond=None)
                    except:
                        pass

                # Update W2
                for c in range(p2*n2):
                    M_c = np.zeros((n2*m2*m2*p2, r_block))
                    for a in range(n2*m2):
                        for b in range(m2*p2):
                            M_c[a*(m2*p2)+b, :] = U2[a, :] * V2[b, :]
                    target = T2[:, :, c].reshape(n2*m2, m2*p2).flatten()
                    try:
                        W2[c, :], _, _, _ = np.linalg.lstsq(M_c.T, target, rcond=None)
                    except:
                        pass

                # Check convergence
                error = 0.0
                for a in range(n2*m2):
                    for b in range(m2*p2):
                        for c in range(p2*n2):
                            val = np.dot(U2[a, :], V2[b, :] * W2[c, :])
                            error += (val - T2[a, b, c])**2

                if error < 1e-10:
                    success = True
                    break
                elif error > 1e4:
                    break

            if success:
                # Build full 2x3x9 decomposition from 3 blocks
                U_full = np.zeros((n*m, r_full))
                V_full = np.zeros((m*p, r_full))
                W_full = np.zeros((p*n, r_full))

                # U is same for all 3 blocks (depends only on i,j)
                U_full[:, 0:r_block] = U2
                U_full[:, r_block:2*r_block] = U2
                U_full[:, 2*r_block:3*r_block] = U2

                # V depends on (j,k): row index = j*9+k
                for j in range(m):
                    for k in range(p):
                        v_row = j * p + k
                        block = k // p2
                        k_local = k % p2
                        v_233_row = j * p2 + k_local
                        col_start = block * r_block
                        V_full[v_row, col_start:col_start+r_block] = V2[v_233_row, :]

                # W depends on (k,i): row index = k*2+i
                for k in range(p):
                    for i in range(n):
                        w_row = k * n + i
                        block = k // p2
                        k_local = k % p2
                        w_233_row = k_local * n + i
                        col_start = block * r_block
                        W_full[w_row, col_start:col_start+r_block] = W2[w_233_row, :]

                # Verify the full decomposition
                full_error = 0.0
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            val = np.dot(U_full[a, :], V_full[b, :] * W_full[c, :])
                            full_error += (val - T[a, b, c])**2

                if full_error < 1e-8:
                    best_rank = r_full
                    best_U, best_V, best_W = U_full, V_full, W_full
                    break

        if best_rank == r_full:
            break

    # Strategy 2: Direct search on full 2x3x9 if block approach didn't find good rank
    if best_rank == rank_sb:
        for r in range(30, rank_sb):
            if time.time() > deadline - 10:
                break
            if r >= best_rank:
                continue

            for trial in range(20):
                if time.time() > deadline - 10:
                    break

                np.random.seed(1000 + r * 1000 + trial)
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                W = np.random.randn(p*n, r)

                success = False
                for iteration in range(150):
                    if time.time() > deadline - 10:
                        break

                    # Update U
                    for a in range(n*m):
                        M_a = np.zeros((m*p*p*n, r))
                        for b in range(m*p):
                            for c in range(p*n):
                                M_a[b*(p*n)+c, :] = V[b, :] * W[c, :]
                        target = T[a, :, :].flatten()
                        try:
                            U[a, :], _, _, _ = np.linalg.lstsq(M_a.T, target, rcond=None)
                        except:
                            pass

                    # Update V
                    for b in range(m*p):
                        M_b = np.zeros((n*m*p*n, r))
                        for a in range(n*m):
                            for c in range(p*n):
                                M_b[a*(p*n)+c, :] = U[a, :] * W[c, :]
                        target = T[:, b, :].reshape(n*m, p*n).flatten()
                        try:
                            V[b, :], _, _, _ = np.linalg.lstsq(M_b.T, target, rcond=None)
                        except:
                            pass

                    # Update W
                    for c in range(p*n):
                        M_c = np.zeros((n*m*m*p, r))
                        for a in range(n*m):
                            for b in range(m*p):
                                M_c[a*(m*p)+b, :] = U[a, :] * V[b, :]
                        target = T[:, :, c].reshape(n*m, m*p).flatten()
                        try:
                            W[c, :], _, _, _ = np.linalg.lstsq(M_c.T, target, rcond=None)
                        except:
                            pass

                    # Check convergence
                    error = 0.0
                    for a in range(n*m):
                        for b in range(m*p):
                            for c in range(p*n):
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                error += (val - T[a, b, c])**2

                    if error < 1e-10:
                        success = True
                        break
                    elif error > 1e4:
                        break

                if success:
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    break

            if best_rank == r:
                break

    # Final safety: if time is running out, return best found
    if time.time() > deadline - 5:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
