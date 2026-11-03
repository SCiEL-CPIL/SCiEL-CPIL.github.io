"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    # Split dimensions:
    # n=3 -> [1, 2]
    # m=5 -> [2, 2, 1]
    # p=5 -> [2, 2, 1]
    n_splits = [1, 2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]

    # Compute total rank
    def block_rank(ns, ms, ps):
        if ns == 2 and ms == 2 and ps == 2:
            return 7  # Strassen
        else:
            return ns * ms * ps

    total_rank = 0
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                total_rank += block_rank(ns, ms, ps)

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    i_offset = 0
    for ns in n_splits:
        j_offset = 0
        for ms in m_splits:
            k_offset = 0
            for ps in p_splits:
                if ns == 2 and ms == 2 and ps == 2:
                    # Strassen 2x2x2
                    for r in range(7):
                        # Define U, V, W for each Strassen multiply
                        if r == 0:  # M1 = (A00+A11)(B00+B11)
                            # U
                            U[(i_offset + 0) * m + (j_offset + 0), col] = 1
                            U[(i_offset + 1) * m + (j_offset + 1), col] = 1
                            # V
                            V[(j_offset + 0) * p + (k_offset + 0), col] = 1
                            V[(j_offset + 1) * p + (k_offset + 1), col] = 1
                            # W
                            W[(k_offset + 0) * n + (i_offset + 0), col] = 1
                            W[(k_offset + 1) * n + (i_offset + 1), col] = 1
                        elif r == 1:  # M2 = (A10+A11)B00
                            U[(i_offset + 1) * m + (j_offset + 0), col] = 1
                            U[(i_offset + 1) * m + (j_offset + 1), col] = 1
                            V[(j_offset + 0) * p + (k_offset + 0), col] = 1
                            W[(k_offset + 0) * n + (i_offset + 1), col] = 1
                        elif r == 2:  # M3 = A00(B01-B11)
                            U[(i_offset + 0) * m + (j_offset + 0), col] = 1
                            V[(j_offset + 0) * p + (k_offset + 1), col] = 1
                            V[(j_offset + 1) * p + (k_offset + 1), col] = -1
                            W[(k_offset + 1) * n + (i_offset + 0), col] = 1
                        elif r == 3:  # M4 = A11(B10-B00)
                            U[(i_offset + 1) * m + (j_offset + 1), col] = 1
                            V[(j_offset + 1) * p + (k_offset + 0), col] = 1
                            V[(j_offset + 0) * p + (k_offset + 0), col] = -1
                            W[(k_offset + 0) * n + (i_offset + 1), col] = 1
                            W[(k_offset + 1) * n + (i_offset + 1), col] = 1
                        elif r == 4:  # M5 = (A00+A01)B11
                            U[(i_offset + 0) * m + (j_offset + 0), col] = 1
                            U[(i_offset + 0) * m + (j_offset + 1), col] = 1
                            V[(j_offset + 1) * p + (k_offset + 1), col] = 1
                            W[(k_offset + 1) * n + (i_offset + 0), col] = 1
                            W[(k_offset + 1) * n + (i_offset + 1), col] = 1
                        elif r == 5:  # M6 = (A10-A00)(B00+B01)
                            U[(i_offset + 1) * m + (j_offset + 0), col] = 1
                            U[(i_offset + 0) * m + (j_offset + 0), col] = -1
                            V[(j_offset + 0) * p + (k_offset + 0), col] = 1
                            V[(j_offset + 0) * p + (k_offset + 1), col] = 1
                            W[(k_offset + 1) * n + (i_offset + 1), col] = 1
                        elif r == 6:  # M7 = (A01-A11)(B10+B11)
                            U[(i_offset + 0) * m + (j_offset + 1), col] = 1
                            U[(i_offset + 1) * m + (j_offset + 1), col] = -1
                            V[(j_offset + 1) * p + (k_offset + 0), col] = 1
                            V[(j_offset + 1) * p + (k_offset + 1), col] = 1
                            W[(k_offset + 0) * n + (i_offset + 0), col] = 1
                            W[(k_offset + 1) * n + (i_offset + 1), col] = 1
                        col += 1
                else:
                    # Naive for this block
                    for di in range(ns):
                        for dj in range(ms):
                            for dk in range(ps):
                                U[(i_offset + di) * m + (j_offset + dj), col] = 1
                                V[(j_offset + dj) * p + (k_offset + dk), col] = 1
                                W[(k_offset + dk) * n + (i_offset + di), col] = 1
                                col += 1
                k_offset += ps
            j_offset += ms
        i_offset += ns

    # Verify correctness
    if time.time() - start_time < DEADLINE:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 if i,j,k valid, 0 otherwise
        # For efficiency, check a few random entries
        np.random.seed(42)
        for _ in range(100):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
            if abs(val - 1) > 1e-10:
                # Fallback to naive
                rank = n * m * p
                U_naive = np.zeros((n * m, rank))
                V_naive = np.zeros((m * p, rank))
                W_naive = np.zeros((p * n, rank))
                col2 = 0
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            U_naive[ii * m + jj, col2] = 1
                            V_naive[jj * p + kk, col2] = 1
                            W_naive[kk * n + ii, col2] = 1
                            col2 += 1
                return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist}
# EVOLVE-BLOCK-END
