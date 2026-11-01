import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 120.0

    n, m, p = 2, 3, 8
    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 44

    U = np.zeros((n * m, total_rank))  # (6, 44)
    V = np.zeros((m * p, total_rank))  # (24, 44)
    W = np.zeros((p * n, total_rank))  # (16, 44)

    for blk in range(num_blocks):
        base = blk * rank_per_block
        # B columns for this block: 2*blk, 2*blk+1
        # W output indices: for C[i, k] where k = 2*blk or 2*blk+1
        # W index = k * n + i = (2*blk + col_offset) * 2 + i = 4*blk + 2*col_offset + i

        # Strassen for 2x2 core: A[:,0:2] * B[0:2, :]
        # A indices: A[i,j] -> U[i*m + j] = U[i*3 + j]
        # B indices: B[r,c] -> V[r*p + c] = V[r*8 + c]
        # W indices: C[i,k] -> W[k*n + i] = W[k*2 + i]

        b0 = 2 * blk      # first B column in this block
        b1 = 2 * blk + 1  # second B column in this block

        # m1 = (A[0,0]+A[1,1]) * (B[0,b0]+B[1,b1])
        U[0, base + 0] = 1   # A[0,0]
        U[4, base + 0] = 1   # A[1,1]
        V[0 * p + b0, base + 0] = 1    # B[0, b0]
        V[1 * p + b1, base + 0] = 1    # B[1, b1]
        W[0 * n + 0, base + 0] = 1     # C[0, b0]
        W[1 * n + 1, base + 0] = 1     # C[1, b1]

        # m2 = (A[1,0]+A[1,1]) * B[0,b0]
        U[3, base + 1] = 1   # A[1,0]
        U[4, base + 1] = 1   # A[1,1]
        V[0 * p + b0, base + 1] = 1    # B[0, b0]
        W[0 * n + 1, base + 1] = 1     # C[1, b0]
        W[1 * n + 1, base + 1] = -1    # C[1, b1]

        # m3 = A[0,0] * (B[0,b1] - B[1,b1])
        U[0, base + 2] = 1   # A[0,0]
        V[0 * p + b1, base + 2] = 1    # B[0, b1]
        V[1 * p + b1, base + 2] = -1   # B[1, b1]
        W[0 * n + 1, base + 2] = 1     # C[0, b1]
        W[1 * n + 1, base + 2] = 1     # C[1, b1]

        # m4 = A[1,1] * (B[1,b0] - B[0,b0])
        U[4, base + 3] = 1   # A[1,1]
        V[1 * p + b0, base + 3] = 1    # B[1, b0]
        V[0 * p + b0, base + 3] = -1   # B[0, b0]
        W[0 * n + 0, base + 3] = 1     # C[0, b0]
        W[0 * n + 1, base + 3] = 1     # C[1, b0]

        # m5 = (A[0,0]+A[0,1]) * B[1,b1]
        U[0, base + 4] = 1   # A[0,0]
        U[1, base + 4] = 1   # A[0,1]
        V[1 * p + b1, base + 4] = 1    # B[1, b1]
        W[0 * n + 0, base + 4] = -1    # C[0, b0]
        W[0 * n + 1, base + 4] = 1     # C[0, b1]

        # m6 = (A[1,0]-A[0,0]) * (B[0,b0]+B[0,b1])
        U[3, base + 5] = 1   # A[1,0]
        U[0, base + 5] = -1  # A[0,0]
        V[0 * p + b0, base + 5] = 1    # B[0, b0]
        V[0 * p + b1, base + 5] = 1    # B[0, b1]
        W[1 * n + 1, base + 5] = 1     # C[1, b1]

        # m7 = (A[0,1]-A[1,1]) * (B[1,b0]+B[1,b1])
        U[1, base + 6] = 1   # A[0,1]
        U[4, base + 6] = -1  # A[1,1]
        V[1 * p + b0, base + 6] = 1    # B[1, b0]
        V[1 * p + b1, base + 6] = 1    # B[1, b1]
        W[0 * n + 0, base + 6] = 1     # C[0, b0]

        # Extra: A[i,2] * B[2,k] for i in {0,1}, k in {b0, b1}
        # C[0, b0] += A[0,2] * B[2, b0]
        U[2, base + 7] = 1
        V[2 * p + b0, base + 7] = 1
        W[0 * n + 0, base + 7] = 1

        # C[0, b1] += A[0,2] * B[2, b1]
        U[2, base + 8] = 1
        V[2 * p + b1, base + 8] = 1
        W[0 * n + 1, base + 8] = 1

        # C[1, b0] += A[1,2] * B[2, b0]
        U[5, base + 9] = 1
        V[2 * p + b0, base + 9] = 1
        W[0 * n + 1, base + 9] = 1

        # C[1, b1] += A[1,2] * B[2, b1]
        U[5, base + 10] = 1
        V[2 * p + b1, base + 10] = 1
        W[1 * n + 1, base + 10] = 1

    # Quick verification with random test
    if time.time() < _deadline - 10.0:
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(total_rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            prod = np.dot(u_vec, A_test.flatten()) * np.dot(v_vec, B_test.flatten())
            # w_vec has entries at positions k*n+i
            for idx in range(p * n):
                if w_vec[idx] != 0:
                    k = idx // n
                    i = idx % n
                    C_computed[i, k] += w_vec[idx] * prod
        err = np.max(np.abs(C_computed - C_expected))
        if err > 1e-10:
            # Fallback: return schoolbook rank-48
            U_s = np.zeros((n * m, n * m * p))
            V_s = np.zeros((m * p, n * m * p))
            W_s = np.zeros((p * n, n * m * p))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j, idx] = 1
                        V_s[j * p + k, idx] = 1
                        W_s[k * n + i, idx] = 1
                        idx += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    if time.time() > _deadline - 5.0:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
