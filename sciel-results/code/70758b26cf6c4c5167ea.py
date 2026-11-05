import time
import numpy as np
import itertools

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    dim = n
    block_size = 3
    num_blocks = 2

    total_rank = 7 * 27

    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)

    # Strassen steps defined as (A_terms, B_terms, C_terms)
    # Each term is (I, J, coeff)
    steps = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    bs = block_size
    col_idx = 0

    for step in steps:
        a_terms, b_terms, c_terms = step
        for r, s, t in itertools.product(range(bs), repeat=3):
            # Process U
            for I, J, ca in a_terms:
                u_idx = (I * bs + r) * dim + (J * bs + s)
                U[u_idx, col_idx] += ca

            # Process V
            for J, K, cb in b_terms:
                v_idx = (J * bs + s) * dim + (K * bs + t)
                V[v_idx, col_idx] += cb

            # Process W
            for I, K, cc in c_terms:
                w_idx = (K * bs + t) * dim + (I * bs + r)
                W[w_idx, col_idx] += cc

            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1 for all i,j,k
    try:
        if time.time() < deadline - 5:
            rank = U.shape[1]
            result = np.zeros((n, m, p), dtype=np.float64)
            for r in range(rank):
                u_col = U[:, r].reshape(n, m)
                v_col = V[:, r].reshape(m, p)
                w_col = W[:, r].reshape(p, n)  # w_col[k, i] = W[k*n+i, r]
                for i in range(n):
                    temp = v_col * w_col[:, i:i+1]  # (m,p)
                    result[i, :, :] += u_col[i, :] * temp
            if not np.allclose(result, 1.0, atol=1e-10):
                # Fall back to schoolbook
                total_rank_school = n * m * p
                U_s = np.zeros((dim * dim, total_rank_school), dtype=np.float64)
                V_s = np.zeros((dim * dim, total_rank_school), dtype=np.float64)
                W_s = np.zeros((dim * dim, total_rank_school), dtype=np.float64)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            U_s[i * m + j, idx] = 1
                            V_s[j * p + k, idx] = 1
                            W_s[k * n + i, idx] = 1
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
