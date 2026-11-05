"""Combined approach: structural embedding + rank reduction via linear dependence search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 130  # safety margin before 150s kill

    # Start with the known rank-55 decomposition (Strassen + schoolbook)
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Strassen coefficients for 2x2x2
    L = [
        {0: 1, 3: 1},       # M1: a00, a11
        {2: 1, 3: 1},       # M2: a10, a11
        {0: 1},             # M3: a00
        {3: 1},             # M4: a11
        {0: 1, 1: 1},       # M5: a00, a01
        {2: 1, 0: -1},      # M6: a10, -a00
        {1: 1, 3: -1}       # M7: a01, -a11
    ]
    R = [
        {0: 1, 3: 1},       # M1: b00, b11
        {0: 1},             # M2: b00
        {1: 1, 3: -1},      # M3: b01, -b11
        {2: 1, 0: -1},      # M4: b10, -b00
        {3: 1},             # M5: b11
        {0: 1, 1: 1},       # M6: b00, b01
        {2: 1, 3: 1}        # M7: b10, b11
    ]
    Wc = [
        {0: 1, 3: 1},       # M1: c00, c11
        {2: 1, 3: -1},      # M2: c10, -c11
        {1: 1, 3: 1},       # M3: c01, c11
        {0: 1, 2: 1},       # M4: c00, c10
        {1: 1, 0: -1},      # M5: c01, -c00
        {3: 1},             # M6: c11
        {0: 1}              # M7: c00
    ]

    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1

    # Schoolbook for the 2x1 * 1x10 part
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Pre-compute the target tensor (all ones)
    target_tensor = np.ones((n, m, p), dtype=float)

    def check_validity(U_arr, V_arr, W_arr, tol=1e-9):
        """Check if the decomposition is valid using a faster method."""
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this efficiently by iterating over r
        T = np.zeros((n, m, p), dtype=float)
        for r in range(U_arr.shape[1]):
            # u_r is a vector of size n*m
            # v_r is a vector of size m*p
            # w_r is a vector of size p*n
            u_r = U_arr[:, r]
            v_r = V_arr[:, r]
            w_r = W_arr[:, r]
            
            # Reshape to matrices
            u_mat = u_r.reshape(n, m)
            v_mat = v_r.reshape(m, p)
            w_mat = w_r.reshape(p, n)
            
            # The contribution of rank r is u_mat * v_mat * w_mat? No.
            # T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This is not a standard matrix product.
            # We must iterate or use einsum.
            # np.einsum is faster than python loops.
            T += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
        
        return np.allclose(T, target_tensor, atol=tol)

    # Verify initial decomposition
    if not check_validity(U_np, V_np, W_np):
        return {"U": U, "V": V, "W": W}

    current_rank = rank
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()

    # Greedy removal
    removed = set()
    changed = True
    while changed and time.time() - start_time < deadline - 10:
        changed = False
        for r in range(current_rank):
            if r in removed:
                continue
            # Remove column r
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            if check_validity(U_test, V_test, W_test):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                removed.add(r)
                current_rank -= 1
                changed = True
                break

    # Pair removal
    if time.time() - start_time < deadline - 20:
        rank_now = current_rank
        for r1 in range(rank_now):
            if time.time() - start_time > deadline - 15:
                break
            for r2 in range(r1 + 1, rank_now):
                mask = np.ones(rank_now, dtype=bool)
                mask[r1] = False
                mask[r2] = False
                U_test = current_U[:, mask]
                V_test = current_V[:, mask]
                W_test = current_W[:, mask]
                
                if check_validity(U_test, V_test, W_test):
                    current_U = U_test
                    current_V = V_test
                    current_W = W_test
                    current_rank -= 2
                    break
            if time.time() - start_time > deadline - 15:
                break

    U_final = current_U.tolist()
    V_final = current_V.tolist()
    W_final = current_W.tolist()

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
