import numpy as np
import time

def construct():
    """
    Rank-minimal decomposition for 4x4 matrix multiplication using recursive Strassen.
    Attempts to reduce rank by checking for linear dependencies among the 49 terms.
    """
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    block_terms = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]

    sub_terms = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]

    rank_idx = 0
    for A_comb, B_comb, C_comb in block_terms:
        for k in range(7):
            if time.time() - start_time > DEADLINE:
                break
            
            A_loc, B_loc, C_loc = sub_terms[k]
            
            U_elems = {}
            for (bi, bj), coeff_block in A_comb:
                for (lr, lc), coeff_loc in A_loc:
                    gi = 2 * bi + lr
                    gj = 2 * bj + lc
                    idx = u_idx(gi, gj)
                    U_elems[idx] = U_elems.get(idx, 0) + coeff_block * coeff_loc
            
            V_elems = {}
            for (bi, bj), coeff_block in B_comb:
                for (lr, lc), coeff_loc in B_loc:
                    gj = 2 * bi + lr
                    gk = 2 * bj + lc
                    idx = v_idx(gj, gk)
                    V_elems[idx] = V_elems.get(idx, 0) + coeff_block * coeff_loc
            
            W_elems = {}
            for (ci, cj), coeff_block in C_comb:
                for (lr, lc), coeff_loc in C_loc:
                    gi = 2 * ci + lr
                    gj = 2 * cj + lc
                    idx = w_idx(gj, gi)
                    W_elems[idx] = W_elems.get(idx, 0) + coeff_block * coeff_loc
            
            for idx, val in U_elems.items():
                U[idx][rank_idx] += val
            for idx, val in V_elems.items():
                V[idx][rank_idx] += val
            for idx, val in W_elems.items():
                W[idx][rank_idx] += val
                
            rank_idx += 1

    def verify_fast(U_mat, V_mat, W_mat, tol=1e-10):
        U_t = np.array(U_mat, dtype=float)
        V_t = np.array(V_mat, dtype=float)
        W_t = np.array(W_mat, dtype=float)
        r = U_t.shape[1]
        if r == 0:
            return False
        U_r = U_t.reshape(n, m, r)
        V_r = V_t.reshape(m, p, r)
        W_r = W_t.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=tol)

    if not verify_fast(U, V, W):
        return {"U": U, "V": V, "W": W}

    if time.time() - start_time < DEADLINE - 20:
        current_U = [row[:] for row in U]
        current_V = [row[:] for row in V]
        current_W = [row[:] for row in W]
        current_rank = 49
        removed = set()
        
        for col in range(49):
            if time.time() - start_time > DEADLINE - 10:
                break
            if col in removed:
                continue
            
            test_U = [row[:] for row in current_U]
            test_V = [row[:] for row in current_V]
            test_W = [row[:] for row in current_W]
            
            for row in test_U:
                row.pop(col)
            for row in test_V:
                row.pop(col)
            for row in test_W:
                row.pop(col)
            
            if verify_fast(test_U, test_V, test_W):
                current_U = test_U
                current_V = test_V
                current_W = test_W
                removed.add(col)
                current_rank -= 1
        
        if current_rank < 49:
            return {"U": current_U, "V": current_V, "W": current_W}

    return {"U": U, "V": V, "W": W}
