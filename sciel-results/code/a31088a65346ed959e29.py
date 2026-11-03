import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Build the rank-36 Strassen-based decomposition
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        strassen_patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    col = 0
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    assert col == total_rank

    def verify_decomposition(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    continue
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    if k_b != k_c or i_a != i_c:
                        s = np.dot(U[a], V[b] * W[c])
                        if abs(s) > tol:
                            return False
        return True

    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        best_result = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = total_rank

    # Try to reduce rank by finding linear dependencies
    if time.time() < deadline - 10:
        # Try a completely different structural approach:
        # Instead of splitting j into {0,1} and {2,3}, try other pairings
        # and see if we can get a lower rank
        
        for pairing in [([0,1],[2,3]), ([0,2],[1,3]), ([0,3],[1,2])]:
            if time.time() > deadline - 15:
                break
            j1, j2 = pairing
            U2 = np.zeros((nm, 36), dtype=np.float64)
            V2 = np.zeros((mp, 36), dtype=np.float64)
            W2 = np.zeros((pn, 36), dtype=np.float64)
            
            col2 = 0
            # For pair j1: i=0,1; j=j1[0],j1[1]; k=0..4
            for k_start in [0, 2, 4]:
                k_end = min(k_start + 2, 5)
                if k_end - k_start == 2:
                    strassen_patterns = [
                        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
                        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
                        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
                        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
                        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
                        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
                    ]
                    for u_pat, v_pat, w_pat in strassen_patterns:
                        for idx, val in enumerate(u_pat):
                            if val != 0:
                                i_local = idx // 2
                                j_local = idx % 2
                                i_global = i_local
                                j_global = j1[j_local]
                                U2[i_global * m + j_global, col2] = val
                        for idx, val in enumerate(v_pat):
                            if val != 0:
                                j_local = idx // 2
                                k_local = idx % 2
                                j_global = j1[j_local]
                                k_global = k_start + k_local
                                V2[j_global * p + k_global, col2] = val
                        for idx, val in enumerate(w_pat):
                            if val != 0:
                                i_local = idx // 2
                                k_local = idx % 2
                                i_global = i_local
                                k_global = k_start + k_local
                                W2[k_global * n + i_global, col2] = val
                        col2 += 1
                else:
                    for idx in range(4):
                        i_local = idx // 2
                        j_local = idx % 2
                        i_global = i_local
                        j_global = j1[j_local]
                        k_global = k_start
                        U2[i_global * m + j_global, col2] = 1
                        V2[j_global * p + k_global, col2] = 1
                        W2[k_global * n + i_global, col2] = 1
                        col2 += 1
            
            # For pair j2
            for k_start in [0, 2, 4]:
                k_end = min(k_start + 2, 5)
                if k_end - k_start == 2:
                    strassen_patterns = [
                        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
                        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
                        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
                        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
                        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
                        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
                    ]
                    for u_pat, v_pat, w_pat in strassen_patterns:
                        for idx, val in enumerate(u_pat):
                            if val != 0:
                                i_local = idx // 2
                                j_local = idx % 2
                                i_global = i_local
                                j_global = j2[j_local]
                                U2[i_global * m + j_global, col2] = val
                        for idx, val in enumerate(v_pat):
                            if val != 0:
                                j_local = idx // 2
                                k_local = idx % 2
                                j_global = j2[j_local]
                                k_global = k_start + k_local
                                V2[j_global * p + k_global, col2] = val
                        for idx, val in enumerate(w_pat):
                            if val != 0:
                                i_local = idx // 2
                                k_local = idx % 2
                                i_global = i_local
                                k_global = k_start + k_local
                                W2[k_global * n + i_global, col2] = val
                        col2 += 1
                else:
                    for idx in range(4):
                        i_local = idx // 2
                        j_local = idx % 2
                        i_global = i_local
                        j_global = j2[j_local]
                        k_global = k_start
                        U2[i_global * m + j_global, col2] = 1
                        V2[j_global * p + k_global, col2] = 1
                        W2[k_global * n + i_global, col2] = 1
                        col2 += 1
            
            if col2 < best_rank and verify_decomposition(U2[:, :col2], V2[:, :col2], W2[:, :col2]):
                U2 = U2[:, :col2]
                V2 = V2[:, :col2]
                W2 = W2[:, :col2]
                U_list = [[int(round(U2[i, r] * 2)) / 2 for r in range(col2)] for i in range(nm)]
                V_list = [[int(round(V2[i, r] * 2)) / 2 for r in range(col2)] for i in range(mp)]
                W_list = [[int(round(W2[i, r] * 2)) / 2 for r in range(col2)] for i in range(pn)]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = col2

    # Final: try to verify and return best
    if best_rank == 36:
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
# EVOLVE-BLOCK-END
