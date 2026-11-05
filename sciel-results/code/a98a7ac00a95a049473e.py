"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

    def find_2x2x3_rank10():
        """Attempt to find a rank-10 decomposition for 2x2 * 2x3 -> 2x3."""
        import random
        import numpy as np
        
        # Target: U(4x10), V(6x10), W(6x10)
        # Constraint: sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] == 1
        # for all i in 0..1, j in 0..1, k in 0..2
        
        candidates = [-2, -1, 0, 1, 2]
        
        for _ in range(5000): # Limit search time
            U = np.random.choice(candidates, size=(4, 10))
            V = np.random.choice(candidates, size=(6, 10))
            W = np.random.choice(candidates, size=(6, 10))
            
            # Check validity
            valid = True
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        u_idx = i * 2 + j
                        v_idx = j * 3 + k
                        w_idx = k * 2 + i
                        prod = U[u_idx] * V[v_idx] * W[w_idx]
                        if sum(prod) != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
        
        return None

    def build_2x2x3():
        """Rank-11 (or 10 if found) decomposition for 2x2 * 2x3 -> 2x3."""
        rank10_result = find_2x2x3_rank10()
        if rank10_result:
            return rank10_result
            
        # Fallback to Rank-11
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0: U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0: V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0: W[r][c] += W1[r][c]
        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0:1, 1:2, 2:4, 3:5}
        w2_map = {0:2, 1:4, 2:3, 3:5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0: U[r][base+c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0: V[v2_map[r]][base+c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0: W[w2_map[r]][base+c] += W2[r][c]
        return U, V, W

    def build_4x4x6():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        terms = [
            {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]},
            {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1),(3,-1)]},
            {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1),(3,1)]},
            {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},
            {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
            {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
            {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1)]},
        ]
        u_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        v_offset = {0:(0,0), 1:(0,3), 2:(2,0), 3:(2,3)}
        w_offset = {0:(0,0), 1:(3,0), 2:(0,2), 3:(3,2)}
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2; j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 3; k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2; i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff
        return U, V, W

    def verify(U, V, W, n, m, p, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(sum(prod) - 1.0) > tol:
                        return False
        return True

    def try_reduce_rank(U, V, W, n, m, p, target_rank, time_budget):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 terms.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        current_rank = U_arr.shape[1]
        
        if current_rank <= target_rank:
            return U, V, W
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = current_rank
        
        cols_to_try = list(range(current_rank))
        np.random.shuffle(cols_to_try)
        
        for remove_idx in cols_to_try:
            if time.time() > time_budget:
                break
            if best_rank <= target_rank:
                break
            
            mask = [i for i in range(best_rank) if i != remove_idx]
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            
            valid = True
            max_err = 0.0
            for i in range(n):
                if not valid: break
                for j in range(m):
                    if not valid: break
                    for k in range(p):
                        prod = U_test[i*m+j, :] * V_test[j*p+k, :] * W_test[k*n+i, :]
                        err = abs(sum(prod) - 1.0)
                        max_err = max(max_err, err)
                        if err > 1e-6:
                            valid = False
                            break
            
            if valid:
                best_U = U_test
                best_V = V_test
                best_W = W_test
                best_rank -= 1
        
        if best_rank < current_rank:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        return U, V, W

    def try_pairwise_merge(U, V, W, n, m, p, target_rank, time_budget):
        """
        Try to merge pairs of columns by expressing their sum as a single rank-1 term.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        current_rank = U_arr.shape[1]
        
        if current_rank <= target_rank:
            return U, V, W
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = current_rank
        
        cols = list(range(best_rank))
        np.random.shuffle(cols)
        
        for i in range(len(cols)):
            if time.time() > time_budget:
                break
            if best_rank <= target_rank:
                break
            for j in range(i+1, len(cols)):
                if time.time() > time_budget:
                    break
                if best_rank <= target_rank:
                    break
                
                r, s = cols[i], cols[j]
                
                mask = [x for x in range(best_rank) if x != r and x != s]
                U_test = best_U[:, mask]
                V_test = best_V[:, mask]
                W_test = best_W[:, mask]
                
                valid = True
                for ii in range(n):
                    if not valid: break
                    for jj in range(m):
                        if not valid: break
                        for kk in range(p):
                            prod = U_test[ii*m+jj, :] * V_test[jj*p+kk, :] * W_test[kk*n+ii, :]
                            if abs(sum(prod) - 1.0) > 1e-6:
                                valid = False
                                break
                
                if valid:
                    best_U = U_test
                    best_V = V_test
                    best_W = W_test
                    best_rank -= 2
        
        if best_rank < current_rank:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        return U, V, W

    U, V, W = build_4x4x6()

    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    # Final verification
    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
