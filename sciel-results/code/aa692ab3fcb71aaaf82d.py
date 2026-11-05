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

    def build_4x4x4_strassen():
        """Rank-49 Strassen decomposition for 4x4 * 4x4 -> 4x4."""
        U_sub, V_sub, W_sub = build_2x2x2_strassen()
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(16)]
        W = [[0]*total_rank for _ in range(16)]
        
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
        v_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        w_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2; j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * 4 + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(4):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 2; k_local = v_local % 2
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * 4 + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(4):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2; i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * 4 + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff
        return U, V, W

    def build_4x4x2():
        """Rank-16 decomposition for 4x4 * 4x2 -> 4x2."""
        # This is a schoolbook-like decomposition but we can potentially optimize.
        # For now, let's use a direct construction that is valid.
        # 4x4 * 4x2.
        # We can view this as two 4x4 * 4x1 problems.
        # Or just direct.
        # Rank 16 is the schoolbook rank.
        rank = 16
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return U, V, W

    def build_4x4x6():
        """Combine 4x4x4 (rank 49) and 4x4x2 (rank 16) for total rank 65."""
        U1, V1, W1 = build_4x4x4_strassen()
        U2, V2, W2 = build_4x4x2()
        
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        total_rank = rank1 + rank2
        
        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(24)]
        W = [[0]*total_rank for _ in range(24)]
        
        # Embed 4x4x4 part
        for i in range(16):
            for c in range(rank1):
                U[i][c] = U1[i][c]
        for i in range(16): # V1 is 16x16, V is 24x65. Rows 0-15
            for c in range(rank1):
                V[i][c] = V1[i][c]
        # V1 rows 16-23 are 0 (already initialized)
        
        for i in range(16): # W1 is 16x16, W is 24x65. Rows 0-15
            for c in range(rank1):
                W[i][c] = W1[i][c]
        # W1 rows 16-23 are 0
        
        # Embed 4x4x2 part
        base = rank1
        for i in range(16):
            for c in range(rank2):
                U[i][base+c] = U2[i][c]
        for i in range(8): # V2 is 8x16, V is 24x65. Rows 16-23
            for c in range(rank2):
                V[16+i][base+c] = V2[i][c]
        for i in range(8): # W2 is 8x16, W is 24x65. Rows 16-23
            for c in range(rank2):
                W[16+i][base+c] = W2[i][c]
                
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is valid using vectorized numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Vectorized check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(np.sum(prod) - 1.0) > tol:
                        return False
        return True

    def try_reduce_rank(U, V, W, n, m, p, time_budget):
        """Try to reduce rank by systematically removing redundant columns."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        current_rank = U_arr.shape[1]
        
        if time.time() > time_budget:
            return U, V, W, current_rank
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = current_rank
        
        def is_valid(U_t, V_t, W_t):
            """Check if decomposition is valid using vectorized numpy."""
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = U_t[i*m+j, :] * V_t[j*p+k, :] * W_t[k*n+i, :]
                        if abs(np.sum(prod) - 1.0) > 1e-8:
                            return False
            return True
        
        # Strategy: Greedy removal
        remaining = list(range(current_rank))
        
        # First pass: try removing each column individually
        for _ in range(min(50, current_rank)):  # Limit iterations for time
            if time.time() > time_budget:
                break
            
            best_remove = -1
            best_new_rank = best_rank
            
            for col in remaining:
                if time.time() > time_budget:
                    break
                
                # Build test matrices without this column
                mask = [i for i in range(best_rank) if i != col]
                U_test = best_U[:, mask]
                V_test = best_V[:, mask]
                W_test = best_W[:, mask]
                
                if is_valid(U_test, V_test, W_test):
                    best_remove = col
                    best_new_rank = len(mask)
                    break  # Take the first valid removal
            
            if best_remove >= 0:
                mask = [i for i in range(best_rank) if i != best_remove]
                best_U = best_U[:, mask]
                best_V = best_V[:, mask]
                best_W = best_W[:, mask]
                best_rank = len(mask)
                remaining = [i for i in remaining if i != best_remove]
            else:
                break
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

    # Build the initial rank-65 decomposition
    U, V, W = build_4x4x6()
    
    # Verify the initial decomposition
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
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
    
    # Try to reduce the rank
    time_budget = deadline - 15
    U_reduced, V_reduced, W_reduced, reduced_rank = try_reduce_rank(U, V, W, n, m, p, time_budget)
    
    # Use the reduced decomposition if it's valid
    if verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
        return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    else:
        # Fall back to the original decomposition
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
