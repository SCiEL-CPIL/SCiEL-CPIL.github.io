"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Build base rank-36 solution using Strassen blocks
    R_base = 36
    U = np.zeros((n * m, R_base), dtype=float)
    V = np.zeros((m * p, R_base), dtype=float)
    W = np.zeros((p * n, R_base), dtype=float)

    col = 0

    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        # M1
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M2
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1
        # M3
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M4
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        # M5
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1
        # M6
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        # M7
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] += -1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    C11 = [(0,0),(1,0),(0,1),(1,1)]
    C12 = [(2,0),(3,0),(2,1),(3,1)]

    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    # 8 direct terms for the last column (k=4).
    for (ai, aj) in [(0,0),(0,1),(1,0),(1,1)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1
    for (ai, aj) in [(0,2),(0,3),(1,2),(1,3)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1

    assert col == 36, col

    def verify(Ua, Va, Wa, R):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    base_U, base_V, base_W = U.copy(), V.copy(), W.copy()
    best_rank = 36
    best_U, best_V, best_W = base_U.copy(), base_V.copy(), base_W.copy()

    if verify(base_U, base_V, base_W, 36):
        # Try to find a column that is a linear combination of others
        # For each candidate column c2, solve for coefficients in other columns
        for c2 in range(36):
            if time.time() > deadline - 10:
                break
            
            # Build system: U[:,c2] = U[:,others] @ alpha
            # Same for V and W
            others = [r for r in range(36) if r != c2]
            
            # For U: 8 equations, 35 unknowns
            # For V: 20 equations, 35 unknowns  
            # For W: 10 equations, 35 unknowns
            # Combined: 38 equations, 35 unknowns - overdetermined
            
            U_target = U[:, c2]
            V_target = V[:, c2]
            W_target = W[:, c2]
            
            U_mat = U[:, others]
            V_mat = V[:, others]
            W_mat = W[:, others]
            
            # Stack into one system
            A_sys = np.vstack([U_mat, V_mat, W_mat])  # 38 x 35
            b_sys = np.concatenate([U_target, V_target, W_target])  # 38
            
            # Solve least squares
            try:
                alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=1e-12)
            except:
                continue
            
            # Check if solution is exact
            A_sys_recon = A_sys @ alpha
            if np.max(np.abs(A_sys_recon - b_sys)) < 1e-9:
                # Found a valid merge!
                R_new = 35
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                
                for idx, r in enumerate(others):
                    U_new[:, r] += alpha[idx] * U[:, c2]
                    V_new[:, r] += alpha[idx] * V[:, c2]
                    W_new[:, r] += alpha[idx] * W[:, c2]
                
                # Remove column c2
                U_new = np.delete(U_new, c2, axis=1)
                V_new = np.delete(V_new, c2, axis=1)
                W_new = np.delete(W_new, c2, axis=1)
                
                if verify(U_new, V_new, W_new, R_new):
                    best_rank = R_new
                    best_U, best_V, best_W = U_new, V_new, W_new
                    break
    
    # If we found rank 35, try to find rank 34
    if best_rank == 35 and time.time() < deadline - 15:
        for c2 in range(35):
            if time.time() > deadline - 5:
                break
            
            others = [r for r in range(35) if r != c2]
            
            U_target = best_U[:, c2]
            V_target = best_V[:, c2]
            W_target = best_W[:, c2]
            
            U_mat = best_U[:, others]
            V_mat = best_V[:, others]
            W_mat = best_W[:, others]
            
            A_sys = np.vstack([U_mat, V_mat, W_mat])  # 38 x 34
            b_sys = np.concatenate([U_target, V_target, W_target])  # 38
            
            try:
                alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=1e-12)
            except:
                continue
            
            A_sys_recon = A_sys @ alpha
            if np.max(np.abs(A_sys_recon - b_sys)) < 1e-9:
                R_new = 34
                U_new = best_U.copy()
                V_new = best_V.copy()
                W_new = best_W.copy()
                
                for idx, r in enumerate(others):
                    U_new[:, r] += alpha[idx] * best_U[:, c2]
                    V_new[:, r] += alpha[idx] * best_V[:, c2]
                    W_new[:, r] += alpha[idx] * best_W[:, c2]
                
                U_new = np.delete(U_new, c2, axis=1)
                V_new = np.delete(V_new, c2, axis=1)
                W_new = np.delete(W_new, c2, axis=1)
                
                if verify(U_new, V_new, W_new, R_new):
                    best_rank = R_new
                    best_U, best_V, best_W = U_new, V_new, W_new
                    break
    
    # Try more aggressive search: allow linear combinations of multiple columns to be removed
    if best_rank == 35 and time.time() < deadline - 20:
        # Try removing two columns at once
        for c1 in range(35):
            if time.time() > deadline - 10:
                break
            for c2 in range(c1+1, 35):
                if time.time() > deadline - 10:
                    break
                
                others = [r for r in range(35) if r != c1 and r != c2]
                
                U_target = np.column_stack([best_U[:, c1], best_U[:, c2]])
                V_target = np.column_stack([best_V[:, c1], best_V[:, c2]])
                W_target = np.column_stack([best_W[:, c1], best_W[:, c2]])
                
                U_mat = best_U[:, others]
                V_mat = best_V[:, others]
                W_mat = best_W[:, others]
                
                A_sys = np.vstack([U_mat, V_mat, W_mat])  # 38 x 33
                b_sys = np.concatenate([U_target.flatten(), V_target.flatten(), W_target.flatten()])  # 76
                
                try:
                    alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=1e-12)
                except:
                    continue
                
                A_sys_recon = A_sys @ alpha
                if np.max(np.abs(A_sys_recon - b_sys)) < 1e-9:
                    R_new = 33
                    U_new = best_U.copy()
                    V_new = best_V.copy()
                    W_new = best_W.copy()
                    
                    # Reshape alpha to (33, 2)
                    alpha_reshaped = alpha.reshape(33, 2)
                    
                    for idx, r in enumerate(others):
                        U_new[:, r] += alpha_reshaped[idx, 0] * best_U[:, c1] + alpha_reshaped[idx, 1] * best_U[:, c2]
                        V_new[:, r] += alpha_reshaped[idx, 0] * best_V[:, c1] + alpha_reshaped[idx, 1] * best_V[:, c2]
                        W_new[:, r] += alpha_reshaped[idx, 0] * best_W[:, c1] + alpha_reshaped[idx, 1] * best_W[:, c2]
                    
                    # Remove columns c1 and c2 (c2 first since it's higher index)
                    U_new = np.delete(U_new, [c1, c2], axis=1)
                    V_new = np.delete(V_new, [c1, c2], axis=1)
                    W_new = np.delete(W_new, [c1, c2], axis=1)
                    
                    if verify(U_new, V_new, W_new, R_new):
                        best_rank = R_new
                        best_U, best_V, best_W = U_new, V_new, W_new
                        break
            if best_rank < 35:
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
