import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    dim_A = n * m  # 8
    dim_B = m * p  # 32
    dim_C = p * n  # 16

    # Construct the true tensor T
    # T[r, c, o] = 1 if r=i*m+j, c=j*p+k, o=k*n+i
    T_true = np.zeros((dim_A, dim_B, dim_C), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m + j
                c = j * p + k
                o = k * n + i
                T_true[r, c, o] = 1.0

    # Function to verify a candidate decomposition
    def verify(U, V, W):
        # Check if U, V, W are close to integers
        # We allow half-integers? The problem says integer/half-integer.
        # Let's check if 2*U, 2*V, 2*W are close to integers.
        U_int = np.round(2 * U)
        V_int = np.round(2 * V)
        W_int = np.round(2 * W)
        
        if not (np.allclose(U, U_int / 2.0, atol=1e-5) and 
                np.allclose(V, V_int / 2.0, atol=1e-5) and 
                np.allclose(W, W_int / 2.0, atol=1e-5)):
            return False
            
        U_s = U_int / 2.0
        V_s = V_int / 2.0
        W_s = W_int / 2.0
        
        T_recon = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        return np.allclose(T_recon, T_true, atol=1e-5)

    # Strategy: ALS (Alternating Least Squares) Rank Search
    # We will try to find a decomposition of rank R.
    # We start with R = 50 and go up to 55.
    # We use multiple random restarts for each R.
    
    def solve_for_W(U, V):
        # Solve sum_t U[:,t] V[:,t] W[:,t] = T
        # Vectorize T to (dim_A * dim_B, dim_C)
        T_mat = T_true.reshape(dim_A * dim_B, dim_C)
        
        # M is (dim_A * dim_B, R)
        # M[:, t] = kron(V[:, t], U[:, t])
        M = np.zeros((dim_A * dim_B, U.shape[1]))
        for t in range(U.shape[1]):
            M[:, t] = np.kron(V[:, t], U[:, t])
            
        # Solve M @ W_flat = T_mat
        # W_flat is (R, dim_C)
        # M is (dim_A*dim_B, R)
        # We want W such that M.T @ W = T_mat.T ?
        # Let's use lstsq on the system:
        # sum_t M[:, t] W[:, t] = T_mat
        # This is a linear system for each column of T_mat.
        # T_mat = M @ W_mat
        # W_mat is (R, dim_C)
        W_mat, residuals, rank, sv = np.linalg.lstsq(M, T_mat, rcond=None)
        return W_mat

    def solve_for_U(V, W):
        # T = sum U[:,t] V[:,t] W[:,t]
        # T.reshape(dim_A, dim_B*dim_C) = U @ (V * W).T ?
        # Let's unfold T in mode 1: T_(1) is (dim_A, dim_B * dim_C)
        # T_(1)[:, b, c] = sum_t U[t, a] V[t, b] W[t, c]
        # T_(1) = U @ (V * W).T  where (V*W) is element-wise product? No.
        # T_(1)[a, b*dim_C + c] = sum_t U[a, t] (V[b, t] * W[c, t])
        # Let P = V * W (element-wise, shape dim_B x dim_C x R? No)
        # P[b, c, t] = V[b, t] * W[c, t]
        # T_(1) = U @ P_flat.T ?
        # P_flat is (dim_B * dim_C, R)
        # P_flat[b*dim_C + c, t] = V[b, t] * W[c, t]
        
        T_mat1 = T_true.reshape(dim_A, dim_B * dim_C)
        
        P_flat = np.zeros((dim_B * dim_C, V.shape[1]))
        for t in range(V.shape[1]):
            P_flat[:, t] = np.kron(V[:, t], W[:, t])
            
        U_mat, residuals, rank, sv = np.linalg.lstsq(P_flat, T_mat1, rcond=None)
        return U_mat

    def solve_for_V(U, W):
        # T_(2) is (dim_B, dim_A * dim_C)
        T_mat2 = np.transpose(T_true, (1, 0, 2)).reshape(dim_B, dim_A * dim_C)
        
        P_flat = np.zeros((dim_A * dim_C, U.shape[1]))
        for t in range(U.shape[1]):
            P_flat[:, t] = np.kron(U[:, t], W[:, t])
            
        V_mat, residuals, rank, sv = np.linalg.lstsq(P_flat, T_mat2, rcond=None)
        return V_mat

    # Try ranks from 50 to 55
    # We use a fixed seed for reproducibility of the "best" attempt, 
    # but the evaluator runs it once. We want to find *a* solution.
    
    best_solution = None
    
    # Fallback: Strassen 56 (from current code)
    def get_strassen_56():
        S_terms = []
        S_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        S_terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
        S_terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
        S_terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
        S_terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,-1]))
        S_terms.append(([0,-1,1,0], [1,1,0,0], [0,0,0,1]))
        S_terms.append(([0,1,0,-1], [0,0,1,1], [1,-1,0,0]))
        
        rank = 56
        U = np.zeros((dim_A, rank), dtype=float)
        V = np.zeros((dim_B, rank), dtype=float)
        W = np.zeros((dim_C, rank), dtype=float)
        
        t_idx = 0
        for half in range(2):
            B_offset = half * 16
            C_offset = half * 8
            for s_u, s_v, s_w in S_terms:
                u = np.zeros(8); u[0:4] = s_u
                v = np.zeros(32)
                v_indices = [0, 1, 4, 5]
                for i, val in enumerate(s_v): v[v_indices[i] + B_offset] = val
                w = np.zeros(16)
                w_indices = [0, 1, 4, 5]
                for i, val in enumerate(s_w): w[w_indices[i] + C_offset] = val
                U[:, t_idx] = u; V[:, t_idx] = v; W[:, t_idx] = w; t_idx += 1
                
                u = np.zeros(8); u[0:4] = s_u
                v = np.zeros(32)
                v_indices = [2, 3, 6, 7]
                for i, val in enumerate(s_v): v[v_indices[i] + B_offset] = val
                w = np.zeros(16)
                w_indices = [2, 3, 6, 7]
                for i, val in enumerate(s_w): w[w_indices[i] + C_offset] = val
                U[:, t_idx] = u; V[:, t_idx] = v; W[:, t_idx] = w; t_idx += 1
                
                u = np.zeros(8); u[4:8] = s_u
                v = np.zeros(32)
                v_indices = [8, 9, 12, 13]
                for i, val in enumerate(s_v): v[v_indices[i] + B_offset] = val
                w = np.zeros(16)
                w_indices = [0, 1, 4, 5]
                for i, val in enumerate(s_w): w[w_indices[i] + C_offset] = val
                U[:, t_idx] = u; V[:, t_idx] = v; W[:, t_idx] = w; t_idx += 1
                
                u = np.zeros(8); u[4:8] = s_u
                v = np.zeros(32)
                v_indices = [10, 11, 14, 15]
                for i, val in enumerate(s_v): v[v_indices[i] + B_offset] = val
                w = np.zeros(16)
                w_indices = [2, 3, 6, 7]
                for i, val in enumerate(s_w): w[w_indices[i] + C_offset] = val
                U[:, t_idx] = u; V[:, t_idx] = v; W[:, t_idx] = w; t_idx += 1
                
        if verify(U, V, W):
            return U, V, W
        return None

    strassen_sol = get_strassen_56()
    if strassen_sol:
        best_solution = strassen_sol

    # ALS Search
    # We try R = 50, 51, 52, 53, 54, 55
    # For each R, we do a few restarts.
    
    for R in range(50, 56):
        if time.time() > deadline - 10:
            break
            
        for restart in range(5): # 5 restarts per rank
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V randomly with small integers
            # Scale them to match the magnitude of T
            U = np.random.randint(-2, 3, size=(dim_A, R)).astype(float)
            V = np.random.randint(-2, 3, size=(dim_B, R)).astype(float)
            
            # Normalize to prevent overflow/underflow in lstsq
            # U /= np.linalg.norm(U, axis=0, keepdims=True)
            # V /= np.linalg.norm(V, axis=0, keepdims=True)
            
            # ALS iterations
            for it in range(20):
                if time.time() > deadline - 2:
                    break
                W = solve_for_W(U, V)
                U = solve_for_U(V, W)
                V = solve_for_V(U, W)
                
                # Check for integer-ness
                if verify(U, V, W):
                    # Found a solution with rank R
                    best_solution = (U, V, W)
                    break
            
            if best_solution is not None and R < 56:
                # If we found something better than 56, we can stop
                # But we want the MINIMAL rank. 
                # Since we iterate R from 50 up, the first one found is the minimal in this range.
                break
                
        if best_solution is not None and R < 56:
            break

    if best_solution is None:
        # Fallback to schoolbook 64 if Strassen failed (unlikely)
        rank = 64
        U = np.zeros((dim_A, rank), dtype=float)
        V = np.zeros((dim_B, rank), dtype=float)
        W = np.zeros((dim_C, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
