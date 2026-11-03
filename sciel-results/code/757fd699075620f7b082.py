import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110 # Leave 40s margin for safety and fallback
    
    # 1. Try to find Winograd decomposition for 3x3x3 (rank 23)
    U_333, V_333, W_333 = None, None, None
    try:
        U_333, V_333, W_333 = find_decomposition(3, 3, 3, 23, deadline - 20)
    except Exception as e:
        pass
        
    if U_333 is not None:
        # 2. Construct 3x3x7 decomposition using 2x Winograd + 1x Standard
        U_list = []
        V_list = []
        W_list = []
        
        # Helper to map 3x3x3 decomposition to 3x3x7
        def map_333_to_337(U, V, W, offset_k):
            u_new = np.zeros((9, U.shape[1]))
            v_new = np.zeros((21, U.shape[1]))
            w_new = np.zeros((21, U.shape[1]))
            
            for s in range(U.shape[1]):
                u_new[:, s] = U[:, s]
                
                for b_333 in range(9):
                    if V[b_333, s] != 0:
                        # b_333 = 3j + k
                        j = b_333 // 3
                        k = b_333 % 3
                        b_337 = j * 7 + (k + offset_k)
                        v_new[b_337, s] = V[b_333, s]
                        
                for c_333 in range(9):
                    if W[c_333, s] != 0:
                        # c_333 = 3k + i
                        k = c_333 // 3
                        i = c_333 % 3
                        c_337 = (k + offset_k) * 3 + i
                        w_new[c_337, s] = W[c_333, s]
                        
            return u_new, v_new, w_new

        # Add first 3 columns (offset 0)
        u1, v1, w1 = map_333_to_337(U_333, V_333, W_333, 0)
        U_list.append(u1)
        V_list.append(v1)
        W_list.append(w1)
        
        # Add next 3 columns (offset 3)
        u2, v2, w2 = map_333_to_337(U_333, V_333, W_333, 3)
        U_list.append(u2)
        V_list.append(v2)
        W_list.append(w2)
        
        # Add last column (offset 6) using standard 3x3x1
        # Standard 3x3x1 has rank 9
        u3 = np.zeros((9, 9))
        v3 = np.zeros((21, 9))
        w3 = np.zeros((21, 9))
        
        for i in range(3):
            for j in range(3):
                s = i * 3 + j
                a = i * 3 + j
                b = j * 7 + 6
                c = 6 * 3 + i
                u3[a, s] = 1
                v3[b, s] = 1
                w3[c, s] = 1
                
        U_list.append(u3)
        V_list.append(v3)
        W_list.append(w3)
        
        U_final = np.hstack(U_list)
        V_final = np.hstack(V_list)
        W_final = np.hstack(W_list)
        
        # Verify
        if verify(U_final, V_final, W_final):
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    # 3. Fallback to numerical search for 3x3x7
    # Try ranks from 54 down to 45
    for r in range(54, 44, -1):
        if time.time() > deadline:
            break
        res = find_decomposition(n, m, p, r, deadline - 10)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # 4. Final fallback: Schoolbook
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

def verify(U, V, W):
    n, m, p = DIMENSIONS
    U_int = U * 2
    V_int = V * 2
    W_int = W * 2
    
    if not np.allclose(U_int, np.round(U_int)):
        return False
    if not np.allclose(V_int, np.round(V_int)):
        return False
    if not np.allclose(W_int, np.round(W_int)):
        return False
        
    U_int = np.round(U_int).astype(np.int64)
    V_int = np.round(V_int).astype(np.int64)
    W_int = np.round(W_int).astype(np.int64)
    
    T_calc = np.einsum('ak,bk,ck->abc', U_int, V_int, W_int)
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 8
                
    return np.array_equal(T_calc, T_target)

def find_decomposition(n, m, p, r, time_limit, seed=0):
    np.random.seed(seed)
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
                
    for it in range(50):
        if time.time() > time_limit:
            break
            
        # Update U
        for a in range(n*m):
            V_rep = np.repeat(V, p*n, axis=0)
            W_rep = np.tile(W, (m*p, 1))
            M_a = V_rep * W_rep
            t_a = T[a, :, :].flatten()
            U[a, :], residuals, rank, sv = np.linalg.lstsq(M_a, t_a, rcond=None)
            
        # Update V
        for b in range(m*p):
            U_rep = np.repeat(U, p*n, axis=0)
            W_rep = np.tile(W, (n*m, 1))
            M_b = U_rep * W_rep
            t_b = T[:, b, :].flatten()
            V[b, :], residuals, rank, sv = np.linalg.lstsq(M_b, t_b, rcond=None)
            
        # Update W
        for c in range(p*n):
            U_rep = np.repeat(U, m*p, axis=0)
            V_rep = np.tile(V, (n*m, 1))
            M_c = U_rep * V_rep
            t_c = T[:, :, c].flatten()
            W[c, :], residuals, rank, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
            
        if it % 5 == 0:
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            W_rounded = np.round(W * 2) / 2.0
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
                
    U_rounded = np.round(U * 2) / 2.0
    V_rounded = np.round(V * 2) / 2.0
    W_rounded = np.round(W * 2) / 2.0
    if verify(U_rounded, V_rounded, W_rounded):
        return U_rounded, V_rounded, W_rounded
    return None
