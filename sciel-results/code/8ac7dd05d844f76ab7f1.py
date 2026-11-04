import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2x2():
    """
    Returns U, V, W matrices (4x7) for the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    Indices: a=i*2+j, b=j*2+k, c=k*2+i for i,j,k in {0,1}.
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # u vectors (coefficients for A):
    u1 = [1, 0, 0, 1] # A00 + A11
    u2 = [0, 0, 1, 1] # A10 + A11
    u3 = [1, 0, 0, 0] # A00
    u4 = [0, 0, 0, 1] # A11
    u5 = [1, 1, 0, 0] # A00 + A01
    u6 = [-1, 0, 1, 0] # A10 - A00
    u7 = [0, 1, 0, -1] # A01 - A11
    
    # v vectors (coefficients for B):
    v1 = [1, 0, 0, 1] # B00 + B11
    v2 = [1, 0, 0, 0] # B00
    v3 = [0, 1, 0, -1] # B01 - B11
    v4 = [0, 0, 1, -1] # B10 - B00
    v5 = [0, 0, 0, 1] # B11
    v6 = [1, 1, 0, 0] # B00 + B01
    v7 = [0, 0, 1, 1] # B10 + B11
    
    # w vectors (coefficients for C):
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    w1 = [1, 0, 0, 1] # M1: +C00, +C11
    w2 = [0, 0, 1, 0] # M2: +C10
    w3 = [0, 1, 0, 0] # M3: +C01
    w4 = [-1, 0, 1, 0] # M4: -C00, +C10
    w5 = [0, 1, 0, 1] # M5: +C01, +C11
    w6 = [1, 0, 0, -1] # M6: +C00, -C11
    w7 = [-1, 0, 0, 1] # M7: -C00, +C11
    
    U = np.array([u1, u2, u3, u4, u5, u6, u7]).T
    V = np.array([v1, v2, v3, v4, v5, v6, v7]).T
    W = np.array([w1, w2, w3, w4, w5, w6, w7]).T
    
    return U, V, W

def construct_hybrid():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    U_2, V_2, W_2 = get_strassen_2x2x2()
    
    U_4 = []
    V_4 = []
    W_4 = []
    
    for r in range(7):
        for s in range(7):
            u = np.kron(U_2[:, r], U_2[:, s])
            
            v_kron = np.kron(V_2[:, r], V_2[:, s])
            v_full = np.zeros(dim_v)
            for j in range(4):
                for k in range(4):
                    v_full[j * p + k] = v_kron[j * 4 + k]
            
            w_kron = np.kron(W_2[:, r], W_2[:, s])
            w_full = np.zeros(dim_w)
            w_full[:16] = w_kron
            
            U_4.append(u)
            V_4.append(v_full)
            W_4.append(w_full)
            
    U_3 = []
    V_3 = []
    W_3 = []
    
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = np.zeros(dim_u)
                v = np.zeros(dim_v)
                w = np.zeros(dim_w)
                
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                u[a] = 1
                v[b] = 1
                w[c] = 1
                
                U_3.append(u)
                V_3.append(v)
                W_3.append(w)
                
    U_all = np.array(U_4 + U_3).T
    V_all = np.array(V_4 + V_3).T
    W_all = np.array(W_4 + W_3).T
    
    return U_all, V_all, W_all

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1

    # 1. Construct Hybrid Solution (Rank 97)
    U_h, V_h, W_h = construct_hybrid()
    
    # Verify Hybrid
    T_decomp = np.einsum('ar,br,cr->abc', U_h, V_h, W_h)
    if not np.array_equal(T_target, T_decomp):
        # Fallback to pure schoolbook if hybrid fails (should not happen)
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U = U_h
    best_V = V_h
    best_W = W_h
    best_rank = U_h.shape[1]
    
    # 2. Attempt Greedy Peeling to find lower rank
    # This is a structural addition: using search to potentially beat the constructive bound.
    deadline = time.time() + 100.0
    R = T_target.copy().astype(np.float64)
    U_list = []
    V_list = []
    W_list = []
    
    # We only attempt peeling if we have enough time.
    # Peeling is stochastic and might not converge.
    # We run it for a few seconds to see if it can beat 97.
    
    max_peels = best_rank + 10 # Don't go higher than hybrid rank
    
    for _ in range(max_peels):
        if time.time() > deadline:
            break
        if not np.any(R):
            break
            
        u = np.random.choice([-1, 0, 1], size=dim_u)
        v = np.random.choice([-1, 0, 1], size=dim_v)
        w = np.random.choice([-1, 0, 1], size=dim_w)
        
        converged = False
        for _iter in range(20):
            if time.time() > deadline:
                break
            M_u = np.einsum('abc,b,c->a', R, v, w)
            new_u = np.sign(M_u).astype(np.float64)
            M_v = np.einsum('abc,a,c->b', R, u, w)
            new_v = np.sign(M_v).astype(np.float64)
            M_w = np.einsum('abc,a,b->c', R, u, v)
            new_w = np.sign(M_w).astype(np.float64)
            
            if np.array_equal(new_u, u) and np.array_equal(new_v, v) and np.array_equal(new_w, w):
                converged = True
                break
            u, v, w = new_u, new_v, new_w
        
        score = np.einsum('abc,a,b,c', R, u, v, w)
        if score <= 0:
            # Try a few restarts
            found = False
            for _r in range(5):
                if time.time() > deadline:
                    break
                u = np.random.choice([-1, 0, 1], size=dim_u)
                v = np.random.choice([-1, 0, 1], size=dim_v)
                w = np.random.choice([-1, 0, 1], size=dim_w)
                for _i in range(10):
                    M_u = np.einsum('abc,b,c->a', R, v, w)
                    u = np.sign(M_u).astype(np.float64)
                    M_v = np.einsum('abc,a,c->b', R, u, w)
                    v = np.sign(M_v).astype(np.float64)
                    M_w = np.einsum('abc,a,b->c', R, u, v)
                    w = np.sign(M_w).astype(np.float64)
                score = np.einsum('abc,a,b,c', R, u, v, w)
                if score > 0:
                    found = True
                    break
            if not found:
                break
        
        # Subtract the rank-1 term
        R -= np.einsum('a,b,c->abc', u, v, w)
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Check if we have beaten the hybrid rank
        current_peel_rank = len(U_list)
        if current_peel_rank < best_rank:
            # Verify the peeling result is exact
            if np.allclose(R, 0):
                U_p = np.array(U_list).T
                V_p = np.array(V_list).T
                W_p = np.array(W_list).T
                T_p = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                if np.array_equal(T_target, T_p):
                    best_U = U_p
                    best_V = V_p
                    best_W = W_p
                    best_rank = current_peel_rank
                    # If we beat it, we can stop early or continue to see if we can go lower
                    # But for safety, we break if we find a valid lower rank to ensure we return it.
                    # Actually, we should continue to see if we can go even lower.
                    # But R is 0, so the loop will break.

    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
