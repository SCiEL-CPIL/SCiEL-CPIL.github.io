import numpy as np
import time
import itertools

def construct():
    n, m, p = 3, 3, 7
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0
    
    # 1. Build the target tensor T
    # T[u, v, w] = 1 if (i,j) from u, (j,k) from v, (k,i) from w are consistent
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T = np.zeros((num_u, num_v, num_w), dtype=np.float64)
    for u in range(num_u):
        i = u // m
        j = u % m
        for v in range(num_v):
            j_v = v // p
            k = v % p
            if j != j_v:
                continue
            for w in range(num_w):
                k_w = w // n
                i_w = w % n
                if k == k_w and i == i_w:
                    T[u, v, w] = 1.0

    # Schoolbook solution as fallback
    R_school = n * m * p
    U_school = np.zeros((num_u, R_school))
    V_school = np.zeros((num_v, R_school))
    W_school = np.zeros((num_w, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1

    def verify(U, V, W):
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        diff = T_calc - T
        return np.linalg.norm(diff) < 1e-8

    def round_to_half_ints(U, V, W):
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r

    # Strategy:
    # 1. Try known small ranks? For 3x3x7, the rank is at least max(n*m, m*p, p*n) = 21.
    #    Actually, the rank of matrix multiplication tensor M_{n,m,p} is at least max(n*m, m*p, p*n).
    #    For 3x3x7, min rank is at least 21.
    #    Strassen's algorithm reduces rank for square matrices. For rectangular, it's less clear.
    #    Let's try to find a decomposition of rank 21, 22, ... up to 63.
    #    However, optimization of tensor rank is NP-hard.
    #    We will use a heuristic: fix U and V, solve for W?
    #    Or use alternating least squares (ALS).
    
    # Let's try ALS with random restarts for R from 21 to 30.
    # If we find a valid real solution, try to round it.
    # If rounding fails, try to find an integer/half-integer solution directly via search?
    # Search space is too large.
    
    # Alternative: Use the structure of the tensor.
    # The tensor is a "slice" of the 3x3x3 tensor replicated? No.
    
    # Let's implement a simple ALS optimizer.
    
    def als_optimize(R, max_iter=100, tol=1e-10):
        np.random.seed(42 + R)
        U = np.random.randn(num_u, R)
        V = np.random.randn(num_v, R)
        W = np.random.randn(num_w, R)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_err = np.inf
        for it in range(max_iter):
            # Update U: Fix V, W. Minimize || T - sum_r u_r v_r w_r ||
            # This is a linear least squares problem for each row of U?
            # T[u, v, w] = sum_r U[u, r] * (V[v, r] * W[w, r])
            # Let M_u[v, w] = T[u, v, w]
            # Let B_r[v, w] = V[v, r] * W[w, r]
            # Then M_u = sum_r U[u, r] B_r
            # We can stack B_r into a matrix B of size (num_v*num_w, R)
            # And M_u into a vector m_u of size (num_v*num_w)
            # Then solve B^T B u_u = B^T m_u for u_u (vector of length R)
            
            B = np.einsum('vr, wr -> vwr', V, W).reshape(num_v * num_w, R)
            BtB = B.T @ B
            BtB_inv = np.linalg.pinv(BtB)
            
            for u in range(num_u):
                m_u = T[u, :, :].flatten()
                Btm = B.T @ m_u
                u_u = BtB_inv @ Btm
                U[u, :] = u_u
            
            # Similarly for V
            A = np.einsum('ur, wr -> uwr', U, W).reshape(num_u * num_w, R)
            AtA = A.T @ A
            AtA_inv = np.linalg.pinv(AtA)
            for v in range(num_v):
                # T[:, v, :] -> reshape to (num_u * num_w)
                # Note: T is (u, v, w). We need to extract T[:, v, :]
                # This is not contiguous in C order.
                # T[:, v, :] has shape (num_u, num_w)
                m_v = T[:, v, :].flatten()
                Atm = A.T @ m_v
                v_v = AtA_inv @ Atm
                V[v, :] = v_v
            
            # Similarly for W
            C = np.einsum('ur, vr -> uv r', U, V).reshape(num_u * num_v, R)
            CtC = C.T @ C
            CtC_inv = np.linalg.pinv(CtC)
            for w in range(num_w):
                # T[:, :, w]
                m_w = T[:, :, w].flatten()
                Ctm = C.T @ m_w
                w_w = CtC_inv @ Ctm
                W[w, :] = w_w
            
            # Check convergence
            T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
            err = np.linalg.norm(T_calc - T)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W, prev_err

    best_rank = R_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks 21 to 25 with a few restarts
    for R in range(21, 26):
        if time.time() > deadline:
            break
        for restart in range(5):
            if time.time() > deadline:
                break
            # Use different seed
            np.random.seed(1000 + R * 100 + restart)
            # Run ALS
            U, V, W, err = als_optimize(R, max_iter=50)
            if err < 1e-6:
                # Found a real solution
                # Try rounding
                U_r, V_r, W_r = round_to_half_ints(U, V, W)
                if verify(U_r, V_r, W_r):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                else:
                    # If rounding fails, check if the real solution is close enough to integers/half-integers
                    # Or maybe we can return the real solution if the evaluator accepts floats?
                    # The task says "integer/half-integer entries". So we must round.
                    # If rounding fails, we might need to search for a nearby valid solution.
                    # For now, skip.
                    pass
            elif err < 1e-3:
                # Close, maybe try more iterations?
                pass

    # If no better rank found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
