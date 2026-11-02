import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    
    # Build the target tensor T
    T = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    start_time = time.time()
    deadline = 145.0
    
    # Schoolbook solution
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
    
    best_result = (U_school, V_school, W_school)
    best_rank = R_school
    
    def compute_error(U, V, W):
        result = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(result - T)
    
    def check_solution(U, V, W, tol=1e-8):
        result = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(result - T)) < tol
    
    def solve_normal(D, target):
        DTD = D.T @ D
        DTT = D.T @ target
        try:
            X = np.linalg.solve(DTD, DTT)
        except np.linalg.LinAlgError:
            X = np.linalg.lstsq(D, target, rcond=None)[0]
        return X
    
    # Precompute T reshapes
    T_flat_U = T.reshape(-1, num_u).T
    T_flat_V = np.zeros((num_v, num_u * num_w))
    for b in range(num_v):
        T_flat_V[b, :] = T[:, b, :].flatten()
    T_flat_W = np.zeros((num_w, num_u * num_v))
    for c in range(num_w):
        T_flat_W[c, :] = T[:, :, c].flatten()
    
    # Try ranks from 63 down
    for R in range(R_school, 50, -1):
        if time.time() > deadline - 20:
            break
        if R >= best_rank:
            continue
        
        found = False
        num_inits = 25
        
        for init in range(num_inits):
            if time.time() > deadline - 10:
                break
            
            seed = R * 1000 + init
            np.random.seed(seed)
            
            # Vary initialization strategy
            if init < 5:
                scale = 0.2 + 0.1 * init
                U = np.random.randn(num_u, R) * scale
                V = np.random.randn(num_v, R) * scale
                W = np.random.randn(num_w, R) * scale
            elif init < 10:
                # Sparse initialization
                U = np.random.randn(num_u, R) * 0.5
                V = np.random.randn(num_v, R) * 0.5
                W = np.random.randn(num_w, R) * 0.5
                # Zero out some entries
                U[np.random.rand(*U.shape) < 0.5] = 0
                V[np.random.rand(*V.shape) < 0.5] = 0
                W[np.random.rand(*W.shape) < 0.5] = 0
            else:
                # Integer/half-integer initialization
                U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(num_u, R))
                V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(num_v, R))
                W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(num_w, R))
            
            prev_err = np.inf
            stuck_count = 0
            
            for it in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U
                D_U = np.zeros((num_v * num_w, R))
                for r in range(R):
                    D_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                U = solve_normal(D_U, T_flat_U.T).T
                
                # Solve for V
                D_V = np.zeros((num_u * num_w, R))
                for r in range(R):
                    D_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                V = solve_normal(D_V, T_flat_V.T).T
                
                # Solve for W
                D_W = np.zeros((num_u * num_v, R))
                for r in range(R):
                    D_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                W = solve_normal(D_W, T_flat_W.T).T
                
                err = compute_error(U, V, W)
                
                if err < 1e-10:
                    if check_solution(U, V, W):
                        best_result = (U, V, W)
                        best_rank = R
                        found = True
                        break
                    break
                
                if abs(prev_err - err) < 1e-14:
                    stuck_count += 1
                else:
                    stuck_count = 0
                if stuck_count > 25:
                    break
                prev_err = err
            
            if found:
                break
        
        if found:
            continue
        else:
            break
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
