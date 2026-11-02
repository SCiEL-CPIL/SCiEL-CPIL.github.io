import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds
    
    # Schoolbook solution
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find lower rank solutions using alternating minimization
    for target_rank in range(53, best_rank):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(seed + target_rank * 100)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Alternating minimization
            for iteration in range(50):
                if time.time() > deadline - 30:
                    break
                
                # Solve for W (linear in W)
                A_w = np.zeros((n*p*n*m*p, p*n*target_rank))
                b_w = np.zeros(n*p*n*m*p)
                eq_idx = 0
                for i in range(n):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    b_w[eq_idx] = 1.0 if (ip == i and kp == k) else 0.0
                                    for r in range(target_rank):
                                        A_w[eq_idx, (kp*n+ip)*target_rank + r] = U[ip*m+jp, r] * V[jp*p+kp, r]
                                    eq_idx += 1
                
                w_sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W = w_sol.reshape(p*n, target_rank)
                
                # Solve for U (linear in U)
                A_u = np.zeros((n*p*n*m*p, n*m*target_rank))
                b_u = np.zeros(n*p*n*m*p)
                eq_idx = 0
                for i in range(n):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    b_u[eq_idx] = 1.0 if (ip == i and kp == k) else 0.0
                                    for r in range(target_rank):
                                        A_u[eq_idx, (ip*m+jp)*target_rank + r] = V[jp*p+kp, r] * W[kp*n+ip, r]
                                    eq_idx += 1
                
                u_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = u_sol.reshape(n*m, target_rank)
                
                # Solve for V (linear in V)
                A_v = np.zeros((n*p*n*m*p, m*p*target_rank))
                b_v = np.zeros(n*p*n*m*p)
                eq_idx = 0
                for i in range(n):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    b_v[eq_idx] = 1.0 if (ip == i and kp == k) else 0.0
                                    for r in range(target_rank):
                                        A_v[eq_idx, (jp*p+kp)*target_rank + r] = U[ip*m+jp, r] * W[kp*n+ip, r]
                                    eq_idx += 1
                
                v_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = v_sol.reshape(m*p, target_rank)
                
                # Check error
                A_check = np.zeros((n*p*n*m*p, p*n*target_rank))
                b_check = np.zeros(n*p*n*m*p)
                eq_idx = 0
                for i in range(n):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    b_check[eq_idx] = 1.0 if (ip == i and kp == k) else 0.0
                                    for r in range(target_rank):
                                        A_check[eq_idx, (kp*n+ip)*target_rank + r] = U[ip*m+jp, r] * V[jp*p+kp, r]
                                    eq_idx += 1
                
                error = np.linalg.norm(A_check @ W.flatten() - b_check)
                
                if error < 1e-10:
                    # Try rounding
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    A_check2 = np.zeros((n*p*n*m*p, p*n*target_rank))
                    b_check2 = np.zeros(n*p*n*m*p)
                    eq_idx = 0
                    for i in range(n):
                        for k in range(p):
                            for ip in range(n):
                                for jp in range(m):
                                    for kp in range(p):
                                        b_check2[eq_idx] = 1.0 if (ip == i and kp == k) else 0.0
                                        for r in range(target_rank):
                                            A_check2[eq_idx, (kp*n+ip)*target_rank + r] = U_r[ip*m+jp, r] * V_r[jp*p+kp, r]
                                        eq_idx += 1
                    
                    error2 = np.linalg.norm(A_check2 @ W_r.flatten() - b_check2)
                    
                    if error2 < 1e-10:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
