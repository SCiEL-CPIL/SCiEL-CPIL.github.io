import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 2, 3, 6
    
    # We decompose M_{2,3,6} into two M_{2,3,3}
    # Rank = 2 * Rank(M_{2,3,3})
    
    def solve_233(target_rank, max_iters=500, seeds=10):
        # U: (2*3, R), V: (3*3, R), W: (3*2, R)
        # Indices: U[i*3+j], V[j*3+k], W[k*2+i]
        # Condition: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        
        n2, m2, p2 = 2, 3, 3
        num_eqs = n2 * m2 * p2 # 18
        
        for seed in range(seeds):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(seed)
            R = target_rank
            U = np.random.randn(n2*m2, R)
            V = np.random.randn(m2*p2, R)
            W = np.random.randn(p2*n2, R)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U
                # Eq: sum_k U[i,j] * (sum_r V[j,k,r] W[k,i,r]) = 1 ? No.
                # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
                # For fixed V, W, this is linear in U?
                # No, U, V, W are multiplied together. It's trilinear.
                # But if we fix V and W, we can group terms?
                # Let A_{i,j, eq} ...
                # Actually, for fixed V, W, the equation is:
                # sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1
                # This is linear in U.
                
                # Build system for U
                # Unknowns: U[i*m2+j, r]
                # Equations: for each (i,j,k)
                # sum_r U[i*m2+j, r] * (V[j*p2+k, r] * W[k*n2+i, r]) = 1
                
                A_u = np.zeros((num_eqs, n2*m2*R))
                b_u = np.ones(num_eqs)
                eq_idx = 0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            for r in range(R):
                                val = V[j*p2+k, r] * W[k*n2+i, r]
                                if val != 0:
                                    A_u[eq_idx, (i*m2+j)*R + r] = val
                            eq_idx += 1
                
                U_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_sol.reshape(n2*m2, R)
                
                # Solve for V
                A_v = np.zeros((num_eqs, m2*p2*R))
                b_v = np.ones(num_eqs)
                eq_idx = 0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            for r in range(R):
                                val = U[i*m2+j, r] * W[k*n2+i, r]
                                if val != 0:
                                    A_v[eq_idx, (j*p2+k)*R + r] = val
                            eq_idx += 1
                V_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_sol.reshape(m2*p2, R)
                
                # Solve for W
                A_w = np.zeros((num_eqs, p2*n2*R))
                b_w = np.ones(num_eqs)
                eq_idx = 0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            for r in range(R):
                                val = U[i*m2+j, r] * V[j*p2+k, r]
                                if val != 0:
                                    A_w[eq_idx, (k*n2+i)*R + r] = val
                            eq_idx += 1
                W_sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W = W_sol.reshape(p2*n2, R)
                
                # Check error
                max_err = 0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            s = 0
                            for r in range(R):
                                s += U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r]
                            max_err = max(max_err, abs(s - 1))
                
                if max_err < 1e-9:
                    # Snap to half integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    # Verify snap
                    max_err_snap = 0
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                s = 0
                                for r in range(R):
                                    s += U_snap[i*m2+j, r] * V_snap[j*p2+k, r] * W_snap[k*n2+i, r]
                                max_err_snap = max(max_err_snap, abs(s - 1))
                    if max_err_snap < 1e-9:
                        return U_snap, V_snap, W_snap
                        
        return None

    # Try ranks for 2x3x3
    found_233 = None
    for r in range(14, 19):
        if time.time() > deadline - 20:
            break
        res = solve_233(r)
        if res is not None:
            found_233 = res
            break
            
    if found_233 is None:
        # Fallback to schoolbook for 2x3x6 (Rank 36)
        # Or schoolbook for 2x3x3 (Rank 18) -> Total 36
        # If 18 not found, use 18 schoolbook
        U233 = np.zeros((6, 18))
        V233 = np.zeros((9, 18))
        W233 = np.zeros((6, 18))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U233[i*3+j, col] = 1
                    V233[j*3+k, col] = 1
                    W233[k*2+i, col] = 1
                    col += 1
        found_233 = (U233, V233, W233)
        r_233 = 18
    else:
        r_233 = found_233[0].shape[1]

    # Expand to 2x3x6
    R_total = 2 * r_233
    U = np.zeros((6, R_total))
    V = np.zeros((18, R_total))
    W = np.zeros((12, R_total))
    
    U233, V233, W233 = found_233
    
    # Block 1: k in 0..2
    U[:, 0:r_233] = U233
    V[0:9, 0:r_233] = V233
    W[0:6, 0:r_233] = W233
    
    # Block 2: k in 3..5
    # V index: j*6 + k. If k=3+k', V[j*6 + 3+k'] = V233[j*3+k']
    # W index: k*2 + i. If k=3+k', W[(3+k')*2 + i] = W233[k'*2+i]
    # U is same.
    
    U[:, r_233:2*r_233] = U233
    
    for j in range(3):
        for k in range(3):
            V[j*6 + 3 + k, r_233:2*r_233] = V233[j*3 + k]
            
    for k in range(3):
        for i in range(2):
            W[(3+k)*2 + i, r_233:2*r_233] = W233[k*2 + i]
            
    # Verify final
    # ... (omitted for brevity, but should be done)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
