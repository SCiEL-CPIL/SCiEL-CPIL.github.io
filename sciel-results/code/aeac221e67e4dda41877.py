import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1

    # Try to find rank 49
    R = 49
    start_time = time.time()
    timeout = 115
    
    # We will try a few random restarts
    best_residual = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(10):
        if time.time() - start_time > timeout:
            break
            
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        
        for it in range(50):
            if time.time() - start_time > timeout:
                break
                
            # Solve for W
            W = np.zeros((p*n, R))
            for c in range(p*n):
                # M_c is (m*p*n*m) x R? No.
                # Equation: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                # For fixed c, variables are W[c, r].
                # Let x = W[c, :].
                # sum_r x_r * (U[:,r] * V[:,r]) = T[:,:,c]
                # Let A_c be a matrix of size (n*m * m*p) x R
                # A_c[(a,b), r] = U[a,r] * V[b,r]
                # Then A_c @ x = T[:,:,c].flatten()
                
                A_c = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(-1, R)
                b_c = T[:, :, c].flatten()
                # Solve A_c x = b_c
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A_c, b_c, rcond=None)
                    W[c, :] = x
                except:
                    W[c, :] = np.random.randn(R)
            
            # Calculate residual
            # Residual = sum_{a,b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
            # This is expensive. We can check a few samples or compute fully.
            # Full computation:
            # Tensor product is expensive.
            # We can compute the residual for the current U, V, W.
            # Let's just check if the time is running out.
            
            # Update U
            U_new = np.zeros_like(U)
            for a in range(n*m):
                # sum_r U[a,r] * (V[b,r] W[c,r]) = T[a,b,c]
                # For fixed a, variables U[a,r].
                # Let y = U[a, :].
                # sum_r y_r * (V[:,r] * W[:,r]) = T[a, :, :]
                # A_a is (m*p * p*n) x R
                # A_a[(b,c), r] = V[b,r] * W[c,r]
                A_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, R)
                b_a = T[a, :, :].flatten()
                try:
                    y, residuals, rank, sv = np.linalg.lstsq(A_a, b_a, rcond=None)
                    U_new[a, :] = y
                except:
                    U_new[a, :] = np.random.randn(R)
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for b in range(m*p):
                # sum_r V[b,r] * (U[a,r] W[c,r]) = T[:, b, c]
                # For fixed b, variables V[b,r].
                # A_b is (n*m * p*n) x R
                # A_b[(a,c), r] = U[a,r] * W[c,r]
                A_b = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, R)
                b_b = T[:, b, :].flatten()
                try:
                    z, residuals, rank, sv = np.linalg.lstsq(A_b, b_b, rcond=None)
                    V_new[b, :] = z
                except:
                    V_new[b, :] = np.random.randn(R)
            V = V_new
            
            # Check residual every few iterations
            if it % 5 == 0:
                # Compute residual
                # This is O(n*m*p*n*m*p*n * R) which is large.
                # 10*25*10*49 = 122500 ops.
                # We can do this.
                residual = 0
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            val = np.dot(U[a, :], np.dot(V[b, :], W[c, :]))
                            target = T[a, b, c]
                            residual += (val - target) ** 2
                if residual < best_residual:
                    best_residual = residual
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                if residual < 1e-6:
                    # Try to round
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    # Verify
                    res_round = 0
                    for a in range(n*m):
                        for b in range(m*p):
                            for c in range(p*n):
                                val = np.dot(U_round[a, :], np.dot(V_round[b, :], W_round[c, :]))
                                target = T[a, b, c]
                                res_round += (val - target) ** 2
                    if res_round < 1e-9:
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }

    # If no better solution found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
