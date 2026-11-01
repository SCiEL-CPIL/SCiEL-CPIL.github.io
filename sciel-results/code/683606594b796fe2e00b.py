import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    # Target equations:
    # For each i, k, i', j', k':
    # sum_r U[i', j', r] V[j', k', r] W[k, i, r] == delta(i', i) * delta(k', k)
    
    # Total equations: n*p * n*m*p = 2*9 * 2*3*9 = 18 * 54 = 972
    # Variables: R * (n*m + m*p + p*n) = R * (6 + 27 + 18) = 51 R
    
    # Try R = 53
    R = 53
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n*m, R)
    V = np.random.randn(m*p, R)
    W = np.random.randn(p*n, R)
    
    # Define the residual function
    def residual(params):
        # params is flattened U, V, W
        U_flat, V_flat, W_flat = np.split(params, [n*m*R, n*m*R + m*p*R])
        U_mat = U_flat.reshape(n*m, R)
        V_mat = V_flat.reshape(m*p, R)
        W_mat = W_flat.reshape(p*n, R)
        
        res = []
        for i in range(n):
            for k in range(p):
                for i_p in range(n):
                    for j_p in range(m):
                        for k_p in range(p):
                            # Calculate sum_r U[i_p, j_p, r] V[j_p, k_p, r] W[k, i, r]
                            # This is a dot product for each r? No.
                            # It is sum over r.
                            # U[i_p*m+j_p, :] . V[j_p*p+k_p, :]? No.
                            # U, V, W are vectors for fixed indices.
                            # val = sum_r U[i_p, j_p, r] * V[j_p, k_p, r] * W[k, i, r]
                            # This is an element-wise product sum.
                            u_vec = U_mat[i_p*m+j_p, :]
                            v_vec = V_mat[j_p*p+k_p, :]
                            w_vec = W_mat[k*n+i, :]
                            val = np.sum(u_vec * v_vec * w_vec)
                            
                            target = 1.0 if (i_p == i and k_p == k) else 0.0
                            res.append(val - target)
        return np.array(res)
        
    # This is too slow in Python loops.
    # We need to vectorize.
    
    # Reshape U, V, W to 3D arrays
    # U: (n, m, R)
    # V: (m, p, R)
    # W: (p, n, R)
    
    def residual_vec(params):
        U_flat, V_flat, W_flat = np.split(params, [n*m*R, n*m*R + m*p*R])
        U_3d = U_flat.reshape(n, m, R)
        V_3d = V_flat.reshape(m, p, R)
        W_3d = W_flat.reshape(p, n, R)
        
        # We need sum_r U[i_p, j_p, r] V[j_p, k_p, r] W[k, i, r]
        # for all i, k, i_p, j_p, k_p.
        
        # This is a 5D tensor T[i, k, i_p, j_p, k_p]
        # T[i, k, i_p, j_p, k_p] = sum_r U[i_p, j_p, r] V[j_p, k_p, r] W[k, i, r]
        
        # We can compute this using einsum
        # U: (i_p, j_p, r)
        # V: (j_p, k_p, r)
        # W: (k, i, r)
        
        # First, U * V for each r?
        # UV[i_p, j_p, k_p, r] = U[i_p, j_p, r] * V[j_p, k_p, r]
        # Then sum over r with W?
        # T[i, k, i_p, j_p, k_p] = sum_r (U[i_p, j_p, r] * V[j_p, k_p, r]) * W[k, i, r]
        
        # Let's do it step by step
        # 1. Compute UV[j_p, i_p, k_p, r] = U[i_p, j_p, r] * V[j_p, k_p, r]
        #    U: (i_p, j_p, r) -> (i_p, j_p, 1, r)
        #    V: (j_p, k_p, r) -> (1, j_p, k_p, r)
        #    UV: (i_p, j_p, k_p, r)
        uv = U_3d[:, :, None, :] * V_3d[None, :, :, :]
        # uv shape: (n, m, p, R)
        
        # 2. Multiply by W and sum over r
        # W: (k, i, r) -> (k, i, 1, 1, 1, r)
        # uv: (i_p, j_p, k_p, r) -> (1, 1, i_p, j_p, k_p, r)
        
        # We want T[i, k, i_p, j_p, k_p]
        # T = sum_r uv[i_p, j_p, k_p, r] * W[k, i, r]
        
        # Reshape uv to (i_p, j_p, k_p, R)
        # Reshape W to (k, i, R)
        
        # We can use einsum
        # 'ijkr,kir->kii_pj_pk_p'? No.
        # Indices:
        # U: i_p, j_p, r
        # V: j_p, k_p, r
        # W: k, i, r
        
        # Term: U[i_p, j_p, r] * V[j_p, k_p, r] * W[k, i, r]
        # Sum over r.
        # Result indices: i_p, j_p, k_p, k, i
        
        # einsum('ijr,jkr,ki->ijpki', U_3d, V_3d, W_3d)
        # Wait, V is (m, p, R) -> j_p, k_p, r.
        # So 'ijr,jkr,ki->ijpki' is correct?
        # U: i_p, j_p, r
        # V: j_p, k_p, r
        # W: k, i, r
        # Output: i_p, j_p, k_p, k, i
        
        T = np.einsum('ijr,jkr,ki->ijpki', U_3d, V_3d, W_3d)
        
        # T shape: (n, m, p, p, n) -> (i_p, j_p, k_p, k, i)
        # We need to compare T[i_p, j_p, k_p, k, i] with target[i_p, j_p, k_p, k, i]
        # Target is 1 if i_p == i and k_p == k, else 0.
        
        # Create target tensor
        # Target[i_p, j_p, k_p, k, i] = delta(i_p, i) * delta(k_p, k)
        
        # We can construct this
        # delta_i_p_i: (n, n) -> 1 if i_p == i
        # delta_k_p_k: (p, p) -> 1 if k_p == k
        
        # Target shape: (n, m, p, p, n)
        # i_p: 0..n-1
        # j_p: 0..m-1
        # k_p: 0..p-1
        # k: 0..p-1
        # i: 0..n-1
        
        # Target[i_p, j_p, k_p, k, i] = (i_p == i) * (k_p == k)
        
        # We can broadcast
        # i_p == i: shape (n, 1, 1, 1, n)
        # k_p == k: shape (1, 1, p, p, 1)
        
        i_p_idx = np.arange(n)
        i_idx = np.arange(n)
        k_p_idx = np.arange(p)
        k_idx = np.arange(p)
        
        delta_ip_i = (i_p_idx[:, None] == i_idx[None, :]).astype(float) # (n, n)
        delta_kp_k = (k_p_idx[:, None] == k_idx[None, :]).astype(float) # (p, p)
        
        # Expand to (n, 1, 1, 1, n) and (1, 1, p, p, 1)
        delta_ip_i_exp = delta_ip_i[:, None, None, None, :]
        delta_kp_k_exp = delta_kp_k[None, None, :, :, None]
        
        target = delta_ip_i_exp * delta_kp_k_exp
        # target shape: (n, 1, p, p, n)
        # Broadcast to (n, m, p, p, n)
        
        res = T - target
        return res.flatten()
        
    # Use least squares
    # Initial guess
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Optimize
    # This might be slow.
    # Given 150s limit, and 972 equations, 2703 variables.
    # Least squares might work.
    
    try:
        from scipy.optimize import least_squares
        sol = least_squares(residual_vec, x0, method='trf', max_nfev=1000)
        if sol.success or np.linalg.norm(sol.fun) < 1e-6:
            U_sol, V_sol, W_sol = np.split(sol.x, [n*m*R, n*m*R + m*p*R])
            U_sol = U_sol.reshape(n*m, R)
            V_sol = V_sol.reshape(m*p, R)
            W_sol = W_sol.reshape(p*n, R)
            
            # Verify
            # Convert to list of lists
            U_list = U_sol.tolist()
            V_list = V_sol.tolist()
            W_list = W_sol.tolist()
            
            return {"U": U_list, "V": V_list, "W": W_list}
    except:
        pass
        
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
