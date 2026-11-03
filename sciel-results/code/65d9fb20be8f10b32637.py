import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Search for a low-rank decomposition of the 2x3x2 tensor
    # Then tile it 5 times for p=10
    
    # First, try to find rank-8 for 2x3x2
    # If that fails, fall back to the rank-11 approach (20 + 5*7 = 55)
    
    # Build the target tensor for 2x3x2
    # T[i, j, k] = 1 if it's a valid multiplication term
    # A is 2x3 (indices i in {0,1}, j in {0,1,2})
    # B is 3x2 (indices j in {0,1,2}, k in {0,1})
    # C is 2x2 (indices i in {0,1}, k in {0,1})
    # T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
    
    # Actually the tensor is: for each (i,j,k), the coefficient is 1
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    # Let's search for rank-8
    found_rank8 = False
    U8 = None
    V8 = None
    W8 = None
    
    if time.time() < deadline - 5:
        # Try a few random initializations with simple optimization
        best_err = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        for trial in range(3):
            np.random.seed(42 + trial)
            R = 8
            # Random initialization
            U_init = np.random.randn(n, m, R) * 0.5
            V_init = np.random.randn(m, 2, R) * 0.5  # m=3, k_dim=2
            W_init = np.random.randn(n, 2, R) * 0.5  # n=2, k_dim=2
            
            # Simple iterative refinement
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                    
                # Update U: for each (i,j), solve least squares
                for i in range(n):
                    for j in range(m):
                        # T[i,j,:] should equal sum_k V[j,k,:] * W[i,k,:] * U[i,j,:]
                        # Actually: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
                        # For fixed i,j: vector over k: T[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[i,k,r])
                        # This is: t_vec = U[i,j,:] @ M where M[r,k] = V[j,k,r]*W[i,k,r]... 
                        # Actually t_vec[k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
                        # Let M[r,k] = V[j,k,r] * W[i,k,r], then t_vec = M^T @ U[i,j,:]
                        # So U[i,j,:] = pinv(M^T) @ t_vec... but this is nonlinear coupling
                        
                        # Simpler: just do a gradient step
                        pass
                
                # Use a simpler approach: alternating least squares
                # Fix V,W, update U
                for i in range(n):
                    for j in range(m):
                        # t_vec[k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
                        # Build matrix: A_mat[r,k] = V[j,k,r] * W[i,k,r]
                        # t_vec = A_mat.T @ U[i,j,:]
                        A_mat = np.zeros((R, 2))
                        for r in range(R):
                            for k in range(2):
                                A_mat[r, k] = V_cur[j, k, r] * W_cur[i, k, r]
                        # t_vec[k] = sum_r U[i,j,r] * A_mat[r,k]
                        # So t_vec = A_mat.T @ U[i,j,:]
                        # U[i,j,:] = pinv(A_mat.T) @ t_vec
                        t_vec = np.ones(2)  # T[i,j,k] = 1 for all k
                        try:
                            U_cur[i, j, :] = np.linalg.lstsq(A_mat.T, t_vec, rcond=None)[0]
                        except:
                            pass
                
                # Fix U,W, update V
                for j in range(m):
                    for k in range(2):
                        # t_vec[i] = sum_r V[j,k,r] * U[i,j,r] * W[i,k,r]
                        A_mat = np.zeros((R, n))
                        for r in range(R):
                            for i in range(n):
                                A_mat[r, i] = U_cur[i, j, r] * W_cur[i, k, r]
                        t_vec = np.ones(n)
                        try:
                            V_cur[j, k, :] = np.linalg.lstsq(A_mat.T, t_vec, rcond=None)[0]
                        except:
                            pass
                
                # Fix U,V, update W
                for i in range(n):
                    for k in range(2):
                        # t_vec[j] = sum_r W[i,k,r] * U[i,j,r] * V[j,k,r]
                        A_mat = np.zeros((R, m))
                        for r in range(R):
                            for j in range(m):
                                A_mat[r, j] = U_cur[i, j, r] * V_cur[j, k, r]
                        t_vec = np.ones(m)
                        try:
                            W_cur[i, k, :] = np.linalg.lstsq(A_mat.T, t_vec, rcond=None)[0]
                        except:
                            pass
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(2):
                            val = 0.0
                            for r in range(R):
                                val += U_cur[i, j, r] * V_cur[j, k, r] * W_cur[i, k, r]
                            err += (val - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_U = U_cur.copy()
                    best_V = V_cur.copy()
                    best_W = W_cur.copy()
                
                if best_err < 1e-10:
                    break
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10:
            found_rank8 = True
            U8 = best_U
            V8 = best_V
            W8 = best_W
    
    if found_rank8 and time.time() < deadline - 5:
        # Use rank-8 blocks: total rank = 5 * 8 = 40
        rank = 40
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            for r in range(8):
                col = b * 8 + r
                for i in range(n):
                    for j in range(m):
                        U[i*m + j, col] = U8[i, j, r]
                    for k_off in range(2):
                        k = k0 + k_off
                        W[k*n + i, col] = W8[i, k_off, r]
                for j in range(m):
                    for k_off in range(2):
                        k = k0 + k_off
                        V[j*p + k, col] = V8[j, k_off, r]
        
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: original rank-55 solution
    rank = 55
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    c = 0
    
    for i in range(n):
        for k in range(p):
            U[i*m + 0, c] = 1
            V[0*p + k, c] = 1
            W[k*n + i, c] = 1
            c += 1
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        u11 = 0 * m + 1
        u12 = 0 * m + 2
        u21 = 1 * m + 1
        u22 = 1 * m + 2
        
        v11 = 1 * p + k0
        v12 = 1 * p + k1
        v21 = 2 * p + k0
        v22 = 2 * p + k1
        
        w00 = k0 * n + 0
        w01 = k1 * n + 0
        w10 = k0 * n + 1
        w11 = k1 * n + 1
        
        U[u11, c] += 1; U[u22, c] += 1
        V[v11, c] += 1; V[v22, c] += 1
        W[w00, c] += 1; W[w11, c] += 1
        c += 1
        
        U[u21, c] += 1; U[u22, c] += 1
        V[v11, c] += 1
        W[w10, c] += 1; W[w11, c] += -1
        c += 1
        
        U[u11, c] += 1
        V[v12, c] += 1; V[v22, c] += -1
        W[w01, c] += 1; W[w11, c] += 1
        c += 1
        
        U[u22, c] += 1
        V[v21, c] += 1; V[v11, c] += -1
        W[w00, c] += 1; W[w10, c] += 1
        c += 1
        
        U[u11, c] += 1; U[u12, c] += 1
        V[v22, c] += 1
        W[w00, c] += -1; W[w01, c] += 1
        c += 1
        
        U[u21, c] += 1; U[u11, c] += -1
        V[v11, c] += 1; V[v12, c] += 1
        W[w11, c] += 1
        c += 1
        
        U[u12, c] += 1; U[u22, c] += -1
        V[v21, c] += 1; V[v22, c] += 1
        W[w00, c] += 1
        c += 1
        
        if time.time() > deadline:
            break
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
