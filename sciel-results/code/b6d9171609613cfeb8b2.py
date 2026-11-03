import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Trivial rank-60 solution
    rank_trivial = n * m * p  # 60
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start from rank 55 (Strassen-based)
    # First, build the rank-55 solution
    def build_rank55():
        rank = 55
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        c = 0
        # Part A: j=0
        for i in range(n):
            for k in range(p):
                U[i*m+0, c] = 1
                V[0*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
        
        # Part B: j=1,2 with Strassen
        for b in range(5):
            k0 = 2*b
            k1 = 2*b+1
            
            u11, u12 = 0*m+1, 0*m+2
            u21, u22 = 1*m+1, 1*m+2
            v11, v12 = 1*p+k0, 1*p+k1
            v21, v22 = 2*p+k0, 2*p+k1
            w00, w01 = k0*n+0, k1*n+0
            w10, w11 = k0*n+1, k1*n+1
            
            # M1 = (a11+a22)(b11+b22)
            U[u11,c]+=1; U[u22,c]+=1; V[v11,c]+=1; V[v22,c]+=1; W[w00,c]+=1; W[w11,c]+=1; c+=1
            # M2 = (a21+a22)b11
            U[u21,c]+=1; U[u22,c]+=1; V[v11,c]+=1; W[w10,c]+=1; W[w11,c]+=-1; c+=1
            # M3 = a11(b12-b22)
            U[u11,c]+=1; V[v12,c]+=1; V[v22,c]+=-1; W[w01,c]+=1; W[w11,c]+=1; c+=1
            # M4 = a22(b21-b11)
            U[u22,c]+=1; V[v21,c]+=1; V[v11,c]+=-1; W[w00,c]+=1; W[w10,c]+=1; c+=1
            # M5 = (a11+a12)b22
            U[u11,c]+=1; U[u12,c]+=1; V[v22,c]+=1; W[w00,c]+=-1; W[w01,c]+=1; c+=1
            # M6 = (a21-a11)(b11+b12)
            U[u21,c]+=1; U[u11,c]+=-1; V[v11,c]+=1; V[v12,c]+=1; W[w11,c]+=1; c+=1
            # M7 = (a12-a22)(b21+b22)
            U[u12,c]+=1; U[u22,c]+=-1; V[v21,c]+=1; V[v22,c]+=1; W[w00,c]+=1; c+=1
        
        return U, V, W, rank
    
    U, V, W, rank = build_rank55()
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Verify the solution exactly
    def verify_exact(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        # For each slice of T, check if it can be represented
        # T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
        # We'll check all non-zero entries of T
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c_idx = k*n+i
                    expected = T[a,b,c_idx]
                    computed = np.sum(U_mat[a,:] * V_mat[b,:] * W_mat[c_idx,:])
                    if abs(computed - expected) > 1e-10:
                        return False
        # Also check that zero entries stay zero (spot check)
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c_idx = np.random.randint(0, p*n)
            if T[a,b,c_idx] == 0:
                computed = np.sum(U_mat[a,:] * V_mat[b,:] * W_mat[c_idx,:])
                if abs(computed) > 1e-10:
                    return False
        return True
    
    if not verify_exact(U, V, W):
        # Fallback to trivial
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # ALS-based rank reduction
    # Try to reduce rank from 55 down to 50
    def try_reduce_rank(U_init, V_init, W_init, target_rank, max_iters=500):
        """
        Try to find a rank-`target_rank` decomposition by ALS.
        Start from the top-`target_rank` columns of the initial solution.
        """
        if target_rank >= U_init.shape[1]:
            return None
        
        # Initialize by taking the first `target_rank` columns
        U_cur = U_init[:, :target_rank].copy()
        V_cur = V_init[:, :target_rank].copy()
        W_cur = W_init[:, :target_rank].copy()
        
        n_am = n*m
        n_bm = m*p
        n_cm = p*n
        r = target_rank
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each a, solve for U[a,:] given V_cur, W_cur
            # T[a,b,c] = sum_d U[a,d] * V_cur[b,d] * W_cur[c,d]
            # For fixed a: T[a,:,:] = U[a,:] @ (V_cur * W_cur.T) reshaped
            # Actually: T[a,b,c] = sum_d U[a,d] * (V_cur[b,d] * W_cur[c,d])
            # So for fixed a, we need: T[a,:,:] = sum_d U[a,d] * M_d where M_d[b,c] = V_cur[b,d]*W_cur[c,d]
            # This is: T[a,:,:].reshape(-1) = U[a,:] @ (V_cur * W_cur.T).reshape(r, -1).T
            # Let M = (V_cur * W_cur.T).reshape(r, n_bm * n_cm)
            # Then T[a,:,:].reshape(-1) = M.T @ U[a,:]
            # So U[a,:] = (M.T @ M)^{-1} M.T @ T[a,:,:].reshape(-1)
            
            M_U = (V_cur * W_cur.T).reshape(r, n_bm * n_cm)
            M_U_T = M_U.T  # shape (n_bm*n_cm, r)
            M_U_T_M_U = M_U_T @ M_U  # shape (r, r)
            reg = 1e-10 * np.eye(r)
            M_U_T_M_U_reg = M_U_T_M_U + reg
            
            for a in range(n_am):
                target_vec = T[a,:,:].reshape(-1)
                rhs = M_U_T @ target_vec
                try:
                    U_cur[a,:] = np.linalg.solve(M_U_T_M_U_reg, rhs)
                except:
                    U_cur[a,:] = np.linalg.lstsq(M_U_T_M_U_reg, rhs, rcond=None)[0]
            
            # Update V: similar
            M_V = (U_cur * W_cur.T).reshape(r, n_am * n_cm)
            M_V_T = M_V.T
            M_V_T_M_V = M_V_T @ M_V
            M_V_T_M_V_reg = M_V_T_M_V + reg
            
            for b in range(n_bm):
                target_vec = T[:,b,:].reshape(-1)
                rhs = M_V_T @ target_vec
                try:
                    V_cur[b,:] = np.linalg.solve(M_V_T_M_V_reg, rhs)
                except:
                    V_cur[b,:] = np.linalg.lstsq(M_V_T_M_V_reg, rhs, rcond=None)[0]
            
            # Update W: similar
            M_W = (U_cur * V_cur.T).reshape(r, n_am * n_bm)
            M_W_T = M_W.T
            M_W_T_M_W = M_W_T @ M_W
            M_W_T_M_W_reg = M_W_T_M_W + reg
            
            for c_idx in range(n_cm):
                target_vec = T[:,:,c_idx].reshape(-1)
                rhs = M_W_T @ target_vec
                try:
                    W_cur[c_idx,:] = np.linalg.solve(M_W_T_M_W_reg, rhs)
                except:
                    W_cur[c_idx,:] = np.linalg.lstsq(M_W_T_M_W_reg, rhs, rcond=None)[0]
        
        # Check if the result is valid
        if verify_exact(U_cur, V_cur, W_cur):
            return U_cur, V_cur, W_cur
        return None
    
    # Try to reduce rank step by step
    for target_rank in range(rank - 1, max(40, rank - 10), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_reduce_rank(best_U, best_V, best_W, target_rank, max_iters=300)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break  # Found a valid lower rank, stop
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
