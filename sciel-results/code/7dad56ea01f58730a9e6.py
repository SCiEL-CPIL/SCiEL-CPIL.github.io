import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Construct the target tensor unfoldings
    # T is a 3D tensor with indices u, v, w
    # u in [0, n*m), v in [0, m*p), w in [0, p*n)
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k, else 0
    
    # We will work with the first unfolding T1: shape (n*m, m*p * p*n)
    # T1[u, v*w_idx] where v*w_idx is flattened index of (v, w)
    # Actually, let's just store the non-zero entries or construct the matrices directly for ALS.
    
    # For ALS, we need T1, T2, T3.
    # T1 is (n*m) x (m*p * p*n)
    # T2 is (m*p) x (n*m * p*n)
    # T3 is (p*n) x (n*m * m*p)
    
    # It is more efficient to store the lists of (index, value) for non-zeros.
    # But for ALS, we need to multiply U @ M.T.
    # M is constructed from V and W.
    
    # Let's precompute the mapping for non-zero entries.
    # For each r, we need to update U, V, W.
    
    # We will use a deadline.
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Define the target tensor as a sparse structure for fast computation
    # We need to compute residual or solve linear systems.
    # For ALS step for U:
    # T1 = U @ M_VW.T
    # M_VW is (m*p * p*n) x R
    # M_VW[v * (p*n) + w, r] = V[v, r] * W[w, r]
    
    # To solve U = T1 @ M_VW.T+
    # We need T1. T1 is mostly zeros.
    # T1[u, v*(p*n) + w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    
    # Let's create the non-zero indices for T1, T2, T3.
    # T1 non-zeros: (u, v, w) where u=i*m+j, v=j*p+k, w=k*n+i
    t1_rows = []
    t1_cols = []
    t1_vals = []
    
    t2_rows = []
    t2_cols = []
    t2_vals = []
    
    t3_rows = []
    t3_cols = []
    t3_vals = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                
                # T1: row u, col v * (p*n) + w
                t1_rows.append(u)
                t1_cols.append(v * (p * n) + w)
                t1_vals.append(1.0)
                
                # T2: row v, col u * (p*n) + w
                t2_rows.append(v)
                t2_cols.append(u * (p * n) + w)
                t2_vals.append(1.0)
                
                # T3: row w, col u * (m * p) + v
                t3_rows.append(w)
                t3_cols.append(u * (m * p) + v)
                t3_vals.append(1.0)
                
    t1_rows = np.array(t1_rows)
    t1_cols = np.array(t1_cols)
    t1_vals = np.array(t1_vals)
    
    t2_rows = np.array(t2_rows)
    t2_cols = np.array(t2_cols)
    t2_vals = np.array(t2_vals)
    
    t3_rows = np.array(t3_rows)
    t3_cols = np.array(t3_cols)
    t3_vals = np.array(t3_vals)
    
    def compute_residual(U, V, W, R):
        # Compute sum_r U[:,r] V[:,r] W[:,r] and compare to T
        # We only check the non-zero positions
        # T1[u, v, w] should be 1
        # Sum_r U[u,r] V[v,r] W[w,r]
        
        # Efficiently compute the contraction for the specific indices
        # This might be slow if done naively.
        # Let's just compute the full tensor if R is small?
        # n*m*p is 60. R is ~50.
        # Full tensor is 12*20*15 = 3600 entries.
        # Contraction is O(R * 3600). 50 * 3600 = 180,000 ops. Very fast.
        
        T_comp = np.zeros((n * m, m * p, p * n))
        for r in range(R):
            T_comp += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            
        # Check error at non-zero positions
        # T1 non-zeros are at (u, v, w)
        # T_comp[u, v, w] should be 1
        err = 0.0
        for idx in range(len(t1_rows)):
            u = t1_rows[idx]
            # v and w are encoded in t1_cols? No, we need v and w separately.
            # We stored t1_cols as v * (p*n) + w.
            v = t1_cols[idx] // (p * n)
            w = t1_cols[idx] % (p * n)
            err += (T_comp[u, v, w] - 1.0) ** 2
        return np.sqrt(err)

    def als_step(U, V, W, R, max_iter=50):
        # Alternating Least Squares
        # We fix V, W and solve for U.
        # T1 = U @ M_VW.T
        # M_VW is (m*p * p*n) x R
        # M_VW[idx, r] = V[v, r] * W[w, r] where idx = v * (p*n) + w
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Solve for U
            # Construct M_VW
            # V is (m*p, R), W is (p*n, R)
            # We need M_VW of size (m*p * p*n, R)
            # M_VW[v * (p*n) + w, r] = V[v, r] * W[w, r]
            # This is the Kronecker product? No.
            # It's the element-wise product of V[:, r] repeated and W[:, r] repeated?
            # Actually, M_VW[:, r] = np.kron(V[:, r], W[:, r])?
            # Let's check indices.
            # v in 0..m*p-1, w in 0..p*n-1.
            # np.kron(V[:, r], W[:, r]) has size (m*p) * (p*n).
            # Index is v * (p*n) + w.
            # Value is V[v, r] * W[w, r].
            # Yes!
            
            M_VW = np.zeros((m * p * p * n, R))
            for r in range(R):
                M_VW[:, r] = np.kron(V[:, r], W[:, r])
                
            # Solve U @ M_VW.T = T1
            # U = T1 @ M_VW.T+
            # T1 is sparse. We only have non-zeros.
            # But we can construct the dense T1? 12 x 300. Small.
            T1 = np.zeros((n * m, m * p * p * n))
            T1[t1_rows, t1_cols] = t1_vals
            
            # Least squares
            # U = T1 @ np.linalg.pinv(M_VW.T)
            # M_VW.T is R x 300.
            # Pinv is 300 x R.
            # T1 @ (300 x R) -> 12 x R.
            try:
                U_new = T1 @ np.linalg.pinv(M_VW.T)
            except np.linalg.LinAlgError:
                break
                
            # Solve for V
            # T2 = V @ M_WU.T
            # M_WU is (p*n * n*m) x R
            # M_WU[w * (n*m) + u, r] = W[w, r] * U[u, r]
            # Index in T2 is v, col is u * (p*n) + w?
            # T2 rows are v (m*p). Cols are (u, w) flattened?
            # T2[v, u * (p*n) + w] = T[v, u, w]
            # So M_WU must have rows corresponding to (u, w).
            # M_WU[u * (p*n) + w, r] = U[u, r] * W[w, r]
            # This is np.kron(U[:, r], W[:, r])?
            # Size (n*m) * (p*n).
            # Index u * (p*n) + w.
            # Yes.
            
            M_WU = np.zeros((n * m * p * n, R))
            for r in range(R):
                M_WU[:, r] = np.kron(U_new[:, r], W[:, r])
                
            T2 = np.zeros((m * p, n * m * p * n))
            T2[t2_rows, t2_cols] = t2_vals
            
            try:
                V_new = T2 @ np.linalg.pinv(M_WU.T)
            except np.linalg.LinAlgError:
                break
                
            # Solve for W
            # T3 = W @ M_UV.T
            # M_UV is (n*m * m*p) x R
            # M_UV[u * (m*p) + v, r] = U[u, r] * V[v, r]
            # Index u * (m*p) + v.
            
            M_UV = np.zeros((n * m * m * p, R))
            for r in range(R):
                M_UV[:, r] = np.kron(U_new[:, r], V_new[:, r])
                
            T3 = np.zeros((p * n, n * m * m * p))
            T3[t3_rows, t3_cols] = t3_vals
            
            try:
                W_new = T3 @ np.linalg.pinv(M_UV.T)
            except np.linalg.LinAlgError:
                break
                
            U, V, W = U_new, V_new, W_new
            
        return U, V, W

    # Try to find a lower rank
    # We will try ranks from 60 down to 50
    # For each rank, we do a few random restarts
    
    for R in range(rank_schoolbook - 1, 49, -1):
        if time.time() > deadline:
            break
            
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Run ALS
            U, V, W = als_step(U, V, W, R, max_iter=100)
            
            # Check residual
            res = compute_residual(U, V, W, R)
            if res < 1e-10:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded solution
                res_r = compute_residual(U_r, V_r, W_r, R)
                if res_r < 1e-10:
                    best_rank = R
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
        
        if found:
            # We found a lower rank. We can try to go even lower?
            # The loop continues to R-1 automatically.
            pass

    # Return the best found
    # Convert to lists of lists as required?
    # The contract says: "factor matrices of shapes ... with integer/half-integer entries"
    # It doesn't specify numpy arrays vs lists, but the seed used lists.
    # I will return lists to be safe, or numpy arrays if accepted.
    # The seed returned lists. I will return lists.
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
