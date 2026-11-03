import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _verify_decomposition_full(U, V, W, n, m, p):
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
                
    Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    if not np.allclose(Rec, T, atol=1e-10):
        return False
    return True

def _alternating_minimization(rank, n, m, p, max_iters=100, time_limit=10):
    start_time = time.time()
    target_time = start_time + time_limit
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
                
    # Initialize
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for _ in range(max_iters):
        if time.time() > target_time:
            break
            
        # Solve for U: T_uvw = sum_r U_ur V_vr W_wr
        # For each u, T_u is a matrix of size (m*p) x (p*n)
        # T_u[v, w] = sum_r U_ur V_vr W_wr
        # Let M_u = V^T @ W? No.
        # T_u = U_u @ (V^T W) ? No.
        # T_u[v, w] = sum_r U_ur (V_vr W_wr)
        # Let Z_r = V[:, r] * W[:, r] (element-wise product, size m*p and p*n? No, V is m*p x rank, W is p*n x rank)
        # V_vr is scalar. W_wr is scalar.
        # So T_u[v, w] = sum_r U_ur (V_vr W_wr)
        # This is U_u @ M where M is a matrix of size rank x (m*p * p*n)?
        # No, V and W have different row dimensions.
        # We can reshape T_u to a vector of length (m*p)*(p*n).
        # Then T_u_vec = U_u @ (V_flat_r * W_flat_r) ?
        # Let's use the fact that:
        # T = sum_r U_r (outer) V_r (outer) W_r
        # This is a 3-tensor.
        # Fixing V and W, the problem is linear in U.
        # T[u, v, w] = sum_r U[u, r] * (V[v, r] * W[w, r])
        # Let B_r[v, w] = V[v, r] * W[w, r]
        # Then T[u, :, :] = sum_r U[u, r] B_r
        # This is a linear regression: T_u (flattened) = U_u @ B_flat
        # Where B_flat is a matrix of size rank x (m*p * p*n)
        # B_flat[r, idx] = B_r[idx]
        
        # Construct B_flat
        # B_r is size (m*p, p*n)
        # We can compute this for all r at once?
        # V is (m*p, rank), W is (p*n, rank)
        # B_r[v, w] = V[v, r] * W[w, r]
        # This is the element-wise product of columns of V and W.
        # We can form a tensor B of size (m*p, p*n, rank)
        # B[v, w, r] = V[v, r] * W[w, r]
        # Then T[u, v, w] = sum_r U[u, r] B[v, w, r]
        # This is a matmul if we reshape.
        # T (n*m, m*p, p*n)
        # B (m*p, p*n, rank)
        # U (n*m, rank)
        # T[u, v, w] = sum_r U[u, r] B[v, w, r]
        # This is not a standard matmul because the sum is over r, but B has v, w as first two dims.
        # We can reshape T to (n*m, (m*p)*(p*n))
        # And B to ((m*p)*(p*n), rank)
        # Then T_flat = U @ B_flat
        # So U = T_flat @ B_flat^+ (pseudo-inverse)
        
        # Let's do this
        B = np.einsum('vr,wr->vwr', V, W) # (m*p, p*n, rank)
        B_flat = B.reshape(m*p*p*n, rank)
        T_flat = T.reshape(n*m, m*p*p*n)
        
        # Solve U = T_flat @ pinv(B_flat)
        U_new = T_flat @ np.linalg.pinv(B_flat)
        
        # Similarly for V
        # T[u, v, w] = sum_r V[v, r] (U[u, r] * W[w, r])
        # C_r[u, w] = U[u, r] * W[w, r]
        # T[:, v, :] = sum_r V[v, r] C_r
        # C is (n*m, p*n, rank)
        C = np.einsum('ur,wr->uwr', U, W)
        C_flat = C.reshape(p*n*n*m, rank) # Note: T is (n*m, m*p, p*n). For V, we fix v.
        # T[:, v, :] has shape (n*m, p*n)
        # So we need to reshape T to (m*p, n*m*p*n)
        T_v_flat = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
        # C_flat should match this shape
        # C is (n*m, p*n, rank). Reshape to (n*m*p*n, rank)
        C_flat = C.reshape(n*m*p*n, rank)
        V_new = T_v_flat @ np.linalg.pinv(C_flat)
        
        # Similarly for W
        # T[u, v, w] = sum_r W[w, r] (U[u, r] * V[v, r])
        # D_r[u, v] = U[u, r] * V[v, r]
        # T[:, :, w] = sum_r W[w, r] D_r
        # D is (n*m, m*p, rank)
        D = np.einsum('ur,vr->uvr', U, V)
        D_flat = D.reshape(n*m*m*p, rank)
        T_w_flat = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
        W_new = T_w_flat @ np.linalg.pinv(D_flat)
        
        U = U_new
        V = V_new
        W = W_new
        
        # Check convergence / validity
        if _verify_decomposition_full(U, V, W, n, m, p):
            # Round to nearest 0.5
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            if _verify_decomposition_full(U_round, V_round, W_round, n, m, p):
                return U_round.tolist(), V_round.tolist(), W_round.tolist()
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    def _schoolbook():
        rank = n*m*p
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
        return U.tolist(), V.tolist(), W.tolist()
    
    best_result = _schoolbook()
    
    for rank in range(35, 20, -1):
        if time.time() - start_time > 120:
            break
        time_per_rank = max(1, (120 - (time.time() - start_time)) / (35 - rank))
        
        for restart in range(5):
            if time.time() - start_time > 120:
                break
            result = _alternating_minimization(rank, n, m, p, max_iters=50, time_limit=time_per_rank/5)
            if result is not None:
                best_result = result
                break
        
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
