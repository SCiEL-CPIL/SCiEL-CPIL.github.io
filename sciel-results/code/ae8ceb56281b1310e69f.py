import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _generate_tensor(n, m, p):
    """Generate the flattened matrix multiplication tensor T of shape (n*m*p,).
    T[index] = 1 if index corresponds to (i,j,k) in the multiplication sum.
    Actually, we need the 3-way tensor.
    We will work with the matrix form of the tensor for ALS.
    T is in R^{nm x mp x np}.
    """
    # We don't need to generate the full sparse tensor if we use ALS on the matrix unfolding.
    pass

def _als_decompose(target_U, target_V, target_W, rank, iters=50, tol=1e-9):
    """
    Alternating Least Squares for tensor decomposition.
    target_U, target_V, target_W are not directly available as separate targets.
    We have the target tensor T.
    T is defined by T[i, j, k] = 1 for all i,j,k? No.
    T is the matrix multiplication tensor.
    
    Let's define the target tensor T of shape (n*m, m*p, p*n).
    T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) ... wait.
    w index is k*n + i.
    So T[i*m+j, j*p+k, k*n+i] = 1.
    
    We want U (nm x rank), V (mp x rank), W (pn x rank)
    such that sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w].
    
    We can reshape T to a matrix.
    Let's unfold T along the 3rd mode to get T_(3) of shape (pn, nm*mp).
    T_(3)[w, u*mp + v] = T[u, v, w].
    Then T_(3) = W * (V * U.T)_reshaped?
    sum_r W[w,r] * (V[v,r] * U[u,r])
    Let M = U.T @ V. M has shape (rank, rank)? No.
    M[u,v] = sum_r U[u,r] V[v,r].
    Then sum_r W[w,r] M[u,v] ... no.
    The term is sum_r U[u,r] V[v,r] W[w,r].
    This is element-wise product of U[:,r], V[:,r], W[:,r] summed over r.
    
    ALS updates:
    Fix V, W. Update U.
    For each u, U[u, :] is a vector.
    We want to minimize || sum_r U[u,r] (V[:,r] * W[:,r]) - T_u ||^2
    where T_u is the slice of T at u.
    T_u has shape (mp, pn).
    T_u[v, w] = T[u, v, w].
    Let A_u = V * W.T (element-wise? No).
    The term is sum_r U[u,r] V[v,r] W[w,r].
    This is a linear system for U[u, :].
    Let B_u be a matrix of shape (mp*pn, rank).
    B_u[v*pn + w, r] = V[v,r] * W[w,r].
    Then T_u_flat = B_u @ U[u, :].
    So U[u, :] = B_u^+ @ T_u_flat.
    
    Similarly for V and W.
    """
    n, m, p = 3, 3, 4 # For the sub-problem
    # Generate T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    # Initialize U, V, W
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for it in range(iters):
        # Update U
        for u in range(n*m):
            # B_u shape (m*p * p*n, rank)
            B = np.outer(V.flatten(), W.flatten()) # shape (mp*pn, 1) * (1, pn)? No.
            # B[v*pn + w, r] = V[v,r] * W[w,r]
            B = np.zeros((m*p * p*n, rank))
            for v in range(m*p):
                for w in range(p*n):
                    B[v*p*n + w, :] = V[v, :] * W[w, :]
            # Target T_u flat
            T_u = T[u, :, :].flatten()
            # Solve least squares
            U[u, :] = np.linalg.lstsq(B, T_u, rcond=None)[0]
            
        # Update V
        for v in range(m*p):
            B = np.zeros((n*m * p*n, rank))
            for u in range(n*m):
                for w in range(p*n):
                    B[u*p*n + w, :] = U[u, :] * W[w, :]
            T_v = T[:, v, :].flatten()
            V[v, :] = np.linalg.lstsq(B, T_v, rcond=None)[0]
            
        # Update W
        for w in range(p*n):
            B = np.zeros((n*m * m*p, rank))
            for u in range(n*m):
                for v in range(m*p):
                    B[u*m*p + v, :] = U[u, :] * V[v, :]
            T_w = T[:, :, w].flatten()
            W[w, :] = np.linalg.lstsq(B, T_w, rcond=None)[0]
            
        # Check error
        # err = 0
        # for r in range(rank):
        #     err += np.sum((U[:,r] * V[:,r] * W[:,r]) - T)**2 # This is expensive
        # pass
        
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the tensor."""
    rank = U.shape[1]
    # Reconstruct T
    T_recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    # Target T
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_target[u, v, w] = 1.0
                
    # Check difference
    diff = T_recon - T_target
    max_diff = np.max(np.abs(diff))
    return max_diff < 1e-9

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    # p=8. Split into two blocks of p=4.
    # Find rank-20 decomposition for 3x3x4.
    
    # Try to find a solution for 3x3x4 with rank 20.
    U_sub, V_sub, W_sub = _als_decompose(3, 3, 4, 20, iters=20)
    
    # Round to half-integers
    U_sub = np.round(U_sub * 2) / 2
    V_sub = np.round(V_sub * 2) / 2
    W_sub = np.round(W_sub * 2) / 2
    
    # Validate
    if _validate(U_sub, V_sub, W_sub, 3, 3, 4):
        # Construct full 3x3x8
        rank = 40
        U_full = np.zeros((9, 40))
        V_full = np.zeros((24, 40))
        W_full = np.zeros((24, 40))
        
        # Block 1 (k=0..3)
        U_full[:, 0:20] = U_sub
        # V indices: j*4 + k. k=0..3.
        for j in range(3):
            for k in range(4):
                v_idx = j*4 + k
                V_full[v_idx, 0:20] = V_sub[j*4 + k, :]
        # W indices: k*3 + i. k=0..3.
        for k in range(4):
            for i in range(3):
                w_idx = k*3 + i
                W_full[w_idx, 0:20] = W_sub[k*3 + i, :]
                
        # Block 2 (k=4..7)
        U_full[:, 20:40] = U_sub
        # V indices: j*4 + k. k=4..7.
        # In the sub-problem, k is 0..3.
        # So for k=4, we use sub-k=0.
        for j in range(3):
            for k_sub in range(4):
                k_full = 4 + k_sub
                v_idx = j*4 + k_full
                V_full[v_idx, 20:40] = V_sub[j*4 + k_sub, :]
        # W indices: k*3 + i. k=4..7.
        for k_sub in range(4):
            k_full = 4 + k_sub
            for i in range(3):
                w_idx = k_full*3 + i
                W_full[w_idx, 20:40] = W_sub[k_sub*3 + i, :]
                
        # Convert to lists
        U_list = U_full.tolist()
        V_list = V_full.tolist()
        W_list = W_full.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
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
