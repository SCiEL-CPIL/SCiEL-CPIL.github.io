import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]

def get_target_tensor():
    n, m, p = DIMENSIONS
    # U dimension: n*m = 4
    # V dimension: m*p = 6
    # W dimension: p*n = 6
    
    # Tensor T of shape (4, 6, 6)
    # T[u, v, w] is 1 if A_u * B_v contributes to C_w
    # A index u: i * m + k (0-based) -> i in 0..1, k in 0..1
    # B index v: k * p + j (0-based) -> k in 0..1, j in 0..2
    # C index w: i + j*n (from prompt W[k*n+i,r] implies col-major indexing for C in W matrix, 
    # but here W is a factor vector. The prompt says W[k*n+i,r]. 
    # In the seed code: 
    # w = i + j * n. 
    # Let's verify: 
    # C is 2x3. 
    # i=0, j=0 -> w=0. C[0,0]
    # i=1, j=0 -> w=1. C[1,0]
    # i=0, j=1 -> w=2. C[0,1]
    # i=1, j=1 -> w=3. C[1,1]
    # i=0, j=2 -> w=4. C[0,2]
    # i=1, j=2 -> w=5. C[1,2]
    # This matches w = i + j*n.
    
    T = np.zeros((n*m, m*p, p*n))
    
    for i in range(n):
        for j in range(p):
            w = i + j * n
            for k in range(m):
                u = i * m + k
                v = k * p + j
                T[u, v, w] += 1
                
    return T

def compute_error(U, V, W, T):
    rec = np.einsum('ur, vr, wr -> uvw', U, V, W)
    diff = T - rec
    return np.sum(diff**2)

def construct():
    n, m, p = DIMENSIONS
    T = get_target_tensor()
    
    # Default Rank 11 Solution
    def get_rank_11():
        rank = 11
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Col 0: m1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[4,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # Col 1: m2
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1
        
        # Col 2: m3
        U[0,2] = 1
        V[1,2] = 1; V[4,2] = -1
        W[2,2] = 1; W[3,2] = 1
        
        # Col 3: m4
        U[3,3] = 1
        V[3,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1
        
        # Col 4: m5
        U[0,4] = 1; U[1,4] = 1
        V[4,4] = 1
        W[0,4] = -1; W[2,4] = 1
        
        # Col 5: m6
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # Col 6: m7
        U[1,6] = 1; U[3,6] = -1
        V[3,6] = 1; V[4,6] = 1
        W[0,6] = 1
        
        # Col 7: a11*b13
        U[0,7] = 1
        V[2,7] = 1
        W[4,7] = 1
        
        # Col 8: a12*b23
        U[1,8] = 1
        V[5,8] = 1
        W[4,8] = 1
        
        # Col 9: a21*b13
        U[2,9] = 1
        V[2,9] = 1
        W[5,9] = 1
        
        # Col 10: a22*b23
        U[3,10] = 1
        V[5,10] = 1
        W[5,10] = 1
        
        return U, V, W

    U_def, V_def, W_def = get_rank_11()
    
    start_time = time.time()
    time_limit = 145.0
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Try to find Rank 10
    target_rank = 10
    
    # ALS Solver
    def als_decompose(T, rank, max_iters=100, tol=1e-12):
        # Random init
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        prev_err = float('inf')
        for it in range(max_iters):
            # Solve for U: fix V, W
            # T_uvw = sum_r U_ur V_vr W_wr
            # Reshape T to (nm, mp*pn)
            # T_mat_u = T.reshape(nm, mp*pn)
            # RHS_u = T_mat_u @ (V * W).T ? 
            # Let's use einsum logic.
            # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
            # For fixed V, W, this is linear in U.
            # T_u = sum_r U_ur (V_vr W_wr)
            # Let M_u_r = sum_{v,w} T[u,v,w] V[v,r] W[w,r]
            # This is not quite right. 
            # T is a tensor. 
            # We want to minimize || T - sum_r U_r V_r^T W_r^T ||_F^2
            # This is equivalent to minimizing || T_unfold_u - U (V^T W^T) ||_F^2 ? No.
            # 
            # Standard ALS for CP:
            # T = sum_r U_r \otimes V_r \otimes W_r
            # T_unfold_u (shape nm x mp*pn) = U (V \odot W)^T
            # Where \odot is Khatri-Rao product (column-wise Kronecker).
            # V \odot W has shape (mp*pn) x rank.
            # Column r is V[:, r] \otimes W[:, r].
            
            # Construct Khatri-Rao product of V and W
            # V: (mp, rank), W: (pn, rank)
            # V_kron_W: (mp*pn, rank)
            # Row index is v*pn + w. Column r is V[v, r] * W[w, r].
            
            # We can compute this efficiently.
            # T_unfold_u: (nm, mp*pn)
            # T_unfold_u = U @ (V_kron_W).T
            
            # Let's compute T_unfold_u
            T_u = T.reshape(nm, mp * pn)
            
            # Compute V_kron_W
            # We can do this by broadcasting
            # V_col = V[:, r] (mp, 1)
            # W_col = W[:, r] (pn, 1)
            # Kronecker is (mp*pn, 1)
            # Doing this for all r is expensive if done naively.
            # But rank is small (10). mp=6, pn=6. mp*pn=36.
            # So V_kron_W is 36x10.
            
            V_kron_W = np.zeros((mp * pn, rank))
            for r in range(rank):
                V_kron_W[:, r] = np.kron(V[:, r], W[:, r])
            
            # Solve U = T_u @ V_kron_W.T @ inv(V_kron_W @ V_kron_W.T)
            # Or use lstsq
            try:
                U_new, _, _, _ = np.linalg.lstsq(V_kron_W.T, T_u.T, rcond=None)
                U = U_new.T
            except np.linalg.LinAlgError:
                continue
                
            # Solve for V
            T_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            # U_kron_W
            U_kron_W = np.zeros((nm * pn, rank))
            for r in range(rank):
                U_kron_W[:, r] = np.kron(U[:, r], W[:, r])
            try:
                V_new, _, _, _ = np.linalg.lstsq(U_kron_W.T, T_v.T, rcond=None)
                V = V_new.T
            except np.linalg.LinAlgError:
                continue
                
            # Solve for W
            T_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            # U_kron_V
            U_kron_V = np.zeros((nm * mp, rank))
            for r in range(rank):
                U_kron_V[:, r] = np.kron(U[:, r], V[:, r])
            try:
                W_new, _, _, _ = np.linalg.lstsq(U_kron_V.T, T_w.T, rcond=None)
                W = W_new.T
            except np.linalg.LinAlgError:
                continue
                
            err = compute_error(U, V, W, T)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W, err

    best_rank = 11
    best_U, best_V, best_W = U_def, V_def, W_def
    
    # Try random restarts for Rank 10
    # ALS is fast, so we can do many restarts
    max_restarts = 500 
    
    for r in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random init with some structure? 
        # Sometimes starting close to integers helps, but random is fine for global search.
        U, V, W, err = als_decompose(T, target_rank, max_iters=20)
        
        if err < 1e-8:
            # Snap to half-integers
            U_snapped = np.round(U * 2) / 2.0
            V_snapped = np.round(V * 2) / 2.0
            W_snapped = np.round(W * 2) / 2.0
            
            if compute_error(U_snapped, V_snapped, W_snapped, T) < 1e-9:
                return {
                    "U": U_snapped.tolist(),
                    "V": V_snapped.tolist(),
                    "W": W_snapped.tolist()
                }
        
        # Also check if raw solution is close to integers/half-integers
        # Sometimes ALS converges to exact solution without snapping needed if init is good?
        # But problem requires integer/half-integer.
        
    # If Rank 10 failed, return Rank 11
    return {
        "U": U_def.tolist(),
        "V": V_def.tolist(),
        "W": W_def.tolist()
    }
