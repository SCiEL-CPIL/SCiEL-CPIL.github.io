import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 12
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    def compute_residual(U, V, W):
        """Compute Frobenius norm of tensor difference."""
        C = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            C += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
        return np.linalg.norm(C - T)
    
    def snap_to_half_int(arr):
        """Snap to nearest 0.5."""
        return np.round(arr * 2) / 2.0
    
    def verify(U, V, W):
        """Check if entries are half-integers and decomposition is exact."""
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-6): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-6): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-6): return False
        
        C = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            C += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
        
        return np.allclose(C, T, atol=1e-6)
    
    def als_step(U, V, W, max_iter=50, tol=1e-8):
        """Alternating least squares to minimize tensor residual."""
        current_res = compute_residual(U, V, W)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: fix V, W, solve linear system for U
            # Minimize ||sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] - T||^2
            # Reshape T to (nm, mp*pn)
            T_reshaped = T.reshape(n*m, m*p*p*n)
            
            # Build the linear system: for each c, the contribution is U[:,c] * (V[:,c] ⊗ W[:,c])^T
            # So we need: sum_c U[:,c] * (V[:,c] ⊗ W[:,c])^T = T_reshaped
            # Let A be a matrix where A[i, c*mp*pn + j] = V[j,c] * W[k,c] for appropriate k
            # Actually, let's think differently.
            # C[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
            # For fixed V,W, this is linear in U.
            # C_reshaped[u, (v,w)] = sum_c U[u,c] * (V[v,c]*W[w,c])
            # So C_reshaped = U @ B where B[u,c] = V[v,c]*W[w,c]... no.
            # C_reshaped[u, idx] where idx = v*(pn) + w
            # C_reshaped[u, v*pn+w] = sum_c U[u,c] * V[v,c] * W[w,c]
            # Let M[c, v, w] = V[v,c] * W[w,c]
            # Then C_reshaped = U @ (M reshaped to (R, mp*pn))^T
            # So U = (M^T M)^{-1} M^T T_reshaped (least squares)
            
            # Build M: shape (R, mp, pn)
            M = np.zeros((target_rank, m*p, p*n))
            for c in range(target_rank):
                for v in range(m*p):
                    for w in range(p*n):
                        M[c, v, w] = V[v, c] * W[w, c]
            
            M_2d = M.reshape(target_rank, m*p*p*n)
            
            # Solve U @ M_2d^T = T_reshaped
            # U = (M_2d M_2d^T)^{-1} M_2d T_reshaped... 
            # Actually: C_reshaped = U @ M_2d^T
            # So U = T_reshaped @ pinv(M_2d^T)
            try:
                U_new = np.linalg.lstsq(M_2d.T, T_reshaped, rcond=None)[0]
            except:
                break
            
            # Update V: fix U, W
            T_reshaped_v = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
            M_v = np.zeros((target_rank, n*m, p*n))
            for c in range(target_rank):
                for u in range(n*m):
                    for w in range(p*n):
                        M_v[c, u, w] = U[u, c] * W[w, c]
            M_v_2d = M_v.reshape(target_rank, n*m*p*n)
            try:
                V_new = np.linalg.lstsq(M_v_2d.T, T_reshaped_v, rcond=None)[0]
            except:
                break
            
            # Update W: fix U, V
            T_reshaped_w = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
            M_w = np.zeros((target_rank, n*m, m*p))
            for c in range(target_rank):
                for u in range(n*m):
                    for v in range(m*p):
                        M_w[c, u, v] = U[u, c] * V[v, c]
            M_w_2d = M_w.reshape(target_rank, n*m*m*p)
            try:
                W_new = np.linalg.lstsq(M_w_2d.T, T_reshaped_w, rcond=None)[0]
            except:
                break
            
            new_res = compute_residual(U_new, V_new, W_new)
            if abs(new_res - current_res) < tol:
                U, V, W = U_new, V_new, W_new
                break
            U, V, W = U_new, V_new, W_new
            current_res = new_res
        
        return U, V, W, compute_residual(U, V, W)
    
    best_score = float('inf')
    best_solution = None
    
    # Try multiple random initializations with ALS
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Structured initialization: use schoolbook-like structure with random mixing
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Add some structure: initialize based on schoolbook patterns
        for c in range(target_rank):
            # Random subset of indices
            u_idx = np.random.choice(n*m, size=2, replace=False)
            v_idx = np.random.choice(m*p, size=2, replace=False)
            w_idx = np.random.choice(p*n, size=2, replace=False)
            U[u_idx, c] = 1.0
            V[v_idx, c] = 1.0
            W[w_idx, c] = 1.0
        
        U, V, W, res = als_step(U, V, W, max_iter=30)
        
        if res < best_score:
            best_score = res
            # Try snapping
            U_snap = snap_to_half_int(U)
            V_snap = snap_to_half_int(V)
            W_snap = snap_to_half_int(W)
            
            if verify(U_snap, V_snap, W_snap):
                # Convert to list format
                U_list = [[float(x) for x in row] for row in U_snap]
                V_list = [[float(x) for x in row] for row in V_snap]
                W_list = [[float(x) for x in row] for row in W_snap]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Also try without snapping if residual is very small
        if res < 1e-10:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no solution found, return fallback
    if best_solution:
        return best_solution
    else:
        return fallback
