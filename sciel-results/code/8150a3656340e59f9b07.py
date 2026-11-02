import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor: T[a, b, c] = 1 if indices match, else 0
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    def residual(U, V, W):
        R = U.shape[1]
        T_comp = np.zeros((nm, mp, pn))
        for r in range(R):
            T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return T_comp - T
    
    def check_valid(U, V, W, tol=1e-8):
        res = residual(U, V, W)
        return np.max(np.abs(res)) < tol
    
    def snap_to_halfint(U, V, W):
        """Snap values to nearest 0.5 and verify."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if check_valid(U_s, V_s, W_s, tol=1e-9):
            return U_s, V_s, W_s, True
        # Try rounding to integers first
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check_valid(U_i, V_i, W_i, tol=1e-9):
            return U_i, V_i, W_i, True
        return U, V, W, False
    
    def try_rank_als(R, max_iters=500, tol=1e-10):
        """Try to find a rank-R decomposition using ALS."""
        if time.time() > deadline:
            return None
        
        # Initialize: use schoolbook pattern truncated to R columns
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < R:
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
        
        # For columns beyond the schoolbook pattern, add small random values
        rng = np.random.RandomState(R * 1000)
        for r in range(col, R):
            U[:, r] = rng.randn(nm) * 0.01
            V[:, r] = rng.randn(mp) * 0.01
            W[:, r] = rng.randn(pn) * 0.01
        
        # ALS iterations
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Fix V, W; solve for U
            # For each a: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # This is: (V * W^T) flattened gives a linear system
            # A[(b,c), r] = V[b,r] * W[c,r]
            A = np.zeros((mp * pn, R))
            for b in range(mp):
                for c in range(pn):
                    A[b * pn + c, :] = V[b, :] * W[c, :]
            
            for a in range(nm):
                t_a = T[a, :, :].reshape(-1)
                U[a, :], _, _, _ = np.linalg.lstsq(A, t_a, rcond=None)
            
            # Fix U, W; solve for V
            A = np.zeros((nm * pn, R))
            for a in range(nm):
                for c in range(pn):
                    A[a * pn + c, :] = U[a, :] * W[c, :]
            
            for b in range(mp):
                t_b = T[:, b, :].reshape(-1)
                V[b, :], _, _, _ = np.linalg.lstsq(A, t_b, rcond=None)
            
            # Fix U, V; solve for W
            A = np.zeros((nm * mp, R))
            for a in range(nm):
                for b in range(mp):
                    A[a * mp + b, :] = U[a, :] * V[b, :]
            
            for c in range(pn):
                t_c = T[:, :, c].reshape(-1)
                W[c, :], _, _, _ = np.linalg.lstsq(A, t_c, rcond=None)
            
            # Check convergence every 20 iterations
            if it % 20 == 0:
                res = residual(U, V, W)
                max_res = np.max(np.abs(res))
                if max_res < tol:
                    U_s, V_s, W_s, valid = snap_to_halfint(U, V, W)
                    if valid:
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                    # Even without snapping, if residual is tiny, try to use as-is
                    # but we need half-integers, so let's continue
        
        # Final check
        if check_valid(U, V, W, tol=1e-8):
            U_s, V_s, W_s, valid = snap_to_halfint(U, V, W)
            if valid:
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        return None
    
    # Try ranks from 22 upward (theoretical minimum for 2x3*3x4 is around 20-22)
    for R in [22, 23, 24]:
        if time.time() > deadline:
            break
        
        result = try_rank_als(R)
        if result is not None:
            return result
    
    # Fallback to schoolbook rank 24
    R = n * m * p
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
