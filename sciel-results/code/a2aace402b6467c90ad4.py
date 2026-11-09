import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T as a 3D array of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    def try_rank(r):
        if r >= rank_schoolbook:
            return None
            
        # Initialize U, V, W randomly
        # Small integers to help with convergence to integer solutions?
        # Or just random floats.
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Alternating Least Squares
        # We want to minimize || T - sum_c U_c V_c W_c ||_F
        # This is equivalent to minimizing || T - U (V^T \otimes I) W^T ||? No.
        # T is 3D. Flatten T to 2D?
        # T[a, b, c] = sum_c U[a, c] V[b, c] W[c, c] (index c reused, bad)
        # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        
        # For fixed V, W, the equation for U[a, k] is:
        # T[a, b, c] = sum_k U[a, k] (V[b, k] W[c, k])
        # This is a linear system for U[:, k] for each k? No, U[a, k] is coupled with V[b, k] W[c, k].
        # Actually, for a fixed k, U[:, k] is a vector of length n*m.
        # The term is U[:, k] (V[:, k] \otimes W[:, k])^T?
        # Let's reshape T to (n*m, m*p*p*n)? No.
        
        # Better:
        # T is shape (A, B, C) with A=n*m, B=m*p, C=p*n.
        # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k].
        
        # Fix V, W. Solve for U.
        # For each a, T[a, :, :] = sum_k U[a, k] (V[:, k] W[:, k])^T?
        # V[:, k] is shape (B,). W[:, k] is shape (C,).
        # V[:, k] \otimes W[:, k] is shape (B, C).
        # So T[a, b, c] = sum_k U[a, k] (V[:, k] \otimes W[:, k])[b, c].
        # This is a linear system for U[a, :] given T[a, :, :] and the matrices M_k = V[:, k] \otimes W[:, k].
        # We can solve this for all a simultaneously?
        # Let M be a matrix of shape (B*C, r) where M[:, k] = (V[:, k] \otimes W[:, k]).flatten().
        # Let T_flat be T reshaped to (A, B*C).
        # Then T_flat[a, :] = U[a, :] M.
        # So U = T_flat M^T (M M^T)^{-1}?
        # We can use np.linalg.lstsq.
        
        # Similar for V and W.
        
        for iter in range(50):
            if time.time() > deadline:
                return None
            
            # Solve for U
            # M_VW shape (B*C, r)
            M_VW = np.zeros((m * p * p * n, r))
            for k in range(r):
                # V[:, k] is (m*p,), W[:, k] is (p*n,)
                # Outer product is (m*p, p*n) -> flatten to (m*p*p*n,)
                M_VW[:, k] = np.outer(V[:, k], W[:, k]).flatten()
            
            T_flat = T.reshape(n * m, -1)
            # U (n*m, r) such that U @ M_VW.T == T_flat
            # T_flat.T (B*C, A) = M_VW.T (r, B*C) @ U.T (r, A)?
            # T_flat = U @ M_VW.T
            # U = T_flat @ pinv(M_VW.T)
            U, _, _, _ = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)
            U = U.T
            
            # Solve for V
            # M_UW shape (A*C, r) where M_UW[:, k] = (U[:, k] \otimes W[:, k]).flatten()
            M_UW = np.zeros((n * m * p * n, r))
            for k in range(r):
                M_UW[:, k] = np.outer(U[:, k], W[:, k]).flatten()
            
            # T is (A, B, C). We need T[b, a, c] = sum_k V[b, k] (U[:, k] \otimes W[:, k])[a, c]
            # Reshape T to (B, A*C)
            T_perm = np.transpose(T, (1, 0, 2)).reshape(m * p, -1)
            V, _, _, _ = np.linalg.lstsq(M_UW.T, T_perm.T, rcond=None)
            V = V.T
            
            # Solve for W
            # M_UV shape (A*B, r) where M_UV[:, k] = (U[:, k] \otimes V[:, k]).flatten()
            M_UV = np.zeros((n * m * m * p, r))
            for k in range(r):
                M_UV[:, k] = np.outer(U[:, k], V[:, k]).flatten()
            
            # T is (A, B, C). We need T[c, a, b] = sum_k W[c, k] (U[:, k] \otimes V[:, k])[a, b]
            # Reshape T to (C, A*B)
            T_perm2 = np.transpose(T, (2, 0, 1)).reshape(p * n, -1)
            W, _, _, _ = np.linalg.lstsq(M_UV.T, T_perm2.T, rcond=None)
            W = W.T
            
            # Check error
            error = 0.0
            for k in range(r):
                error += np.sum((np.outer(np.outer(U[:, k], V[:, k]), W[:, k]) - T)**2)
            if error < 1e-6:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness
                err_snap = 0.0
                for k in range(r):
                    err_snap += np.sum((np.outer(np.outer(U_snap[:, k], V_snap[:, k]), W_snap[:, k]) - T)**2)
                if err_snap < 1e-9:
                    return U_snap, V_snap, W_snap
        
        return None

    # Try ranks from 40 up to 49
    best_result = None
    for r in range(40, 50):
        if time.time() > deadline - 10:
            break
        res = try_rank(r)
        if res is not None:
            U, V, W = res
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break # Found a valid one, return it (lowest rank tried first)
            
    if best_result is None:
        # Fallback to schoolbook
        rank = rank_schoolbook
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return best_result
