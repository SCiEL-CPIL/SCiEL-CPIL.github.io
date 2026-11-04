import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T as a 3D array of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    def check_decomposition(U, V, W, tol=1e-6):
        r = U.shape[1]
        result = np.zeros((nm, mp, pn))
        for c in range(r):
            result += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        err = np.max(np.abs(result - T))
        return err < tol

    def als_step(U, V, W, r, max_iters=100, tol=1e-10):
        """Alternating Least Squares to refine U, V, W."""
        # Precompute flattened dimensions
        mp_pn = mp * pn
        nm_mp = nm * mp
        nm_pn = nm * pn
        
        for it in range(max_iters):
            if time.time() > deadline:
                return False
            
            # Update U
            # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
            # For fixed V, W, we want to solve for U.
            # T_flat[a, idx] = sum_k U[a, k] M_VW[idx, k]
            # where idx = b * pn + c, M_VW[idx, k] = V[b, k] * W[c, k]
            
            # Construct M_VW (mp*pn, r)
            M_VW = np.zeros((mp_pn, r))
            for b in range(mp):
                for c in range(pn):
                    M_VW[b*pn+c, :] = V[b, :] * W[c, :]
            
            # Solve U[a, :] = T[a, :].flatten() @ pinv(M_VW.T)
            # M_VW.T is (r, mp*pn)
            # We need U (nm, r) such that U @ M_VW.T = T_flat (nm, mp*pn)
            # U = T_flat @ (M_VW.T)^+
            
            T_flat = T.reshape(nm, mp_pn)
            # Use least squares
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)
                U = U_new.T
            except:
                return False
                
            if time.time() > deadline:
                return False

            # Update V
            # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
            # For fixed U, W, solve for V.
            # T_flat[b, idx] = sum_k V[b, k] M_UW[idx, k]
            # where idx = a * pn + c, M_UW[idx, k] = U[a, k] * W[c, k]
            
            M_UW = np.zeros((nm_pn, r))
            for a in range(nm):
                for c in range(pn):
                    M_UW[a*pn+c, :] = U[a, :] * W[c, :]
            
            T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm_pn)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_UW.T, T_flat_V.T, rcond=None)
                V = V_new.T
            except:
                return False

            if time.time() > deadline:
                return False

            # Update W
            # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
            # For fixed U, V, solve for W.
            # T_flat[c, idx] = sum_k W[c, k] M_UV[idx, k]
            # where idx = a * mp + b, M_UV[idx, k] = U[a, k] * V[b, k]
            
            M_UV = np.zeros((nm_mp, r))
            for a in range(nm):
                for b in range(mp):
                    M_UV[a*mp+b, :] = U[a, :] * V[b, :]
            
            T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm_mp)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_UV.T, T_flat_W.T, rcond=None)
                W = W_new.T
            except:
                return False

            # Check convergence
            # err = np.linalg.norm(T - np.einsum('ak,bk,ck->abc', U, V, W))
            # if err < tol: break

        return True

    def try_rank(r, n_restarts=10):
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None, None, None, False
            
            np.random.seed(restart * 100 + r)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Scale down to avoid overflow/instability initially?
            # ALS is scale invariant in the product, but individual terms can grow.
            # Let's just run ALS.
            
            success = als_step(U, V, W, r, max_iters=50)
            if not success:
                continue
                
            # Check decomposition
            if check_decomposition(U, V, W, tol=1e-5):
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if check_decomposition(U_r, V_r, W_r, tol=1e-9):
                    return U_r, V_r, W_r, True
                # If rounding fails, maybe we can just return the float? 
                # The task says "integer/half-integer entries".
                # So we MUST return half-integers.
                # If we can't round, this decomposition is invalid for the score.
                pass
                
        return None, None, None, False

    # Try ranks from low to high
    # We know 72 is valid. Try 71, 70, ...
    # It is unlikely we will find anything, but we must try.
    for r in range(20, best_rank):
        if time.time() > deadline:
            break
        U, V, W, success = try_rank(r, n_restarts=5)
        if success:
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
