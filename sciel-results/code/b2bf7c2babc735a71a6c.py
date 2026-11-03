import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Function to verify a decomposition
    def verify(U, V, W, atol=1e-9):
        R = U.shape[1]
        # Efficient reconstruction: T_approx = sum_ alpha U[:, alpha] * V[:, alpha] * W[:, alpha]
        # We can do this via einsum or matrix multiplications
        # Let's try a fast check
        # T_approx[i, j, k] = sum_alpha U[i, alpha] V[j, alpha] W[k, alpha]
        # We can compute this as:
        # M1 = U @ V.T  (shape nm x mp)
        # Then T_approx[i, j, k] = sum_alpha M1[i, alpha] W[k, alpha] = (M1 @ W.T)[i, k]? No.
        # Let's stick to einsum for correctness, it's fast enough for small dims.
        # Or: T_approx = np.einsum('ia,ja,ka->ijk', U, V, W)
        try:
            T_approx = np.einsum('ia,ja,ka->ijk', U, V, W, optimize=True)
        except Exception:
            # Fallback
            T_approx = np.zeros_like(T)
            for alpha in range(R):
                T_approx += np.einsum('i,j,k->ijk', U[:, alpha], V[:, alpha], W[:, alpha])
        
        return np.allclose(T_approx, T, atol=atol)

    # Function to generate schoolbook
    def get_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = U_sb.shape[1]
    
    start_time = time.time()
    deadline = 140.0 # 140s limit, 10s margin
    
    # Strategy:
    # 1. Try to find a lower rank decomposition using a structured random search.
    # 2. We can parameterize U, V, W as sparse matrices or use a random initialization and refine.
    # 3. A very effective method for small tensor ranks is to use the "alternating least squares" or similar,
    #    but that requires solving linear systems which might be overkill or slow if not careful.
    # 4. Instead, let's try a simple random search with a fixed structure.
    #    We know that for 3x4x5, the schoolbook rank is 60.
    #    Can we do better?
    #    For 3x3x3, Strassen is 23.
    #    For 3x4x4, we might be able to do better than 48.
    #    For 3x4x5, let's try ranks from 50 down to 40.
    
    # A more robust approach:
    # Fix the rank R.
    # Initialize U, V, W randomly.
    # Iterate:
    #   - Fix V, W, solve for U (linear least squares)
    #   - Fix U, W, solve for V
    #   - Fix U, V, solve for W
    # This is the standard ALS for tensor decomposition.
    
    def als_decompose(T, R, max_iter=50, tol=1e-9):
        """
        Alternating Least Squares for tensor decomposition.
        T is shape (nm, mp, pn)
        Returns U (nm, R), V (mp, R), W (pn, R)
        """
        # Initialize
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(pn, R)
        
        # Flatten T for efficient solving?
        # T[i, j, k] = sum_alpha U[i, alpha] V[j, alpha] W[k, alpha]
        # Fix V, W:
        # T[i, j, k] = sum_alpha U[i, alpha] (V[j, alpha] W[k, alpha])
        # Let M_VW[j, k] = V[j, :] * W[k, :] (element-wise)
        # Then T[i, :, :] = sum_alpha U[i, alpha] M_VW[:, :, alpha]? No.
        # T[i, j, k] = sum_alpha U[i, alpha] V[j, alpha] W[k, alpha]
        # For fixed i, T[i, :, :] is a matrix.
        # T[i, j, k] = sum_alpha U[i, alpha] (V[j, alpha] W[k, alpha])
        # Let's define a matrix B_i of size (mp*pn, R) where B_i[j*p+k, alpha] = V[j, alpha] W[k, alpha]
        # Then T[i, :] (flattened) = B_i @ U[i, :]
        # So U[i, :] = (B_i^T B_i)^{-1} B_i^T T[i, :]
        
        # This can be done for all i at once if we construct a big matrix.
        # Let's try a simpler approach for small R:
        # Just iterate and solve.
        
        for it in range(max_iter):
            # Solve for U
            # Construct B_VW: shape (mp*pn, R)
            # B_VW[j*p+k, alpha] = V[j, alpha] W[k, alpha]
            B_VW = np.einsum('ja,ka->jka', V, W).reshape(mp*pn, R)
            # T_flat: shape (nm, mp*pn)
            T_flat = T.reshape(nm, mp*pn)
            # Solve U: T_flat = U @ B_VW^T
            # U = T_flat @ (B_VW^T)^+
            # Using least squares
            try:
                U, _, _, _ = np.linalg.lstsq(B_VW.T, T_flat.T, rcond=None)
                U = U.T
            except Exception:
                pass
            
            # Solve for V
            # T[i, j, k] = sum_alpha U[i, alpha] V[j, alpha] W[k, alpha]
            # Fix U, W.
            # T[:, j, k] = sum_alpha U[:, alpha] V[j, alpha] W[k, alpha]
            # Let C_UW[i, k] = U[i, :] * W[k, :] (element-wise)
            # T[i, j, k] = sum_alpha C_UW[i, k, alpha] V[j, alpha]
            # For fixed j, T[:, j, :] (shape nm x pn) = V[j, :] @ C_UW^T?
            # T[i, j, k] = sum_alpha V[j, alpha] C_UW[i, k, alpha]
            # Let's flatten T to (mp, nm*pn)
            # T_j = T[:, j, :].flatten()
            # T_j[i*p+n] = sum_alpha V[j, alpha] C_UW[i, k, alpha]
            # C_UW_flat[i*p+k, alpha] = C_UW[i, k, alpha]
            # T_j = C_UW_flat @ V[j, :]
            # So V[j, :] = (C_UW_flat^T C_UW_flat)^{-1} C_UW_flat^T T_j
            C_UW = np.einsum('ia,ka->ika', U, W).reshape(nm*pn, R)
            T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm*pn)
            try:
                V, _, _, _ = np.linalg.lstsq(C_UW.T, T_flat_v.T, rcond=None)
                V = V.T
            except Exception:
                pass
            
            # Solve for W
            # T[i, j, k] = sum_alpha U[i, alpha] V[j, alpha] W[k, alpha]
            # Fix U, V.
            # T[i, j, k] = sum_alpha W[k, alpha] (U[i, alpha] V[j, alpha])
            # Let D_UV[i, j, alpha] = U[i, alpha] V[j, alpha]
            # T[i, j, k] = sum_alpha W[k, alpha] D_UV[i, j, alpha]
            # For fixed k, T[:, :, k] (shape nm x mp) = W[k, :] @ D_UV[:, :, k]^T?
            # T[i, j, k] = sum_alpha W[k, alpha] D_UV[i, j, alpha]
            # Flatten T to (pn, nm*mp)
            # T_k = T[:, :, k].flatten()
            # T_k[i*mp+j] = sum_alpha W[k, alpha] D_UV[i, j, alpha]
            # D_UV_flat[i*mp+j, alpha] = D_UV[i, j, alpha]
            # T_k = D_UV_flat @ W[k, :]
            # So W[k, :] = (D_UV_flat^T D_UV_flat)^{-1} D_UV_flat^T T_k
            D_UV = np.einsum('ia,ja->ija', U, V).reshape(nm*mp, R)
            T_flat_w = T.transpose(2, 0, 1).reshape(pn, nm*mp)
            try:
                W, _, _, _ = np.linalg.lstsq(D_UV.T, T_flat_w.T, rcond=None)
                W = W.T
            except Exception:
                pass
            
            # Check convergence
            if it % 5 == 0:
                # Calculate error
                try:
                    T_approx = np.einsum('ia,ja,ka->ijk', U, V, W, optimize=True)
                    err = np.linalg.norm(T_approx - T)
                    if err < tol:
                        return U, V, W
                except Exception:
                    pass
            
            if time.time() > deadline:
                break
        
        return U, V, W

    # Try ranks from best_rank - 1 down to a reasonable lower bound
    # Lower bound for 3x4x5 is not well known, but let's try down to 40.
    for R in range(best_rank - 1, 39, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Use ALS
            U_c, V_c, W_c = als_decompose(T, R, max_iter=30)
            
            if verify(U_c, V_c, W_c, atol=1e-6):
                # We found a valid decomposition!
                best_U, best_V, best_W = U_c, V_c, W_c
                best_rank = R
                break # Move to next R
        
        # If we found a valid decomposition for R, we can stop trying lower R for now
        # or continue to see if we can go even lower.
        # Let's continue to try lower R.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
