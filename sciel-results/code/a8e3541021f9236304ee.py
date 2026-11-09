"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the full multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly reconstruct T."""
        r = U.shape[1]
        recon = np.tensordot(U, V, axes=([1], [1]))  # (nm, mp, r)
        recon = np.tensordot(recon, W, axes=([2], [1]))  # (nm, mp, pn)
        diff = np.abs(recon - T)
        return np.max(diff) < tol
    
    def to_rational_matrix(A, tol=1e-10):
        """Convert to half-integers if possible, else return as-is."""
        B = np.round(A * 2) / 2
        if np.max(np.abs(B - A)) < tol:
            return B
        B = np.round(A)
        if np.max(np.abs(B - A)) < tol:
            return B
        return A
    
    def als_decompose(T, rank, max_iter=200, tol=1e-12, seed=42):
        """Alternating least squares for tensor decomposition."""
        np.random.seed(seed)
        nm, mp, pn = T.shape
        U = np.random.randn(nm, rank) * 0.01
        V = np.random.randn(mp, rank) * 0.01
        W = np.random.randn(pn, rank) * 0.01
        
        for it in range(max_iter):
            # Fix V, W; solve for U
            # T reshaped: for each (a, b), T[a,b,:] = sum_r U[a,r] * V[b,r] * W[:,r]
            # Let's do it mode by mode.
            # Mode 1: U = (T reshape) * (V ⊗ W)^+
            # T as (nm, mp*pn), for each a: T[a, :] = U[a, :] @ (V ⊗ W).T ... 
            # Actually, let's use the standard ALS formulation.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix V, W: for each a, T[a, :, :] = sum_r U[a,r] (V[:,r] ⊗ W[:,r])
            # So T_reshaped (nm, mp*pn) = U @ M where M[r, b*pn+c] = V[b,r]*W[c,r]
            M = np.outer(V, W).reshape(rank, mp * pn)  # M[r, b*pn+c] = V[b,r]*W[c,r]
            T_flat = T.reshape(nm, mp * pn)
            # U = T_flat @ pinv(M)
            U_new = T_flat @ np.linalg.pinv(M)
            
            # Fix U, W; solve for V
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each b: T[:, b, :] reshaped = (U ⊗ W) @ V[b, :].T ... 
            # T_reshaped (mp, nm*pn): T_flat2[b, a*pn+c] = sum_r V[b,r] * U[a,r]*W[c,r]
            M2 = np.outer(U, W).reshape(rank, nm * pn)  # M2[r, a*pn+c] = U[a,r]*W[c,r]
            T_flat2 = T.reshape(nm, pn).reshape(nm * pn)  # not right
            # T is (nm, mp, pn). Reshape to (mp, nm*pn) by moving axis 1 to front
            T_flat2 = np.transpose(T, (1, 0, 2)).reshape(mp, nm * pn)
            V_new = T_flat2 @ np.linalg.pinv(M2)
            
            # Fix U, V; solve for W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each c: T[:, :, c] = (U ⊗ V) @ W[c, :].T
            # T_reshaped (pn, nm*mp): T_flat3[c, a*mp+b] = sum_r W[c,r] * U[a,r]*V[b,r]
            M3 = np.outer(U, V).reshape(rank, nm * mp)  # M3[r, a*mp+b] = U[a,r]*V[b,r]
            T_flat3 = np.transpose(T, (2, 0, 1)).reshape(pn, nm * mp)
            W_new = T_flat3 @ np.linalg.pinv(M3)
            
            # Check convergence
            if (np.max(np.abs(U_new - U)) < tol and
                np.max(np.abs(V_new - V)) < tol and
                np.max(np.abs(W_new - W)) < tol):
                U, V, W = U_new, V_new, W_new
                break
            
            U, V, W = U_new, V_new, W_new
        
        return U, V, W
    
    # Try ranks from 96 down to 80
    best_rank = nm * mp  # 96
    best_U = None
    best_V = None
    best_W = None
    
    # First, verify schoolbook works
    U_school = np.zeros((nm, nm * mp))
    V_school = np.zeros((mp, nm * mp))
    W_school = np.zeros((pn, nm * mp))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    for rank in range(nm * mp - 1, 79, -1):
        found = False
        for seed in range(10):
            try:
                U, V, W = als_decompose(T, rank, max_iter=100, seed=seed)
                if check_decomposition(U, V, W):
                    # Try to convert to half-integers
                    U_r = to_rational_matrix(U)
                    V_r = to_rational_matrix(V)
                    W_r = to_rational_matrix(W)
                    if check_decomposition(U_r, V_r, W_r):
                        best_rank = rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
            except Exception:
                continue
        if found:
            break
    
    if best_U is not None:
        # Convert to lists
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        U_list = [[float(x) for x in row] for row in U_school]
        V_list = [[float(x) for x in row] for row in V_school]
        W_list = [[float(x) for x in row] for row in W_school]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
