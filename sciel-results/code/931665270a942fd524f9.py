"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the multiplication tensor T[i,j,k] = delta[j,k]
    # U shape: (nm, rank), V shape: (mp, rank), W shape: (pn, rank)
    # T[i, j, k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # where i = i_a*m + j_a (row of A, col of A), 
    #       j = j_a*p + k_a (row of B, col of B), 
    #       k = k_a*n + i_a (row of C, col of C)
    # T[i,j,k] = 1 if j_a == k_a and i_a == i_a (always), so T[i,j,k] = 1 if j_a==k_a
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i_a in range(n):
        for j_a in range(m):
            for k_a in range(p):
                if j_a == k_a:
                    T[i_a * m + j_a, j_a * p + k_a, k_a * n + i_a] = 1.0
    
    def try_rank(r, seed=0):
        """Try to find a rank-r decomposition."""
        np.random.seed(seed)
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each i, U[i,:] should satisfy sum_c U[i,c]*V[j,c]*W[k,c] = T[i,j,k]
            # Fix V,W: U[i,:] * (V[j,:] * W[k,:]) = T[i,j,k] for all j,k
            # This is a linear system: let M[i,j,k] = V[j,c]*W[k,c], reshape to (nm*mp, r)
            # Actually: for fixed i, we need U[i,:] such that for all (j,k): sum_c U[i,c] * (V[j,c]*W[k,c]) = T[i,j,k]
            # Let A_mat[j,k,c] = V[j,c]*W[k,c], reshape to (mp*pn, r)
            # Then A_mat @ U[i,:] = T[i,:,:].ravel()
            
            # Build the linear system matrix
            # For efficiency, compute V * W outer product structure
            # M[c, j, k] = V[j,c] * W[k,c]
            # Reshape to (r, mp*pn) then transpose
            
            M = np.einsum('jc,kc->ckj', V, W)  # (r, pn, mp)
            M_flat = M.transpose(1, 2, 0).reshape(mp * pn, r)  # (mp*pn, r)
            
            # Solve for U: M_flat @ U[i,:] = T[i,:,:].ravel()
            # Use least squares
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_flat, T.reshape(nm, mp*pn).T, rcond=None)
                U_new = U_new.T  # (nm, r)
            except:
                break
            
            # Update V
            M2 = np.einsum('ic,kc->cki', U, W)  # (r, pn, nm)
            M2_flat = M2.transpose(2, 1, 0).reshape(nm * pn, r)  # (nm*pn, r)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M2_flat, T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm*pn).T, rcond=None)
                V_new = V_new.T  # (mp, r)
            except:
                break
            
            # Update W
            M3 = np.einsum('ic,jc->cij', U, V)  # (r, nm, mp)
            M3_flat = M3.transpose(2, 1, 0).reshape(mp * nm, r)  # (mp*nm, r)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M3_flat, T.reshape(nm, mp, pn).transpose(2, 1, 0).reshape(pn, mp*nm).T, rcond=None)
                W_new = W_new.T  # (pn, r)
            except:
                break
            
            U, V, W = U_new, V_new, W_new
            
            # Check error
            err = np.sum((np.einsum('ic,jc,kc->ijk', U, V, W) - T)**2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
        
        return best_U, best_V, best_W, best_err
    
    # Try ranks from low to high
    for r in range(20, 32):
        if time.time() > deadline - 10:
            break
        
        best_U, best_V, best_W, best_err = None, None, None, np.inf
        
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            U, V, W, err = try_rank(r, seed)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U, V, W
        
        if best_err < 1e-6:
            # Found a good solution, now round to half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            # Verify
            recon = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
            verify_err = np.sum((recon - T)**2)
            
            if verify_err < 1e-8:
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U_rounded]
                V_list = [[float(x) for x in row] for row in V_rounded]
                W_list = [[float(x) for x in row] for row in W_rounded]
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # Try different rounding
            for offset in [0.0, 0.25, -0.25, 0.5, -0.5]:
                U_r = np.round((best_U + offset) * 2) / 2
                V_r = np.round((best_V + offset) * 2) / 2
                W_r = np.round((best_W + offset) * 2) / 2
                recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                verify_err = np.sum((recon - T)**2)
                if verify_err < 1e-8:
                    U_list = [[float(x) for x in row] for row in U_r]
                    V_list = [[float(x) for x in row] for row in V_r]
                    W_list = [[float(x) for x in row] for row in W_r]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # Also try rounding without offset
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        verify_err = np.sum((recon - T)**2)
        if verify_err < 1e-8:
            U_list = [[float(x) for x in row] for row in U_r]
            V_list = [[float(x) for x in row] for row in V_r]
            W_list = [[float(x) for x in row] for row in W_r]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
