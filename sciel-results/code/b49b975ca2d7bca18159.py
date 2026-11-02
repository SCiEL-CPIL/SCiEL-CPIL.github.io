import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if (i,j) in A and (j,k) in B, else 0
    # Actually, the tensor for matrix multiplication C = A*B is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T is defined such that T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} exists.
    # So T is a 3D array of 1s and 0s.
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Trivial lower bound
    R_min = max(n*m, m*p, p*n)
    # Schoolbook rank
    R_school = n*m*p
    
    # We will try ranks from R_min to R_school
    # To save time, we might jump, but let's try sequentially for a bit
    # The deadline is 150s.
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_result = None
    best_rank = R_school
    
    # Pre-compute schoolbook solution as fallback
    U_school = np.zeros((n*m, R_school))
    V_school = np.zeros((m*p, R_school))
    W_school = np.zeros((p*n, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = R_school
    
    def snap_to_half_integers(mat):
        # Multiply by 2, round to nearest integer, divide by 2
        return (np.round(mat * 2) / 2.0)
        
    def check_exact(U, V, W):
        # Reconstruct tensor and compare to T
        # T_recon[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is a tensor contraction.
        # We can do this efficiently with np.einsum
        # U is (nm, R), V is (mp, R), W is (pn, R)
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        # Contract over R
        # T_recon[i, j, k] = sum_r U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
        # Note: W index is k*n+i. So W_r[k, i, r] is correct.
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T_recon, T, atol=1e-9)

    def try_rank(R, max_restarts=50, max_iters=100):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            # Use random signs or small random values
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Normalize columns to prevent scaling issues
            # (Optional, but helps stability)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                # For each i in range(n):
                # T[i, :, :] = sum_c U[i*m+j, c] * (V[:, j, c] * W[:, k, c]) ? No.
                # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # Let's fix i.
                # T[i, j, k] = sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c])
                # This is not linear in U because V and W are multiplied.
                # Wait, the ALS update for U is:
                # For a fixed i, we want to find U_i (vector of length R for fixed i? No, U has rows i*m+j).
                # The equation is:
                # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # This is linear in U if we group by j? No.
                # Let's look at the structure again.
                # T is (n, m, p).
                # U is (n*m, R).
                # If we fix V and W, the equation is linear in the entries of U.
                # Let's flatten T to (n, m*p).
                # T_flat[i, j*p+k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # This is still coupled.
                
                # Correct ALS step for U:
                # We want to minimize || T - sum_c U_c \otimes V_c \otimes W_c ||_F^2
                # This is linear in U if V and W are fixed.
                # Let M_c be the vector (V_c \otimes W_c) of length m*p.
                # But W_c depends on i! W_{k*n+i, c}.
                # So for each i, the "basis" vectors are different.
                # For fixed i, let W_i_c be the vector W[k*n+i, c] for k=0..p-1.
                # Then T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W_i_c[k]
                # This is not a simple outer product.
                
                # Alternative view:
                # Fix V and W.
                # For each i, we have a system for U_i (which has m*R entries).
                # T[i, :, :] (size m x p) = sum_c U[i*m+j, c] * (V[:, j, c] * W[:, k, c])?
                # No, V and W are indexed by j and k.
                # Let's define a matrix A_i of size (m*p) x R.
                # A_i[j*p+k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then T[i, j, k] = sum_c U[i*m+j, c] * A_i[j*p+k, c] ??
                # No, U is indexed by i*m+j.
                # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # This is NOT linear in U[i*m+j, c] alone because V and W are also indexed by c.
                # Wait, if we fix V and W, then for a specific i, j, k, the term is U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]).
                # This is a sum over c.
                # So T[i, j, k] = sum_c U[i*m+j, c] * B_{i, j, k, c} where B_{i, j, k, c} = V[j*p+k, c] * W[k*n+i, c].
                # This is linear in the vector U_{i, *}?
                # Let U_i be the submatrix of U corresponding to row i (size m x R).
                # Then T[i, j, k] = sum_c U_i[j, c] * B_{i, j, k, c}.
                # This is a linear system for the entries of U_i.
                # The unknowns are U_i[j, c] for j=0..m-1, c=0..R-1. Total m*R unknowns.
                # The equations are for each j, k. Total m*p equations.
                # Since m*p (20) > m*R (4R) usually?
                # If R=20, m*R=80. 20 < 80. Underdetermined.
                # We can use least squares.
                
                # Construct the system for U_i:
                # Flatten U_i to length m*R.
                # Flatten T[i] to length m*p.
                # The matrix A_i has size (m*p) x (m*R).
                # A_i[j*p+k, j*R+c] = V[j*p+k, c] * W[k*n+i, c]
                # Note: A_i depends on j in the row index and column index block.
                # This is a block structure.
                
                # Let's build A_i for each i.
                for i in range(n):
                    # Build A_i of shape (m*p, m*R)
                    A_i = np.zeros((m*p, m*R))
                    for j in range(m):
                        for k in range(p):
                            row = j*p + k
                            for c in range(R):
                                col = j*R + c
                                A_i[row, col] = V[j*p+k, c] * W[k*n+i, c]
                    
                    # Solve A_i * U_i_flat = T[i, :, :].flatten()
                    T_i_flat = T[i, :, :].flatten()
                    U_i_flat, _, _, _ = np.linalg.lstsq(A_i, T_i_flat, rcond=None)
                    U[i*m:(i+1)*m, :] = U_i_flat.reshape(m, R)
                
                # Update V
                # Symmetric logic.
                # For fixed j, solve for V_j (size p x R).
                # T[i, j, k] = sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c])
                # Unknowns: V_j[k, c] for k=0..p-1, c=0..R-1. Total p*R unknowns.
                # Equations: for each i, k. Total n*p equations.
                # n*p = 10. p*R = 5R.
                # If R=20, 10 < 100. Underdetermined.
                
                for j in range(m):
                    A_j = np.zeros((n*p, p*R))
                    for k in range(p):
                        for i in range(n):
                            row = i*p + k
                            for c in range(R):
                                col = k*R + c
                                A_j[row, col] = U[i*m+j, c] * W[k*n+i, c]
                    
                    T_j_flat = T[:, j, :].flatten() # Shape (n, p) -> (n*p)
                    V_j_flat, _, _, _ = np.linalg.lstsq(A_j, T_j_flat, rcond=None)
                    V[j*p:(j+1)*p, :] = V_j_flat.reshape(p, R)
                
                # Update W
                # For fixed k, solve for W_k (size n x R).
                # T[i, j, k] = sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c])
                # Unknowns: W_k[i, c] for i=0..n-1, c=0..R-1. Total n*R unknowns.
                # Equations: for each i, j. Total n*m equations.
                # n*m = 8. n*R = 2R.
                # If R=20, 8 < 40. Underdetermined.
                
                for k in range(p):
                    A_k = np.zeros((n*m, n*R))
                    for i in range(n):
                        for j in range(m):
                            row = i*m + j
                            for c in range(R):
                                col = i*R + c
                                A_k[row, col] = U[i*m+j, c] * V[j*p+k, c]
                    
                    T_k_flat = T[:, :, k].flatten() # Shape (n, m) -> (n*m)
                    W_k_flat, _, _, _ = np.linalg.lstsq(A_k, T_k_flat, rcond=None)
                    W[k*n:(k+1)*n, :] = W_k_flat.reshape(n, R)
                
                # Check convergence
                # Compute error
                # Simple check: if error is small, break
                # We can compute the Frobenius norm of the residual
                # But it's expensive. Let's just check every 10 iterations or at the end.
                
            # After ALS, check if exact
            if check_exact(U, V, W):
                # Snap to half-integers
                U_snap = snap_to_half_integers(U)
                V_snap = snap_to_half_integers(V)
                W_snap = snap_to_half_integers(W)
                if check_exact(U_snap, V_snap, W_snap):
                    return {"U": U_snap, "V": V_snap, "W": W_snap}
                else:
                    # If snapping fails, maybe the solution is exact but not on the grid?
                    # The problem requires integer/half-integer entries.
                    # If the exact solution is not on the grid, it's invalid.
                    # But usually, if it's exact, it should be close to the grid.
                    # We can try to return the unsnapped version if it's exact?
                    # No, the interface requires integer/half-integer.
                    # So we must snap.
                    pass
            
            # Try a few more iterations or restart if not exact
        return None

    # Search for minimal rank
    for R in range(R_min, R_school + 1):
        if time.time() > deadline:
            break
        
        # Number of restarts decreases as R increases?
        # Or keep constant.
        res = try_rank(R, max_restarts=20, max_iters=50)
        if res is not None:
            best_result = res
            best_rank = R
            break # Found minimal rank? 
            # Note: We should verify if a lower rank exists? 
            # We are iterating from R_min upwards, so the first one found is minimal.
            
    return best_result
# EVOLVE-BLOCK-END
