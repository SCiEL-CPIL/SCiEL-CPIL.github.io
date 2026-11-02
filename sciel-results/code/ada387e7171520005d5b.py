"""Matrix multiplication 3x4x7 decomposition - iterative rank reduction."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook_solution(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    nm = n * m
    mp = m * p
    pn = p * n
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Build the reconstructed tensor
    # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j, :]
                vj = V[j * p + k, :]
                wk = W[k * n + i, :]
                val = np.dot(ui, vj * wk)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _snap_to_half_integers(mat, tol=1e-8):
    """Snap values close to half-integers to exact half-integers."""
    mat = np.array(mat, dtype=float)
    snapped = np.round(mat * 2.0) / 2.0
    if np.max(np.abs(mat - snapped)) < tol:
        return snapped
    return mat

def _try_rank_reduction(n, m, p, target_rank, max_iters=200, tol=1e-10, deadline=None):
    """
    Try to find a rank-target_rank decomposition via alternating minimization.
    Initialize with a random subset of schoolbook terms plus perturbation.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    nm = n * m
    mp = m * p
    pn = p * n
    r = target_rank
    
    # Build the target tensor in flattened form
    # We need: sum_{t=0}^{r-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    # This is equivalent to: for each (i,j,k), the dot product equals 1.
    # 
    # Alternate: fix U,V -> solve for W; fix U,W -> solve for V; fix V,W -> solve for U.
    # For fixed U,V: W[k*n+i, t] should satisfy sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    # For fixed U,W: V[j*p+k, t] should satisfy sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    # For fixed V,W: U[i*m+j, t] should satisfy sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    
    # Linear system approach:
    # For fixed U,V: For each (k,i), we need sum_t W[k*n+i,t] * (sum_j U[i*m+j,t]*V[j*p+k,t]) = 1
    # Let A_w[k*n+i, t] = sum_j U[i*m+j,t]*V[j*p+k,t], then W[k*n+i,:] * A_w[k*n+i,:].T ... no
    # Actually: sum_t W[k*n+i, t] * A_w[k*n+i, t] = 1 where A_w[k*n+i, t] = sum_j U[i*m+j,t]*V[j*p+k,t]
    # This is: W_row · A_w_row = 1, which is one equation per (k,i) pair.
    # But W has pn rows and r columns. We have pn equations. If r >= pn, we can solve.
    # Actually we have pn equations (one per (k,i)) and pn*r unknowns. Underdetermined if r > 1.
    # We want the minimum-norm solution or just any solution.
    
    # Better approach: treat it as a linear least squares problem.
    # For fixed U, V:
    #   For each (i,k): sum_j sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    #   Define B[i,k,t] = sum_j U[i*m+j,t]*V[j*p+k,t]
    #   Then: sum_t B[i,k,t] * W[k*n+i,t] = 1
    #   This is pn scalar equations. W is (pn, r).
    #   We can solve: for each (i,k), W[k*n+i,:] is a row of W.
    #   The equation is: W[k*n+i,:] @ B[i,k,:] = 1
    #   So W[k*n+i,:] needs to satisfy a linear constraint.
    #   If r >= 1, we can pick W[k*n+i,:] = B[i,k,:] / ||B[i,k,:]||^2 (if B is nonzero)
    #   But we need this to work simultaneously for all (i,k).
    #   Actually each (i,k) gives an independent equation on W[k*n+i,:], so we can solve each independently!
    
    # Similarly for V:
    #   For fixed U, W:
    #   For each (j,k): sum_i sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    #   Define C[j,k,t] = sum_i U[i*m+j,t]*W[k*n+i,t]
    #   Then: V[j*p+k,:] @ C[j,k,:] = 1
    #   Each (j,k) gives an independent equation on V[j*p+k,:].
    
    # And for U:
    #   For fixed V, W:
    #   For each (i,j): sum_k sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    #   Define D[i,j,t] = sum_k V[j*p+k,t]*W[k*n+i,t]
    #   Then: U[i*m+j,:] @ D[i,j,:] = 1
    #   Each (i,j) gives an independent equation on U[i*m+j,:].
    
    # So the alternating minimization is:
    # 1. Initialize U, V, W randomly (or from schoolbook subset)
    # 2. Update W: for each (i,k), set W[k*n+i,:] = B[i,k,:] / ||B[i,k,:]||^2
    # 3. Update V: for each (j,k), set V[j*p+k,:] = C[j,k,:] / ||C[j,k,:]||^2
    # 4. Update U: for each (i,j), set U[i*m+j,:] = D[i,j,:] / ||D[i,j,:]||^2
    # 5. Check convergence
    
    best_result = None
    best_err = float('inf')
    
    for init in range(3):
        if deadline is not None and time.time() > deadline:
            break
        
        # Initialize from a random perturbation of a subset of schoolbook terms
        if init == 0:
            # Use first r schoolbook terms as initialization
            U, V, W = _schoolbook_solution(n, m, p)
            U = U[:, :r]
            V = V[:, :r]
            W = W[:, :r]
            # Add small random perturbation to break degeneracy
            U += np.random.randn(nm, r) * 0.01
            V += np.random.randn(mp, r) * 0.01
            W += np.random.randn(pn, r) * 0.01
        else:
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update W: for each (i,k), W[k*n+i,:] = B[i,k,:] / ||B[i,k,:]||^2
            for i in range(n):
                for k in range(p):
                    B = np.zeros(r)
                    for j in range(m):
                        B += U[i * m + j, :] * V[j * p + k, :]
                    norm_sq = np.dot(B, B)
                    if norm_sq > 1e-15:
                        W[k * n + i, :] = B / norm_sq
                    else:
                        W[k * n + i, :] = B  # keep as is if degenerate
            
            # Update V: for each (j,k), V[j*p+k,:] = C[j,k,:] / ||C[j,k,:]||^2
            for j in range(m):
                for k in range(p):
                    C = np.zeros(r)
                    for i in range(n):
                        C += U[i * m + j, :] * W[k * n + i, :]
                    norm_sq = np.dot(C, C)
                    if norm_sq > 1e-15:
                        V[j * p + k, :] = C / norm_sq
                    else:
                        V[j * p + k, :] = C
            
            # Update U: for each (i,j), U[i*m+j,:] = D[i,j,:] / ||D[i,j,:]||^2
            for i in range(n):
                for j in range(m):
                    D = np.zeros(r)
                    for k in range(p):
                        D += V[j * p + k, :] * W[k * n + i, :]
                    norm_sq = np.dot(D, D)
                    if norm_sq > 1e-15:
                        U[i * m + j, :] = D / norm_sq
                    else:
                        U[i * m + j, :] = D
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = U[i * m + j, :]
                        vj = V[j * p + k, :]
                        wk = W[k * n + i, :]
                        val = np.dot(ui, vj * wk)
                        e = abs(val - 1.0)
                        if e > err:
                            err = e
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                break
    
    if best_result is not None and best_err < 1e-8:
        U, V, W = best_result
        U = _snap_to_half_integers(U)
        V = _snap_to_half_integers(V)
        W = _snap_to_half_integers(W)
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-7):
            return U, V, W
    
    return None

def construct():
    """
    Returns an exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    
    Tries to find a lower-rank decomposition via alternating minimization.
    Falls back to the schoolbook rank-84 solution if no lower rank is found.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin before 140s kill
    
    # Try to find a lower-rank solution
    # Known lower bounds: rank >= n*p = 21 (trivial), but better bounds exist
    # Try ranks from 42 down to 30
    for target_rank in [42, 40, 38, 36, 34, 32, 30]:
        if time.time() > deadline:
            break
        result = _try_rank_reduction(n, m, p, target_rank, 
                                      max_iters=100, tol=1e-10, deadline=deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 84
    U, V, W = _schoolbook_solution(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
