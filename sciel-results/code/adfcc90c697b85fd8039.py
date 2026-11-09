import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Precompute which (a,b) pairs are "active" (have non-zero T entries)
    # For alternating LS, we need to solve linear systems
    
    def solve_for_W(U, V, T):
        """Given U (nm x r) and V (mp x r), solve for W (pn x r).
        Constraint: sum_c U[a,c] * V[b,c] * W[c,d] = T[a,b,d]
        For each d, this is: sum_{a,b} T[a,b,d] * (U[a,:] * V[b,:]) = W[:,d] dot (U[a,:]*V[b,:])
        Actually: for fixed d, we have linear equations in W[:,d]:
        For each (a,b): sum_c U[a,c] * V[b,c] * W[c,d] = T[a,b,d]
        Let M[a,b,c] = U[a,c] * V[b,c]. Then for each d:
        sum_c M[a,b,c] * W[c,d] = T[a,b,d]
        This is: (M reshaped) @ W[:,d] = T[:, :, d]
        M reshaped: (nm*mp, r) where row (a*mp+b) has entries M[a,b,c]
        """
        r = U.shape[1]
        W = np.zeros((pn, r))
        # Build the linear system: A @ W[:,d] = T[:,:,d] for each d
        # A is (nm*mp, r): A[a*mp+b, c] = U[a,c] * V[b,c]
        A = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm * mp, r)
        # For each d, solve A @ W[:,d] = T[:,:,d]
        T_flat = T.reshape(nm * mp, pn)
        # Solve least squares: A @ W^T = T_flat^T
        # W^T is (r, pn), A is (nm*mp, r)
        # A @ W^T = T_flat^T => (nm*mp, r) @ (r, pn) = (nm*mp, pn)
        W_T, _, _, _ = np.linalg.lstsq(A, T_flat.T, rcond=None)
        W = W_T.T
        return W
    
    def solve_for_U(V, W, T):
        """Given V (mp x r) and W (pn x r), solve for U (nm x r).
        Constraint: sum_c U[a,c] * V[b,c] * W[c,d] = T[a,b,d]
        For fixed a: sum_c U[a,c] * (V[b,c] * W[c,d]) = T[a,b,d] for all b,d
        Let N[b,d,c] = V[b,c] * W[c,d]. Then:
        sum_c U[a,c] * N[b,d,c] = T[a,b,d]
        Reshape: for each a, U[a,:] @ (N reshaped to (mp*pn, r))^T = T[a,:,:] reshaped
        Actually: U[a,:] @ M_b = T[a,b,:] where M_b[c,d] = V[b,c]*W[c,d]
        Let M be (mp, pn, r) -> for each b, M[b] is (pn, r)
        Then U[a,:] @ M[b].T = T[a,b,:]
        Stack over b: U[a,:] @ (M reshaped to (r, mp*pn))^T = T[a,:,:] reshaped to (mp*pn,)
        Let B be (r, mp*pn): B[c, b*pn+d] = V[b,c] * W[c,d]
        Then U[a,:] @ B = T[a, :, :] reshaped to (mp*pn,)
        So: U @ B.T = T_flat where T_flat is (nm, mp*pn)
        B.T is (mp*pn, r)
        U @ B.T = T_flat => U = T_flat @ pinv(B.T)
        """
        r = V.shape[1]
        # B[c, b*pn+d] = V[b,c] * W[c,d]
        B = np.zeros((r, mp * pn))
        for b in range(mp):
            for d in range(pn):
                B[:, b * pn + d] = V[b, :] * W[d, :]
        T_flat = T.reshape(nm, mp * pn)
        # U @ B.T = T_flat
        # U = T_flat @ pinv(B.T)
        B_T = B.T  # (mp*pn, r)
        U, _, _, _ = np.linalg.lstsq(B_T, T_flat.T, rcond=None)
        U = U.T
        return U
    
    def solve_for_V(U, W, T):
        """Given U (nm x r) and W (pn x r), solve for V (mp x r).
        Constraint: sum_c U[a,c] * V[b,c] * W[c,d] = T[a,b,d]
        For fixed b: sum_c V[b,c] * (U[a,c] * W[c,d]) = T[a,b,d] for all a,d
        Let M[a,d,c] = U[a,c] * W[c,d]. Then:
        sum_c V[b,c] * M[a,d,c] = T[a,b,d]
        Reshape: for each b, V[b,:] @ C = T[:,b,:] reshaped
        where C[c, a*pn+d] = U[a,c] * W[c,d]
        C is (r, nm*pn)
        V[b,:] @ C = T[:,b,:] reshaped to (nm*pn,)
        So: V @ C = T_flat2 where T_flat2 is (mp, nm*pn)
        V = T_flat2 @ pinv(C)
        """
        r = U.shape[1]
        # C[c, a*pn+d] = U[a,c] * W[c,d]
        C = np.zeros((r, nm * pn))
        for a in range(nm):
            for d in range(pn):
                C[:, a * pn + d] = U[a, :] * W[d, :]
        T_flat2 = T.T.reshape(mp, nm * pn)
        # V @ C = T_flat2
        # V = T_flat2 @ pinv(C)
        V, _, _, _ = np.linalg.lstsq(C.T, T_flat2.T, rcond=None)
        V = V.T
        return V
    
    def alternating_ls(r, init_U=None, init_V=None, init_W=None, max_iters=200, tol=1e-10):
        """Alternating least squares to find U, V, W of rank r."""
        if init_U is None:
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
        else:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
        
        for it in range(max_iters):
            W_old = W.copy()
            W = solve_for_W(U, V, T)
            U = solve_for_U(V, W, T)
            V = solve_for_V(U, W, T)
            
            # Check convergence
            if np.linalg.norm(W - W_old) < tol:
                break
        
        # Compute error
        R = np.einsum('ak,bk,ck->abc', U, V, W)
        err = np.linalg.norm(R - T)
        return U, V, W, err
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose T."""
        R = np.einsum('ak,bk,ck->abc', U, V, W)
        err = np.linalg.norm(R - T)
        return err < tol, err
    
    # Schoolbook initialization
    rank_schoolbook = n * m * p
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    best_err = 0.0
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Try ranks from schoolbook-1 down
    for r in range(rank_schoolbook - 1, 10, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Multiple restarts
        n_restarts = 10
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                # Initialize from schoolbook, truncate
                U_init = U_school[:, :r]
                V_init = V_school[:, :r]
                W_init = W_school[:, :r]
            elif restart == 1:
                # Random initialization
                U_init = None
                V_init = None
                W_init = None
            else:
                # Perturbed schoolbook
                U_init = U_school[:, :r] + 0.1 * np.random.randn(nm, r)
                V_init = V_school[:, :r] + 0.1 * np.random.randn(mp, r)
                W_init = W_school[:, :r] + 0.1 * np.random.randn(pn, r)
            
            U, V, W, err = alternating_ls(r, U_init, V_init, W_init, max_iters=100)
            
            if err < 1e-8:
                best_U, best_V, best_W = U, V, W
                best_rank = r
                best_err = err
                found = True
                break
        
        if found:
            # Found a solution at rank r, continue to try lower
            pass
        else:
            # If we've tried several ranks without success, we can stop
            # But let's continue a bit more
            pass
    
    # Final polish: if we found a solution, try to verify and clean up
    if best_err < 1e-8:
        # Round small values to zero for cleanliness
        U_clean = np.where(np.abs(best_U) < 1e-10, 0.0, best_U)
        V_clean = np.where(np.abs(best_V) < 1e-10, 0.0, best_V)
        W_clean = np.where(np.abs(best_W) < 1e-10, 0.0, best_W)
        
        # Verify cleaned solution
        R = np.einsum('ak,bk,ck->abc', U_clean, V_clean, W_clean)
        if np.linalg.norm(R - T) < 1e-8:
            best_U, best_V, best_W = U_clean, V_clean, W_clean
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
