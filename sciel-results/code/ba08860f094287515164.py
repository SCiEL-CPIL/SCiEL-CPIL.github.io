"""Rewritten program: exploit Strassen-like structure for 2x3 * 3x4."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook solution as fallback (rank 24)
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The known lower bound for 2x3x4 is at least 13 (from algebraic complexity theory)
    # Let's try ranks from 23 down to 13
    for rank in range(best_rank - 1, 12, -1):
        if time.time() > deadline:
            break
        
        try:
            result = _try_rank_strassen(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                # If we found a good rank, continue trying lower
                continue
        except Exception:
            continue
    
    return best_result

def _schoolbook(n, m, p):
    """Generate the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _try_rank_strassen(n, m, p, rank, deadline):
    """
    Try to find a rank decomposition using structured initialization
    based on Strassen-like block decomposition.
    
    For 2x3 * 3x4, we can think of the 3x4 matrix as having structure.
    The key insight: if we could decompose the inner dimension cleverly.
    
    Strategy: Use alternating minimization with structured initializations.
    """
    
    # Residual function
    def compute_residuals(params, n, m, p, rank):
        U = params[:n*m*rank].reshape(n, m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p, n, rank)
        
        # S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        
        return S.flatten() - 1.0
    
    def alternating_minimize(params, n, m, p, rank, deadline, max_iters=200):
        """
        Alternating minimization: fix two factors, solve for the third.
        """
        U = params[:n*m*rank].reshape(n, m, rank).copy()
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank).copy()
        W = params[n*m*rank + m*p*rank:].reshape(p, n, rank).copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Fix V, W, solve for U:
            # For each (i,j): sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r]).T = 1 for all k
            # This is: U[i,j,:] @ (V[j,:,r] * W[:,i,r]) = 1 for all k
            # Let A_ij = V[j,:,r] * W[:,i,r]  (shape p x rank)
            # Then U[i,j,:] @ A_ij = 1_p  (vector of ones, shape p)
            # So U[i,j,:] = (A_ij^T)^{-1} * 1_p  (if invertible)
            # Or least squares if not
            
            # More precisely: we want sum_r U[i,j,r] * A_ij[k,r] = 1 for each k
            # This is a linear system: A_ij @ U[i,j,:]^T = 1_p
            # where A_ij is p x rank, U[i,j,:] is 1 x rank
            
            for i in range(n):
                for j in range(m):
                    A_ij = V[j, :, :] * W[:, i, :]  # shape (p, rank)
                    # Solve A_ij @ u = 1_p where u is shape (rank,)
                    try:
                        if rank >= p:
                            # Overdetermined or square: use lstsq
                            u, residuals, rank_u, sv = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                        else:
                            # Underdetermined: find minimum norm solution
                            # A_ij @ u = 1_p, A_ij is p x rank with rank < p
                            # This may not have a solution; use lstsq
                            u, residuals, rank_u, sv = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                        U[i, j, :] = u
                    except np.linalg.LinAlgError:
                        pass
            
            # Fix U, W, solve for V:
            # For each (j,k): sum_r V[j,k,r] * (U[:,j,r] * W[k,:,r]).T = 1 for all i
            # A_jk = U[:,j,r] * W[k,:,r]  (shape n x rank)
            # A_jk @ V[j,k,:]^T = 1_n
            
            for j in range(m):
                for k in range(p):
                    A_jk = U[:, j, :] * W[k, :, :]  # shape (n, rank)
                    try:
                        v, residuals, rank_v, sv = np.linalg.lstsq(A_jk, np.ones(n), rcond=None)
                        V[j, k, :] = v
                    except np.linalg.LinAlgError:
                        pass
            
            # Fix U, V, solve for W:
            # For each (k,i): sum_r W[k,i,r] * (U[i,:,r] * V[:,k,r]).T = 1 for all j
            # A_ki = U[i,:,r] * V[:,k,r]  (shape m x rank)
            # A_ki @ W[k,i,:]^T = 1_m
            
            for k in range(p):
                for i in range(n):
                    A_ki = U[i, :, :] * V[:, k, :]  # shape (m, rank)
                    try:
                        w, residuals, rank_w, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                        W[k, i, :] = w
                    except np.linalg.LinAlgError:
                        pass
            
            # Check convergence
            S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            err = np.max(np.abs(S - 1.0))
            if err < 1e-10:
                break
        
        return U, V, W, err
    
    # Multiple restarts with different initializations
    np.random.seed(rank * 7919)  # Large prime for better distribution
    
    num_restarts = 15
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Random initialization
            U0 = np.random.randn(n, m, rank)
            V0 = np.random.randn(m, p, rank)
            W0 = np.random.randn(p, n, rank)
        elif restart == 1:
            # Sparse initialization
            U0 = np.zeros((n, m, rank))
            V0 = np.zeros((m, p, rank))
            W0 = np.zeros((p, n, rank))
            for r in range(rank):
                i, j, k = np.random.randint(n, size=1), np.random.randint(m, size=1), np.random.randint(p, size=1)
                U0[i[0], j[0], r] = 1
                V0[j[0], k[0], r] = 1
                W0[k[0], i[0], r] = 1
        elif restart == 2:
            # Structured: try to exploit the 2x3x4 geometry
            # Initialize with patterns that respect the dimensions
            U0 = np.zeros((n, m, rank))
            V0 = np.zeros((m, p, rank))
            W0 = np.zeros((p, n, rank))
            for r in range(rank):
                # Assign each rank component to cover a subset of indices
                i = r % n
                j = (r // n) % m
                k = (r // (n*m)) % p
                U0[i, j, r] = 1.0
                V0[j, k, r] = 1.0
                W0[k, i, r] = 1.0
        elif restart == 3:
            # Initialize from a perturbed schoolbook-like structure
            U0 = np.zeros((n, m, rank))
            V0 = np.zeros((m, p, rank))
            W0 = np.zeros((p, n, rank))
            # Use rank components to cover (i,j,k) triples with some overlap
            triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            np.random.shuffle(triples)
            for r in range(min(rank, len(triples))):
                i, j, k = triples[r]
                U0[i, j, r] = 1.0
                V0[j, k, r] = 1.0
                W0[k, i, r] = 1.0
            # Add random perturbation to remaining components
            for r in range(len(triples), rank):
                U0[:, :, r] = np.random.randn(n, m) * 0.1
                V0[:, :, r] = np.random.randn(m, p) * 0.1
                W0[:, :, r] = np.random.randn(p, n) * 0.1
        else:
            # More random restarts with varying scales
            scale = 0.5 + 0.1 * restart
            U0 = np.random.randn(n, m, rank) * scale
            V0 = np.random.randn(m, p, rank) * scale
            W0 = np.random.randn(p, n, rank) * scale
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        U_opt, V_opt, W_opt, err = alternating_minimize(params0, n, m, p, rank, deadline)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
        
        # If we found an exact solution, return it
        if best_err < 1e-10:
            # Try to snap to rational values
            for denom in [1, 2, 3, 4, 6]:
                U_snap = np.round(best_U * denom) / denom
                V_snap = np.round(best_V * denom) / denom
                W_snap = np.round(best_W * denom) / denom
                
                S = np.einsum('ijr,jkr,kir->ijk', U_snap, V_snap, W_snap)
                max_err = np.max(np.abs(S - 1.0))
                
                if max_err < 1e-9:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
            
            # Return the floating point solution (should be close to rational)
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
    
    return None

# EVOLVE-BLOCK-END
