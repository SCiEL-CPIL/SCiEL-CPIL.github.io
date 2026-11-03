"""Tensor decomposition for 3x3x7 matrix multiplication with rank-minimization search."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(n, m, p, U, V, W):
    """Check if the decomposition exactly matches the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_tensor(n, m, p):
    """Build the full multiplication tensor as a 3D array T[a,b,c]."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _alternating_minimize(n, m, p, rank, max_iter=2000, tol=1e-10, seed=None):
    """
    Alternating minimization for rank-r tensor decomposition.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    if seed is not None:
        np.random.seed(seed)
    
    T = _build_tensor(n, m, p)
    shape = (n*m, m*p, p*n)
    
    # Initialize with small random perturbations
    U = np.random.randn(shape[0], rank) * 0.1
    V = np.random.randn(shape[1], rank) * 0.1
    W = np.random.randn(shape[2], rank) * 0.1
    
    for it in range(max_iter):
        # Update U: for each a, U[a,:] = argmin sum_r U[a,r]*V[b,r]*W[c,r] vs T[a,:,:]
        # This is a linear system: for each a, solve min ||U[a,:] @ (V*W.T) - T[a,:,:].T||
        # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed V,W: T[a,b,c] = U[a,:] @ (V[b,:]*W[c,:]).T ... no
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Let M_VW[b,c] = sum_r V[b,r]*W[c,r] = V @ W.T
        # Then T[a,:,:] = U[a,:] @ M_VW.T ... check:
        # sum_r U[a,r] * M_VW[b,c] = sum_r U[a,r] * sum_s V[b,s]*W[c,s] -- no that's wrong
        # M_VW[b,c] = sum_r V[b,r]*W[c,r]
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is NOT U[a,:] @ M_VW.T because M_VW mixes over r but we need elementwise product per r
        
        # Correct approach: For fixed V and W, define for each b,c:
        # T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
        # Let P[b,c,r] = V[b,r]*W[c,r]
        # Then T[a,b,c] = sum_r U[a,r] * P[b,c,r]
        # So for fixed a: T[a,b,c] over all b,c is U[a,:] @ P.T where P has shape (b,c,r)
        # Reshape: P_flat has shape (m*p*p*n, rank) where index = b*(p*n)+c
        # T_flat[a, :] has shape (m*p*p*n,)
        # T_flat[a, :] = U[a,:] @ P_flat.T
        # So U[a,:] = T_flat[a,:] @ P_flat @ (P_flat.T @ P_flat)^{-1}
        
        # Build P_flat: shape (m*p*p*n, rank)
        P_flat = np.zeros((m*p*p*n, rank))
        for b in range(m*p):
            for c in range(p*n):
                P_flat[b*(p*n)+c, :] = V[b, :] * W[c, :]
        
        # Solve for U
        PTP = P_flat.T @ P_flat + 1e-12 * np.eye(rank)
        PT = P_flat.T
        U_new = (T.reshape(n*m, -1) @ PT) @ np.linalg.inv(PTP)
        U = U_new
        
        # Update V: for fixed U,W
        # T[a,b,c] = sum_r V[b,r] * (U[a,r]*W[c,r])
        # For fixed b: T[:,b,c] = V[b,:] @ Q.T where Q[a,c,r] = U[a,r]*W[c,r]
        Q_flat = np.zeros((n*m*p*n, rank))
        for a in range(n*m):
            for c in range(p*n):
                Q_flat[a*(p*n)+c, :] = U[a, :] * W[c, :]
        QTP = Q_flat.T @ Q_flat + 1e-12 * np.eye(rank)
        V_new = (T.reshape(-1, m*p, p*n).transpose(1,0,2).reshape(m*p, -1) @ Q_flat.T) @ np.linalg.inv(QTP)
        V = V_new
        
        # Update W: for fixed U,V
        # T[a,b,c] = sum_r W[c,r] * (U[a,r]*V[b,r])
        R_flat = np.zeros((n*m*m*p, rank))
        for a in range(n*m):
            for b in range(m*p):
                R_flat[a*(m*p)+b, :] = U[a, :] * V[b, :]
        RTP = R_flat.T @ R_flat + 1e-12 * np.eye(rank)
        W_new = (T.reshape(n*m, m*p, -1).transpose(2,0,1).reshape(p*n, -1) @ R_flat.T) @ np.linalg.inv(RTP)
        W = W_new
        
        # Check convergence
        if it % 50 == 0:
            T_recon = np.zeros(shape)
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            err = np.linalg.norm(T - T_recon)
            if err < tol:
                break
    
    # Final error
    T_recon = np.zeros(shape)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    err = np.linalg.norm(T - T_recon)
    
    if err < 1e-8:
        return U, V, W, err
    return None, None, None, err

def _try_reduced_rank(n, m, p, target_rank, time_limit, max_seeds=50):
    """Try to find a decomposition of given rank using alternating minimization."""
    start = time.time()
    deadline = start + time_limit
    
    for seed in range(max_seeds):
        if time.time() > deadline:
            break
        U, V, W, err = _alternating_minimize(n, m, p, target_rank, max_iter=1000, tol=1e-10, seed=seed)
        if U is not None:
            # Verify
            if _verify(n, m, p, U.tolist(), V.tolist(), W.tolist()):
                return U, V, W
    return None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (guaranteed valid)
    result = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bounds: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Also known: for 3x3x3, rank is at most 23 (Strassen-like), but we're doing 3x3x7
    # Try ranks from lower bound upward
    
    lower_bound = max(n*m, m*p, p*n)
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 30:  # Leave 30s margin
            break
        
        U, V, W = _try_reduced_rank(n, m, p, target_rank, time_limit=10, max_seeds=20)
        if U is not None:
            best_rank = target_rank
            result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break  # Found minimal rank for this search, stop
    
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
