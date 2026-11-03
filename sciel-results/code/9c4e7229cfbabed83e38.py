"""Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
DEADLINE = 140  # seconds, leave 10s margin before 150s kill


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _verify(U, V, W, tol=1e-9):
    """Check exact decomposition using einsum."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T_computed = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    T_expected = np.ones((n, m, p), dtype=np.float64)
    return np.max(np.abs(T_computed - T_expected)) < tol


def _april_refine(U, V, W, max_iter=5000, lr_init=0.1, tol=1e-12):
    """APRIL-style alternating refinement for tensor decomposition.
    
    For the matrix multiplication tensor T[i,j,k]=1, we want:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    Alternating updates: fix V,W -> optimize U; fix U,W -> optimize V; fix U,V -> optimize W.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _time_left() < 10:
            break
        
        # Compute current tensor
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T_cur = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        diff = T_cur - 1.0
        loss = np.sum(diff**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Adaptive learning rate
        lr = lr_init / (1.0 + 0.01 * it)
        
        # Update U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_r U[i*m+j,r] * (V[j*p+0,r]*W[0*n+i,r] + ... ) = 1
        # More precisely: sum_k sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is a linear system in U[i*m+j,:]:
        # sum_r U[i*m+j,r] * M_U[i,j,r] = 1 where M_U[i,j,r] = sum_k V[j*p+k,r]*W[k*n+i,r]
        
        # Compute M_U[i,j,r] = sum_k V_3d[j,k,r] * W_3d[k,i,r]
        M_U = np.einsum('jkr,kir->ijr', V_3d, W_3d)  # shape (n, m, rank)
        
        # For each (i,j): U[i*m+j, :] = argmin sum_r (U[i*m+j,r]*M_U[i,j,r] - 1)^2
        # This is a least squares problem: M_U[i,j,:]^T * U[i*m+j,:] = 1
        # Solution: U[i*m+j,:] = M_U[i,j,:] / (M_U[i,j,:]^T * M_U[i,j,:]) if norm > 0
        # But this minimizes ||M_U^T * U - 1||^2, which is not quite right.
        # Actually we want sum_r U[r] * M_U[r] = 1, which is underdetermined if rank > 1.
        # The minimum norm solution is U = M_U / ||M_U||^2
        
        # Use gradient step instead for stability:
        # dU[i,j,r] = M_U[i,j,r] * (sum_s U[i,j,s]*M_U[i,j,s] - 1)
        # = M_U[i,j,r] * (T_cur[i,j] - 1) ... wait no
        # Actually dL/dU[i,j,r] = sum_k V[j,k,r]*W[k,i,r]*(T_cur[i,j,k]-1)
        # Let's just use the gradient form
        
        dU = np.einsum('ijk,jkc,kic->ijc', diff, V_3d, W_3d)
        U -= lr * dU.reshape(n * m, rank)
        
        # Recompute with updated U
        U_3d = U.reshape(n, m, rank)
        T_cur = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        diff = T_cur - 1.0
        
        # Update V
        dV = np.einsum('ijk,ijc,kic->jkc', diff, U_3d, W_3d)
        V -= lr * dV.reshape(m * p, rank)
        
        # Recompute with updated V
        V_3d = V.reshape(m, p, rank)
        T_cur = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        diff = T_cur - 1.0
        
        # Update W
        dW = np.einsum('ijk,ijc,jkc->kic', diff, U_3d, V_3d)
        W -= lr * dW.reshape(p * n, rank)
    
    return best_U, best_V, best_W, best_loss


def _try_rank(rank, n_restarts=10, max_iter=3000):
    """Try to find a rank-rank decomposition with multiple random restarts."""
    n, m, p = DIMENSIONS
    best_result = None
    best_loss = np.inf
    
    for seed in range(n_restarts):
        if _time_left() < 15:
            break
        
        rng = np.random.RandomState(seed * 1000 + rank)
        
        # Random initialization
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Normalize to prevent blowup
        for c in range(rank):
            norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
            if norm > 1e-12:
                U[:, c] /= norm
                V[:, c] /= norm
                W[:, c] /= norm
        
        # Run APRIL refinement
        U_r, V_r, W_r, loss = _april_refine(U, V, W, max_iter=max_iter)
        
        if loss < best_loss:
            best_loss = loss
            best_result = (U_r, V_r, W_r)
        
        if best_loss < 1e-12:
            break
    
    if best_result is not None and best_loss < 1e-10:
        U_r, V_r, W_r = best_result
        # Verify
        if _verify(U_r, V_r, W_r):
            return U_r, V_r, W_r, True
        # Try rounding to half-integers
        Ur = np.round(U_r * 2) / 2.0
        Vr = np.round(V_r * 2) / 2.0
        Wr = np.round(W_r * 2) / 2.0
        if _verify(Ur, Vr, Wr):
            return Ur, Vr, Wr, True
    
    return None, None, None, False


def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But we know from theory that the rank is at least something higher.
    # Let's try ranks from a reasonable lower bound up.
    
    lower_bound = max(n * m, m * p, p * n)  # 24
    
    # Try ranks from lower_bound + a few up to schoolbook_rank - 1
    # But be time-efficient: try key ranks first
    ranks_to_try = list(range(lower_bound, schoolbook_rank))
    
    # Sort by rank (ascending) to find the minimum
    for rank in ranks_to_try:
        if _time_left() < 20:
            break
        
        # Fewer restarts for lower ranks (harder to find), more for higher
        n_restarts = 5 if rank < 40 else 8
        max_iter = 2000 if rank < 40 else 3000
        
        U, V, W, success = _try_rank(rank, n_restarts=n_restarts, max_iter=max_iter)
        
        if success:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
