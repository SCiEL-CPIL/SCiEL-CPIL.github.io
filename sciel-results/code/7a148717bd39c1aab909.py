import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor M_{n,m,p}."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the MM tensor."""
    T = _build_tensor(n, m, p)
    T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.max(np.abs(T_recon - T)) < tol, np.max(np.abs(T_recon - T))

def _schoolbook(n, m, p):
    """Generate schoolbook decomposition for M_{n,m,p}."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, idx] = 1
                V[v_idx, idx] = 1
                W[w_idx, idx] = 1
                idx += 1
    return U, V, W

def _strassen_block(n, m, p):
    """
    Try to use Strassen-style block decomposition.
    For 3x3 x 3x8, we can't directly use 2x2 Strassen, but we can try
    a block approach with 2x2 blocks on the A and B matrices.
    
    A is 3x3, B is 3x8.
    Split A into 2x2 blocks (top-left 2x2, top-right 2x1, bottom-left 1x2, bottom-right 1x1)
    This doesn't work cleanly. Let's try a different approach.
    
    Actually, let's just use the schoolbook for now and try to find improvements
    through random initialization and ALS.
    """
    return _schoolbook(n, m, p)

def _als_decompose(n, m, p, rank, max_iter=1000, tol=1e-10, seed=42):
    """
    Alternating Least Squares to find a rank-r decomposition of the MM tensor.
    """
    np.random.seed(seed)
    T = _build_tensor(n, m, p)
    
    # Initialize
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Fix V, W, solve for U
        # T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
        # For each u: T[u,:,:] = sum_r U[u,r] * (V * W)[:,:,r]
        # Let G = V * W, shape (m*p, p*n, rank)
        G = V * W  # broadcast: (m*p, 1, rank) * (1, p*n, rank) -> (m*p, p*n, rank)
        # T[u, :, :] = sum_r U[u,r] * G[:, :, r]
        # Reshape: T.reshape(n*m, m*p*p*n) and G.reshape(m*p*p*n, rank)
        G_flat = G.reshape(m * p * p * n, rank)
        T_flat = T.reshape(n * m, m * p * p * n)
        # Solve T_flat = U @ G_flat.T for U
        try:
            U_new, _, _, _ = np.linalg.lstsq(G_flat.T, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass
        
        # Fix U, W, solve for V
        G = U * W  # (n*m, 1, rank) * (1, p*n, rank) -> (n*m, p*n, rank)
        G_flat = G.reshape(n * m * p * n, rank)
        T_flat = T.reshape(m * p, n * m * p * n)
        try:
            V_new, _, _, _ = np.linalg.lstsq(G_flat.T, T_flat.T, rcond=None)
            V = V_new.T
        except:
            pass
        
        # Fix U, V, solve for W
        G = U * V  # (n*m, 1, rank) * (1, m*p, rank) -> (n*m, m*p, rank)
        G_flat = G.reshape(n * m * m * p, rank)
        T_flat = T.reshape(p * n, n * m * m * p)
        try:
            W_new, _, _, _ = np.linalg.lstsq(G_flat.T, T_flat.T, rcond=None)
            W = W_new.T
        except:
            pass
        
        # Check convergence
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    valid, err = _verify_decomposition(best_U, best_V, best_W, n, m, p, tol)
    if valid:
        return best_U, best_V, best_W, rank
    else:
        return None, None, None, None

def _try_random_rank(n, m, p, rank, n_trials=50, max_iter=200, tol=1e-10):
    """Try multiple random initializations for ALS at a given rank."""
    for trial in range(n_trials):
        seed = trial * 1000 + 42
        U, V, W, achieved = _als_decompose(n, m, p, rank, max_iter=max_iter, tol=tol, seed=seed)
        if U is not None:
            return U, V, W, achieved
    return None, None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank incrementally from schoolbook down
    # The theoretical lower bound for 3x3 x 3x8 is at least max(3*3, 3*8, 3*8) = 24
    # but could be higher. Let's try from 71 down to 30.
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 5:
            break
        
        U_new, V_new, W_new, achieved_rank = _try_random_rank(
            n, m, p, target_rank, n_trials=10, max_iter=100, tol=1e-10
        )
        
        if achieved_rank is not None and achieved_rank < best_rank:
            best_rank = achieved_rank
            best_U, best_V, best_W = U_new, V_new, W_new
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
