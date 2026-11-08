"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _build_tensor():
    """Build the matrix multiplication tensor for 3x3 * 3x4.
    W index = k*n + i (trace convention) where C[i,k] is the output.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, tol=1e-9):
    """Verify exact decomposition."""
    T = _build_tensor()
    rank = U.shape[1]
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol)

def _to_list(M):
    """Convert numpy array to list of lists with clean integer/half-integer values."""
    M = np.round(M * 2) / 2
    # Replace values very close to integers
    M[np.abs(M - np.round(M)) < 1e-8] = np.round(M)
    return M.tolist()

def _get_baseline():
    """Return the known rank-34 solution."""
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen Left (A11 * B1L)
    U[0][col]=1; U[4][col]=1; V[0][col]=1; V[5][col]=1; W[0][col]=1; W[4][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[0][col]=1; W[1][col]=1; W[4][col]=-1; col+=1
    U[0][col]=1; V[1][col]=1; V[5][col]=-1; W[3][col]=1; W[4][col]=1; col+=1
    U[4][col]=1; V[4][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[5][col]=1; W[0][col]=-1; W[3][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[4][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[4][col]=1; V[5][col]=1; W[0][col]=1; col+=1
    # A12*B2L
    U[2][col]=1; V[8][col]=1; W[0][col]=1; col+=1
    U[2][col]=1; V[9][col]=1; W[3][col]=1; col+=1
    U[5][col]=1; V[8][col]=1; W[1][col]=1; col+=1
    U[5][col]=1; V[9][col]=1; W[4][col]=1; col+=1
    # A21*B1L
    U[6][col]=1; V[0][col]=1; W[2][col]=1; col+=1
    U[6][col]=1; V[1][col]=1; W[5][col]=1; col+=1
    U[7][col]=1; V[4][col]=1; W[2][col]=1; col+=1
    U[7][col]=1; V[5][col]=1; W[5][col]=1; col+=1
    # A22*B2L
    U[8][col]=1; V[8][col]=1; W[2][col]=1; col+=1
    U[8][col]=1; V[9][col]=1; W[5][col]=1; col+=1
    # Strassen Right (A11 * B1R)
    U[0][col]=1; U[4][col]=1; V[2][col]=1; V[7][col]=1; W[6][col]=1; W[10][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[2][col]=1; W[7][col]=1; W[10][col]=-1; col+=1
    U[0][col]=1; V[3][col]=1; V[7][col]=-1; W[9][col]=1; W[10][col]=1; col+=1
    U[4][col]=1; V[6][col]=1; V[2][col]=-1; W[6][col]=1; W[7][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[7][col]=1; W[6][col]=-1; W[9][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[2][col]=1; V[3][col]=1; W[10][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[6][col]=1; V[7][col]=1; W[6][col]=1; col+=1
    # A12*B2R
    U[2][col]=1; V[10][col]=1; W[6][col]=1; col+=1
    U[2][col]=1; V[11][col]=1; W[9][col]=1; col+=1
    U[5][col]=1; V[10][col]=1; W[7][col]=1; col+=1
    U[5][col]=1; V[11][col]=1; W[10][col]=1; col+=1
    # A21*B1R
    U[6][col]=1; V[2][col]=1; W[8][col]=1; col+=1
    U[6][col]=1; V[3][col]=1; W[11][col]=1; col+=1
    U[7][col]=1; V[6][col]=1; W[8][col]=1; col+=1
    U[7][col]=1; V[7][col]=1; W[11][col]=1; col+=1
    # A22*B2R
    U[8][col]=1; V[10][col]=1; W[8][col]=1; col+=1
    U[8][col]=1; V[11][col]=1; W[11][col]=1; col+=1
    
    return U, V, W

def _optimize_rank(rank, T, n, m, p, max_iters=200, tol=1e-10, seed=0):
    """Try to find a rank-r decomposition using gradient descent."""
    np.random.seed(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    # Random initialization
    U = np.random.randn(nm, rank) * 0.5
    V = np.random.randn(mp, rank) * 0.5
    W = np.random.randn(pn, rank) * 0.5
    
    lr = 0.01
    for it in range(max_iters):
        # Compute reconstruction
        T_recon = np.zeros_like(T)
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = T_recon - T
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for r in range(rank):
            dU[:, r] = np.einsum('abc,b,c->a', err, V[:, r], W[:, r])
            dV[:, r] = np.einsum('abc,a,c->b', err, U[:, r], W[:, r])
            dW[:, r] = np.einsum('abc,a,b->c', err, U[:, r], V[:, r])
        
        # Update with adaptive learning rate
        loss = np.sum(err**2)
        if loss < tol:
            break
        
        # Simple adaptive step
        step = lr / (1.0 + it * 0.01)
        U -= step * dU
        V -= step * dV
        W -= step * dW
    
    # Check if we got close
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    if np.allclose(T_recon, T, atol=1e-8):
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
    return None

def _try_structured_improvement():
    """Try to find a better structured decomposition by exploiting symmetry."""
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Try ranks from 33 down to 28
    for rank in range(33, 27, -1):
        for seed in range(5):
            result = _optimize_rank(rank, T, n, m, p, max_iters=100, seed=seed)
            if result is not None:
                return result
    return None

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Start with baseline
    U_base, V_base, W_base = _get_baseline()
    best = (U_base, V_base, W_base)
    best_rank = 34
    
    # Try to find better solutions
    remaining = deadline - time.time()
    if remaining > 30:
        result = _try_structured_improvement()
        if result is not None:
            U_new, V_new, W_new = result
            if U_new.shape[1] < best_rank:
                best = (U_new, V_new, W_new)
                best_rank = U_new.shape[1]
    
    # Try additional random searches if time permits
    remaining = deadline - time.time()
    if remaining > 20:
        T = _build_tensor()
        n, m, p = DIMENSIONS
        for rank in range(best_rank - 1, max(best_rank - 4, 24), -1):
            if time.time() > deadline - 15:
                break
            for seed in range(10):
                if time.time() > deadline - 15:
                    break
                result = _optimize_rank(rank, T, n, m, p, max_iters=80, seed=seed)
                if result is not None:
                    U_new, V_new, W_new = result
                    if U_new.shape[1] < best_rank:
                        best = (U_new, V_new, W_new)
                        best_rank = U_new.shape[1]
    
    U, V, W = best
    return {"U": _to_list(np.array(U)), "V": _to_list(np.array(V)), "W": _to_list(np.array(W))}
