"""
Rewritten program for matrix multiplication tensor decomposition.
Task: M_{2,3,10}
Dimensions: n=2, m=3, p=10
Standard rank: 60
"""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145  # seconds, safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dense 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_exact(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _try_rank_als(n, m, p, rank, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition using alternating least squares."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    # Initialize
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    # Pre-compute flat tensors for speed
    T_flat_u = T.reshape(n * m, -1)
    T_flat_v = T.reshape(m * p, -1)
    T_flat_w = T.reshape(p * n, -1)
    
    for it in range(max_iters):
        # Update U
        M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, rank)
        M_flat = M.transpose(2, 0, 1).reshape(rank, -1)  # (rank, m*p*p*n)
        try:
            U = np.linalg.lstsq(M_flat.T, T_flat_u.T, rcond=None)[0].T
        except:
            pass
        
        # Update V
        N = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, rank)
        N_flat = N.transpose(2, 0, 1).reshape(rank, -1)  # (rank, n*m*p*n)
        try:
            V = np.linalg.lstsq(N_flat.T, T_flat_v.T, rcond=None)[0].T
        except:
            pass
        
        # Update W
        P = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, rank)
        P_flat = P.transpose(2, 0, 1).reshape(rank, -1)  # (rank, n*m*m*p)
        try:
            W = np.linalg.lstsq(P_flat.T, T_flat_w.T, rcond=None)[0].T
        except:
            pass
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < 1e-10:
            break
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err < 1e-10:
        return U, V, W, True
    return U, V, W, False

def _try_rank_gradient(n, m, p, rank, max_iters=5000, lr=0.01, seed=42):
    """Try to find a rank-r decomposition using gradient descent."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    for it in range(max_iters):
        # Compute gradients
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_recon - T
        
        # Gradient for U: sum over b,c of diff[a,b,c] * V[b,r] * W[c,r]
        grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
        # Gradient for V
        grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
        # Gradient for W
        grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if it % 100 == 0:
            err = np.max(np.abs(T_recon - T))
            if err < 1e-10:
                break
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err < 1e-10:
        return U, V, W, True
    return U, V, W, False

def _get_trivial_decomposition(n, m, p):
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_structured_decomposition(n, m, p):
    """
    Try a structured decomposition exploiting the small n=2 dimension.
    
    For M_{2,3,10}, we have:
    - Input A is 2x3 (6 elements)
    - Input B is 3x10 (30 elements)
    - Output C is 2x10 (20 elements)
    
    The trivial decomposition has rank 60.
    
    Key insight: Each output element C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can try to share computations across different (i,k) pairs.
    
    For n=2, the two rows of A are A[0,:] and A[1,:].
    
    Let's try to find a decomposition where we use fewer than 60 rank-1 tensors.
    """
    n, m, p = 2, 3, 10
    
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But this is a weak lower bound.
    
    # Let's try ranks from 30 to 59
    # For each rank, try multiple seeds with ALS
    
    start_time = time.time()
    
    for rank in range(30, 60):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Try a few seeds for each rank
        num_seeds = 3 if rank <= 35 else 2
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            max_iters = 5000 if rank <= 35 else 2000
            
            U, V, W, success = _try_rank_als(n, m, p, rank, max_iters=max_iters, seed=seed)
            if success:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Also try gradient descent for lower ranks
            if rank <= 35:
                U, V, W, success = _try_rank_gradient(n, m, p, rank, max_iters=3000, lr=0.005, seed=seed)
                if success:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the structured decomposition approach
    result = _try_structured_decomposition(n, m, p)
    if result is not None:
        return result
    
    # Fall back to trivial decomposition
    return _get_trivial_decomposition(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
