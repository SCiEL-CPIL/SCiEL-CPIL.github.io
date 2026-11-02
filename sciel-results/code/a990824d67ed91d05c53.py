"""Matrix multiplication 4x4x7 - rank minimization via random search + refinement."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [4, 4, 7]

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor T[i,j,k] with shape (n*m, m*p, p*n).
    T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i,j,k."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-6):
    """Verify that sum_c U[:,c] V[:,c] W[:,c] == T."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    recon = np.tensordot(U, V, axes=([1],[1]))  # (n*m, m*p, rank)
    recon = np.tensordot(recon, W, axes=([2],[1]))  # (n*m, m*p, p*n)
    return np.max(np.abs(recon - T)) < tol

def _refine(U, V, W, T, max_iters=200, lr=0.01, tol=1e-8):
    """Gradient descent refinement of the factor matrices."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = np.inf
    
    for it in range(max_iters):
        # Compute reconstruction
        # recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        UV = U @ V.T  # (n*m, m*p, rank) -> actually U @ V.T gives (n*m, rank) @ (rank, m*p)
        # Let's do it properly
        # recon = einsum('ar,br,cr->abc', U, V, W)
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = recon - T
        loss = np.sum(diff * diff)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Gradients
        # dL/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
        # dL/dV[b,r] = 2 * sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
        # dL/dW[c,r] = 2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
        
        gU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        gV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        gW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
    
    return best_U, best_V, best_W, best_loss

def _random_init(n, m, p, rank, seed=42):
    """Random initialization with small values."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    return U, V, W

def _structured_init(n, m, p, rank, seed=42):
    """Structured initialization based on partial schoolbook + random."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Place some schoolbook terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < rank:
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    # Fill remaining with small random
    if col < rank:
        U[:, col:] = rng.randn(n * m, rank - col) * 0.1
        V[:, col:] = rng.randn(m * p, rank - col) * 0.1
        W[:, col:] = rng.randn(p * n, rank - col) * 0.1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Start with schoolbook
    schoolbook_rank = n * m * p  # 112
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Build schoolbook
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try various ranks below schoolbook
    # Known lower bounds: for 4x4x7, best known is around 55-60
    # Try ranks from 111 down
    ranks_to_try = list(range(111, 40, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for seed in range(10):
            if time.time() > deadline - 15:
                break
            
            U, V, W = _random_init(n, m, p, rank, seed=seed + rank)
            
            # Quick gradient descent
            U, V, W, loss = _refine(U, V, W, T, max_iters=100, lr=0.005, tol=1e-7)
            
            if loss < 1e-5:
                # Verify more carefully
                if _verify(U, V, W, T, tol=1e-6):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    print(f"Found rank {rank} with loss {loss:.2e}")
                    break
            elif loss < 1e-3:
                # Try more refinement
                if time.time() < deadline - 10:
                    U, V, W, loss = _refine(U, V, W, T, max_iters=200, lr=0.001, tol=1e-8)
                    if loss < 1e-6 and _verify(U, V, W, T, tol=1e-6):
                        best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        print(f"Found rank {rank} after extra refinement")
                        break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
