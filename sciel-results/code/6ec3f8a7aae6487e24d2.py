"""Matrix multiplication 4x4x7 - rank minimization via structured search + refinement."""
import numpy as np
import time
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
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(recon - T)) < tol

def _refine_adaptive(U, V, W, T, max_iters=300, tol=1e-9, base_lr=0.01):
    """Adaptive gradient descent with momentum and learning rate decay."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = np.inf
    
    # Momentum buffers
    mU = np.zeros_like(U)
    mV = np.zeros_like(V)
    mW = np.zeros_like(W)
    beta1, beta2 = 0.9, 0.999
    lr = base_lr
    
    for it in range(max_iters):
        # Compute reconstruction
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = recon - T
        loss = np.sum(diff * diff)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Gradients
        gU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        gV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        gW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        # Adam-like update
        mU = beta1 * mU + (1 - beta1) * gU
        mV = beta1 * mV + (1 - beta1) * gV
        mW = beta1 * mW + (1 - beta1) * gW
        
        # Simple adaptive lr: decay over time
        lr_t = base_lr / (1.0 + 0.01 * it)
        
        U -= lr_t * mU
        V -= lr_t * mV
        W -= lr_t * mW
    
    return best_U, best_V, best_W, best_loss

def _structured_init_strassen(n, m, p, rank, seed=42):
    """Structured initialization using Strassen-like block patterns."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Try to use a mix of schoolbook and Strassen-inspired patterns
    # For 4x4x7, we can try to pack multiple multiplications per rank term
    
    col = 0
    # First, place some pure schoolbook terms
    n_schoolbook = min(rank // 4, n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < n_schoolbook:
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    # For remaining columns, use structured random patterns
    if col < rank:
        # Use patterns that might capture linear combinations
        for c in range(col, rank):
            # Random sparse pattern
            nnz_U = max(1, int(0.3 * n * m))
            nnz_V = max(1, int(0.3 * m * p))
            nnz_W = max(1, int(0.3 * p * n))
            
            U_idx = rng.choice(n * m, size=nnz_U, replace=False)
            V_idx = rng.choice(m * p, size=nnz_V, replace=False)
            W_idx = rng.choice(p * n, size=nnz_W, replace=False)
            
            U[U_idx, c] = rng.choice([-1, 1], size=nnz_U)
            V[V_idx, c] = rng.choice([-1, 1], size=nnz_V)
            W[W_idx, c] = rng.choice([-1, 1], size=nnz_W)
    
    return U, V, W

def _try_polish(U, V, W, T, max_iters=500, tol=1e-10):
    """High-precision polish pass with very tight tolerance."""
    return _refine_adaptive(U, V, W, T, max_iters=max_iters, tol=tol, base_lr=0.001)

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
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
    
    # Strategy: Try structured initializations at various ranks
    # Focus on ranks that are likely achievable based on known bounds
    # For 4x4x7, best known is around 55-65, so try from 80 down
    
    # Phase 1: Try structured initializations at higher ranks (easier to converge)
    structured_ranks = [80, 75, 70, 65, 60, 55]
    
    for rank in structured_ranks:
        if time.time() > deadline - 40:
            break
        
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            U, V, W = _structured_init_strassen(n, m, p, rank, seed=seed + rank * 100)
            
            # Adaptive refinement
            U, V, W, loss = _refine_adaptive(U, V, W, T, max_iters=150, tol=1e-7, base_lr=0.008)
            
            if loss < 1e-4:
                # Polish pass
                if time.time() < deadline - 10:
                    U, V, W, loss = _try_polish(U, V, W, T, max_iters=200, tol=1e-9)
                
                if loss < 1e-6 and _verify(U, V, W, T, tol=1e-6):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    print(f"Found rank {rank} with loss {loss:.2e}")
                    break
    
    # Phase 2: If we found something, try to push lower
    if best_rank < schoolbook_rank:
        lower_ranks = list(range(best_rank - 1, max(45, best_rank - 15), -1))
        
        for rank in lower_ranks:
            if time.time() > deadline - 15:
                break
            
            for seed in range(3):
                if time.time() > deadline - 8:
                    break
                
                # Initialize near the previous solution by adding small random perturbations
                if rank < best_rank:
                    # Truncate or project
                    U, V, W = _structured_init_strassen(n, m, p, rank, seed=seed + rank * 100)
                else:
                    U, V, W = _structured_init_strassen(n, m, p, rank, seed=seed + rank * 100)
                
                U, V, W, loss = _refine_adaptive(U, V, W, T, max_iters=100, tol=1e-7, base_lr=0.005)
                
                if loss < 1e-5:
                    if time.time() < deadline - 5:
                        U, V, W, loss = _try_polish(U, V, W, T, max_iters=100, tol=1e-9)
                    
                    if loss < 1e-6 and _verify(U, V, W, T, tol=1e-6):
                        best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        print(f"Improved to rank {rank} with loss {loss:.2e}")
                        break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
