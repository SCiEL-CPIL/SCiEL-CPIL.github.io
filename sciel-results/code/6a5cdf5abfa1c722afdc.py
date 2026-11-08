"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    
    # Check all other entries are zero
    T_check = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_check[a, b, c] = 0
    if np.max(np.abs(T_check)) > tol:
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _build_target_tensor(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _structured_init(n, m, p, target_rank, rng, mode=0):
    """Generate structured initializations exploiting the tensor structure."""
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    if mode == 0:
        # Random initialization with scaling
        scale = 1.0 / (target_rank ** 0.25)
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale
    elif mode == 1:
        # Structured: exploit that middle dimension is small (m=3)
        # For each j in 0..m-1, create a block of columns
        cols_per_j = target_rank // m
        for j in range(m):
            start = j * cols_per_j
            end = start + cols_per_j
            # U rows: i*m + j for i in 0..n-1
            for i in range(n):
                U[i * m + j, start:end] = rng.randn(cols_per_j) * 0.5
            # V rows: j*p + k for k in 0..p-1
            for k in range(p):
                V[j * p + k, start:end] = rng.randn(cols_per_j) * 0.5
            # W rows: k*n + i for k in 0..p-1, i in 0..n-1
            for k in range(p):
                for i in range(n):
                    W[k * n + i, start:end] = rng.randn(cols_per_j) * 0.5
        # Add some cross-terms
        remaining = target_rank - m * cols_per_j
        if remaining > 0:
            U[:, m * cols_per_j:] = rng.randn(n * m, remaining) * 0.3
            V[:, m * cols_per_j:] = rng.randn(m * p, remaining) * 0.3
            W[:, m * cols_per_j:] = rng.randn(p * n, remaining) * 0.3
    elif mode == 2:
        # Sparse-ish initialization with random support
        U = rng.randn(n * m, target_rank) * 0.3
        V = rng.randn(m * p, target_rank) * 0.3
        W = rng.randn(p * n, target_rank) * 0.3
        # Zero out some entries to create structure
        for _ in range(target_rank // 3):
            col = rng.randint(target_rank)
            U[rng.choice(n * m, size=n * m // 3, replace=False), col] = 0
            V[rng.choice(m * p, size=m * p // 3, replace=False), col] = 0
            W[rng.choice(p * n, size=p * n // 3, replace=False), col] = 0
    elif mode == 3:
        # Initialize from perturbed schoolbook
        U0, V0, W0 = _schoolbook(n, m, p)
        idx = rng.choice(n * m * p, size=target_rank, replace=False)
        U = U0[:, idx] + rng.randn(n * m, target_rank) * 0.2
        V = V0[:, idx] + rng.randn(m * p, target_rank) * 0.2
        W = W0[:, idx] + rng.randn(p * n, target_rank) * 0.2
    elif mode == 4:
        # Kronecker-like structure: U = A ⊗ B pattern
        # For n=2, m=3, p=7: try to exploit the factorization
        A = rng.randn(n, m) * 0.5
        B = rng.randn(m, p) * 0.5
        # Create rank-m decomposition of A⊗B-like structure
        for j in range(m):
            start = j * (target_rank // m)
            end = start + (target_rank // m)
            U[j:end + j * (target_rank // m), :] = 0  # This is getting complex
            # Simpler: just use structured random
        U = rng.randn(n * m, target_rank) * 0.4
        V = rng.randn(m * p, target_rank) * 0.4
        W = rng.randn(p * n, target_rank) * 0.4
        # Ensure some rows have larger values
        for i in range(n):
            for j in range(m):
                U[i * m + j, :] *= 1.5
    else:
        # Default random
        scale = 1.0 / (target_rank ** 0.3)
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale
    
    return U, V, W

def _gradient_descent_momentum(U, V, W, T_target, n, m, p, max_iter=2000, tol=1e-10, seed=42, momentum=0.9):
    """Gradient descent with momentum and adaptive learning rate."""
    rng = np.random.RandomState(seed)
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    # Momentum buffers
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.02
    patience = 0
    best_patience_err = float('inf')
    
    for it in range(max_iter):
        # Compute current tensor
        T_curr = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T_curr - T_target
        err = np.sum(residual ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < best_patience_err - 1e-8:
                best_patience_err = err
                patience = 0
            else:
                patience += 1
        
        if best_err < tol:
            break
        
        if patience > 80:
            # Restart with perturbation from best
            U = best_U + rng.randn(*best_U.shape) * 0.05
            V = best_V + rng.randn(*best_V.shape) * 0.05
            W = best_W + rng.randn(*best_W.shape) * 0.05
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            patience = 0
            lr = max(0.005, lr * 0.7)
            continue
        
        # Gradient computation
        grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
        grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
        grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
        
        # Scale gradients by the "magnitude" of each factor
        U_norm = np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
        V_norm = np.linalg.norm(V, axis=0, keepdims=True) + 1e-12
        W_norm = np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
        
        grad_U = grad_U / (V_norm * W_norm)
        grad_V = grad_V / (U_norm * W_norm)
        grad_W = grad_W / (U_norm * V_norm)
        
        # Momentum update
        vU = momentum * vU + grad_U
        vV = momentum * vV + grad_V
        vW = momentum * vW + grad_W
        
        # Update
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
        # Decay learning rate
        lr *= 0.9995
    
    return best_U, best_V, best_W, best_err

def _try_rank(n, m, p, target_rank, T_target, max_seeds=5, max_iter=1500, tol=1e-10, deadline=None):
    """Try to find a decomposition of given rank."""
    base_seed = 12345
    modes = [0, 1, 2, 3, 4]
    
    for seed_idx in range(max_seeds):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        seed = base_seed + seed_idx * 1000
        rng = np.random.RandomState(seed)
        mode = modes[seed_idx % len(modes)]
        
        U, V, W = _structured_init(n, m, p, target_rank, rng, mode=mode)
        
        U, V, W, err = _gradient_descent_momentum(U, V, W, T_target, n, m, p, 
                                                   max_iter=max_iter, tol=tol, seed=seed)
        
        if err < tol:
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=tol):
                return U, V, W, True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p  # 42
    
    # Build target tensor once
    T_target = _build_target_tensor(n, m, p)
    
    # Try to find better decompositions using local search
    # Try ranks from low to high, with more seeds for lower ranks
    for target_rank in range(28, 42):
        if time.time() > deadline - 15:
            break
        
        # More seeds for lower ranks (harder to find)
        max_seeds = 6 if target_rank < 32 else 4
        max_iter = 1200 if target_rank < 32 else 800
        
        U, V, W, success = _try_rank(n, m, p, target_rank, T_target, 
                                       max_seeds=max_seeds, max_iter=max_iter, 
                                       tol=1e-10, deadline=deadline)
        
        if success:
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = target_rank
            # Continue trying lower ranks
            continue
    
    # If we found a good rank, try to polish it with more iterations
    if best_rank < n * m * p and time.time() < deadline - 10:
        U_curr = np.array(best_result["U"])
        V_curr = np.array(best_result["V"])
        W_curr = np.array(best_result["W"])
        curr_rank = U_curr.shape[1]
        
        # Try to reduce rank further
        for target_rank in range(max(24, curr_rank - 4), curr_rank):
            if time.time() > deadline - 5:
                break
            
            U, V, W, success = _try_rank(n, m, p, target_rank, T_target,
                                           max_seeds=3, max_iter=600,
                                           tol=1e-10, deadline=deadline)
            if success:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
    
    return best_result
# EVOLVE-BLOCK-END
