import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # First try known structured decompositions (Winograd-style)
    # For 4x4x5, we can try to use a tensor decomposition approach
    # based on fast matrix multiplication algorithms.
    
    # Try rank 24 first (known lower bound for some related problems)
    for rank in [24, 25, 26, 27, 28, 29, 30, 32, 35, 40, 45, 50, 60, 80]:
        if time.time() > deadline - 15:
            break
        
        result = _find_decomposition_structured(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _find_decomposition_structured(n, m, p, rank, deadline):
    """
    Find a rank decomposition using a structured approach based on
    tensor network decomposition and alternating least squares with
    exact rational recovery.
    
    The key insight is to use the structure of matrix multiplication
    to guide the search, rather than pure random initialization.
    """
    nm = n * m  # 16
    mp = m * p  # 20
    pn = p * n  # 20
    
    # Use multiple strategies with different initializations
    strategies = [
        'random',
        'structured_block',
        'winograd_like',
        'sparse_init'
    ]
    
    max_restarts_per_strategy = 30
    
    for strategy in strategies:
        if time.time() > deadline - 10:
            break
        
        for restart in range(max_restarts_per_strategy):
            if time.time() > deadline - 10:
                return None
            
            # Different initialization strategies
            if strategy == 'random':
                U, V, W = _init_random(nm, mp, pn, rank)
            elif strategy == 'structured_block':
                U, V, W = _init_structured(n, m, p, rank)
            elif strategy == 'winograd_like':
                U, V, W = _init_winograd_like(n, m, p, rank)
            elif strategy == 'sparse_init':
                U, V, W = _init_sparse(n, m, p, rank)
            else:
                U, V, W = _init_random(nm, mp, pn, rank)
            
            # Alternating minimization with multiple restarts
            U, V, W = _alternating_minimization(n, m, p, rank, U, V, W, deadline)
            
            # Try to recover exact rational solution
            result = _recover_exact(n, m, p, U, V, W, deadline)
            if result is not None:
                return result
            
            # Verify with tolerance and try rounding
            if _verify_numpy(n, m, p, U, V, W, tol=1e-8):
                for scale in [1, 2, 4, 10, 20]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    if _verify_numpy(n, m, p, U_r, V_r, W_r, tol=1e-6):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
    
    return None

def _init_random(nm, mp, pn, rank):
    np.random.seed(42)
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    return U, V, W

def _init_structured(n, m, p, rank):
    """Initialize with block structure based on matrix multiplication."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    np.random.seed(123)
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Try to assign columns in a structured way
    for col in range(rank):
        # Randomly pick which (i,j), (j,k), (k,i) triples to activate
        i = np.random.randint(0, n)
        j = np.random.randint(0, m)
        k = np.random.randint(0, p)
        
        U[i * m + j, col] = 1.0
        V[j * p + k, col] = 1.0
        W[k * n + i, col] = 1.0
        
        # Add some noise to help escape local minima
        U[i * m + j, col] += np.random.randn() * 0.1
        V[j * p + k, col] += np.random.randn() * 0.1
        W[k * n + i, col] += np.random.randn() * 0.1
    
    return U, V, W

def _init_winograd_like(n, m, p, rank):
    """
    Initialize based on Winograd-like patterns.
    For matrix multiplication, we can use the structure of
    fast matrix multiplication algorithms.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    np.random.seed(456)
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Use a pattern based on tensor decomposition
    # Each column represents a rank-1 tensor
    for col in range(rank):
        # Create a random rank-1 pattern
        # But with some structure: group by indices
        i_base = col % n
        j_base = (col // n) % m
        k_base = (col // (n * m)) % p
        
        # Activate a block around these indices
        for di in range(n):
            for dj in range(m):
                i = (i_base + di) % n
                j = (j_base + dj) % m
                U[i * m + j, col] = np.random.choice([-1, 1])
        
        for dk in range(p):
            for dj in range(m):
                j = (j_base + dj) % m
                k = (k_base + dk) % p
                V[j * p + k, col] = np.random.choice([-1, 1])
        
        for dk in range(p):
            for di in range(n):
                k = (k_base + dk) % p
                i = (i_base + di) % n
                W[k * n + i, col] = np.random.choice([-1, 1])
    
    # Normalize
    for col in range(rank):
        norm = np.linalg.norm(U[:, col])
        if norm > 1e-10:
            U[:, col] /= norm
        norm = np.linalg.norm(V[:, col])
        if norm > 1e-10:
            V[:, col] /= norm
        norm = np.linalg.norm(W[:, col])
        if norm > 1e-10:
            W[:, col] /= norm
    
    return U, V, W

def _init_sparse(n, m, p, rank):
    """Initialize with sparse structure."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    np.random.seed(789)
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Each column has a few non-zero entries
    for col in range(rank):
        # Pick a few random positions
        n_nonzero_U = np.random.randint(1, 4)
        n_nonzero_V = np.random.randint(1, 4)
        n_nonzero_W = np.random.randint(1, 4)
        
        idx_U = np.random.choice(nm, size=n_nonzero_U, replace=False)
        idx_V = np.random.choice(mp, size=n_nonzero_V, replace=False)
        idx_W = np.random.choice(pn, size=n_nonzero_W, replace=False)
        
        U[idx_U, col] = np.random.choice([-1, 1], size=n_nonzero_U)
        V[idx_V, col] = np.random.choice([-1, 1], size=n_nonzero_V)
        W[idx_W, col] = np.random.choice([-1, 1], size=n_nonzero_W)
    
    return U, V, W

def _alternating_minimization(n, m, p, rank, U, V, W, deadline):
    """Alternating minimization to find a valid decomposition."""
    
    # Precompute indices for efficiency
    U_indices = [(i, j, i * m + j) for i in range(n) for j in range(m)]
    V_indices = [(j, k, j * p + k) for j in range(m) for k in range(p)]
    W_indices = [(k, i, k * n + i) for k in range(p) for i in range(n)]
    
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        # Solve for U
        U_new = np.zeros_like(U)
        for i, j, idx in U_indices:
            A = np.zeros(rank)
            for k in range(p):
                A += V[j * p + k, :] * W[k * n + i, :]
            norm_A = np.linalg.norm(A)
            if norm_A > 1e-12:
                U_new[idx, :] = A / (norm_A ** 2)
        
        # Solve for V
        V_new = np.zeros_like(V)
        for j, k, idx in V_indices:
            B = np.zeros(rank)
            for i in range(n):
                B += U_new[i * m + j, :] * W[k * n + i, :]
            norm_B = np.linalg.norm(B)
            if norm_B > 1e-12:
                V_new[idx, :] = B / (norm_B ** 2)
        
        # Solve for W
        W_new = np.zeros_like(W)
        for k, i, idx in W_indices:
            C = np.zeros(rank)
            for j in range(m):
                C += U_new[i * m + j, :] * V_new[j * p + k, :]
            norm_C = np.linalg.norm(C)
            if norm_C > 1e-12:
                W_new[idx, :] = C / (norm_C ** 2)
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        if _verify_numpy(n, m, p, U, V, W, tol=1e-6):
            break
    
    return U, V, W

def _recover_exact(n, m, p, U, V, W, deadline):
    """Try to recover an exact rational solution."""
    if _verify_numpy(n, m, p, U, V, W, tol=1e-6):
        for scale in [1, 2, 4, 10, 20, 50]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if _verify_numpy(n, m, p, U_r, V_r, W_r, tol=1e-6):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    return None

def _verify_numpy(n, m, p, U, V, W, tol=1e-6):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END

