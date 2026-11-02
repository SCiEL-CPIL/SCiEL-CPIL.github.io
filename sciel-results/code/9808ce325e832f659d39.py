"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) -> indices (i,j)
    # V shape: (m*p, rank) -> indices (j,k)  
    # W shape: (p*n, rank) -> indices (k,i)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must equal 1
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = 9, but actual rank is likely higher
    # Let's try ranks from 10 up to 23
    
    for rank in range(10, 24):
        if time.time() > deadline - 10:
            break
            
        result = _try_rank_advanced(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_advanced(n, m, p, rank, deadline, max_iters=2000, tol=1e-12, n_restarts=100):
    """Try to find a rank-r decomposition via improved alternating minimization."""
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            return None
            
        # Random initialization with better structure
        U, V, W = _initialize(n, m, p, rank, restart)
        
        success = _alternating_minimize_improved(n, m, p, rank, U, V, W, max_iters, tol, deadline)
        if success:
            # Verify and clean up
            U, V, W = _clean_solution(n, m, p, rank, U, V, W)
            if _verify(n, m, p, U, V, W, tol=1e-10):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _initialize(n, m, p, rank, seed):
    """Initialize U, V, W with structured random values."""
    np.random.seed(seed)
    
    # Try different initialization strategies
    if seed % 3 == 0:
        # Standard random
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
    elif seed % 3 == 1:
        # Sparse random
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        # Make some entries zero
        for _ in range(rank // 2):
            U[np.random.randint(n*m), np.random.randint(rank)] = 0
            V[np.random.randint(m*p), np.random.randint(rank)] = 0
            W[np.random.randint(p*n), np.random.randint(rank)] = 0
    else:
        # Block-structured
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for c in range(rank):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            U[i*m+j, c] = np.random.choice([-1, 1]) * np.random.rand()
            V[j*p+k, c] = np.random.choice([-1, 1]) * np.random.rand()
            W[k*n+i, c] = np.random.choice([-1, 1]) * np.random.rand()
    
    # Normalize
    U /= (np.linalg.norm(U) + 1e-10)
    V /= (np.linalg.norm(V) + 1e-10)
    W /= (np.linalg.norm(W) + 1e-10)
    
    return U, V, W

def _alternating_minimize_improved(n, m, p, rank, U, V, W, max_iters, tol, deadline):
    """Improved alternating least squares."""
    
    prev_err = float('inf')
    plateau_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            return False
            
        # Solve for U given V, W
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                except:
                    return False
        
        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, rank))
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                except:
                    return False
        
        # Solve for W given U, V
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                except:
                    return False
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        
        if err < tol:
            return True
        
        # Plateau detection
        if err > prev_err * 0.99:
            plateau_count += 1
        else:
            plateau_count = 0
        prev_err = err
        
        if plateau_count > 100:
            break
    
    return False

def _clean_solution(n, m, p, rank, U, V, W):
    """Try to clean up the solution to have exact integer/half-integer values."""
    # Try rounding to nearest 0.5
    U_clean = np.round(U * 2) / 2
    V_clean = np.round(V * 2) / 2
    W_clean = np.round(W * 2) / 2
    
    # If that doesn't work, try integer rounding
    if not _verify(n, m, p, U_clean, V_clean, W_clean, tol=1e-8):
        U_clean = np.round(U)
        V_clean = np.round(V)
        W_clean = np.round(W)
    
    return U_clean, V_clean, W_clean

def _verify(n, m, p, U, V, W, tol=1e-10):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
