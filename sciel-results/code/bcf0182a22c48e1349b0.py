"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _check_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Convention: W[k*n + i, c]
    The tensor T[i,j,k] = 1 should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    """
    r = U.shape[1]
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    
    if not np.allclose(T, 1.0, atol=1e-9, rtol=1e-9):
        return False
    return True

def _make_schoolbook(n, m, p):
    """Create the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U.tolist(), V.tolist(), W.tolist()

def _project_half_integer(x):
    """Project to nearest half-integer, clipped to [-1, 1]."""
    x_rounded = np.round(x * 2) / 2
    return np.clip(x_rounded, -1, 1)

def _gradient_descent_step(U_r, V_r, W_r, n, m, p, lr=0.1):
    """
    One gradient descent step with projection to half-integers.
    U_r: (n, m, r), V_r: (m, p, r), W_r: (p, n, r)
    """
    # Compute current tensor
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    E = T - 1.0
    
    # Gradients
    grad_U = np.einsum('ijk,jkc,kic->ijc', E, V_r, W_r, optimize=True)
    grad_V = np.einsum('ijk,ijc,kic->jkc', E, U_r, W_r, optimize=True)
    grad_W = np.einsum('ijk,ijc,jkc->kic', E, U_r, V_r, optimize=True)
    
    # Update with projection
    U_new = _project_half_integer(U_r - lr * grad_U)
    V_new = _project_half_integer(V_r - lr * grad_V)
    W_new = _project_half_integer(W_r - lr * grad_W)
    
    return U_new, V_new, W_new

def _try_rank_with_gradient_descent(n, m, p, r, max_iter=200, seed=0, lr_start=0.1, lr_decay=0.999):
    """
    Try to find a rank-r decomposition using gradient descent from random initialization.
    Returns (U, V, W) as numpy arrays if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Random initialization with half-integer values
    # Use a mix of small random values and some structured patterns
    U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, r))
    V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, r))
    W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, r))
    
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    lr = lr_start
    
    for it in range(max_iter):
        # Check if we've converged
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        if np.allclose(T, 1.0, atol=1e-9, rtol=1e-9):
            U_final = U_r.reshape(n * m, r)
            V_final = V_r.reshape(m * p, r)
            W_final = W_r.reshape(p * n, r)
            if _check_decomposition(U_final, V_final, W_final, n, m, p):
                return U_final, V_final, W_final
        
        # Gradient descent step
        U_r, V_r, W_r = _gradient_descent_step(U_r, V_r, W_r, n, m, p, lr)
        lr *= lr_decay
    
    # Final check
    U_final = U_r.reshape(n * m, r)
    V_final = V_r.reshape(m * p, r)
    W_final = W_r.reshape(p, n, r).reshape(p * n, r)
    if _check_decomposition(U_final, V_final, W_final, n, m, p):
        return U_final, V_final, W_final
    return None

def _try_structured_init(n, m, p, r, seed=0):
    """
    Try structured initializations that might be closer to a solution.
    """
    rng = np.random.RandomState(seed)
    
    # Strategy 1: Initialize based on the schoolbook pattern, then perturb
    # Strategy 2: Use Kronecker product structures
    
    # For now, try a few different structured approaches
    results = []
    
    # Approach 1: Random with bias towards smaller values
    for attempt in range(3):
        U = rng.choice([-0.5, 0, 0.5], size=(n * m, r))
        V = rng.choice([-0.5, 0, 0.5], size=(m * p, r))
        W = rng.choice([-0.5, 0, 0.5], size=(p * n, r))
        
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        # Run a few gradient steps
        lr = 0.1
        for it in range(50):
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
            if np.allclose(T, 1.0, atol=1e-9, rtol=1e-9):
                break
            E = T - 1.0
            grad_U = np.einsum('ijk,jkc,kic->ijc', E, V_r, W_r, optimize=True)
            grad_V = np.einsum('ijk,ijc,kic->jkc', E, U_r, W_r, optimize=True)
            grad_W = np.einsum('ijk,ijc,jkc->kic', E, U_r, V_r, optimize=True)
            U_r = _project_half_integer(U_r - lr * grad_U)
            V_r = _project_half_integer(V_r - lr * grad_V)
            W_r = _project_half_integer(W_r - lr * grad_W)
            lr *= 0.995
        
        U_final = U_r.reshape(n * m, r)
        V_final = V_r.reshape(m * p, r)
        W_final = W_r.reshape(p * n, r)
        if _check_decomposition(U_final, V_final, W_final, n, m, p):
            return U_final, V_final, W_final
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix 
    by a 5 by 7 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    U, V, W = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Try ranks from lower bound up to schoolbook
    
    # Focus on ranks near the lower bound first, as they're most promising
    ranks_to_try = list(range(35, min(60, best_rank)))
    # Add some higher ranks as backup
    for r in [60, 70, 80, 90]:
        if r < best_rank and r not in ranks_to_try:
            ranks_to_try.append(r)
    ranks_to_try.sort()
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            break
        
        # Try multiple random seeds for this rank
        found = False
        for seed in range(30):
            if time.time() > deadline - 5:
                break
            
            # Try gradient descent
            result = _try_rank_with_gradient_descent(n, m, p, r, max_iter=300, seed=seed)
            if result is not None:
                U_f, V_f, W_f = result
                best_rank = r
                best_result = {
                    "U": U_f.tolist(),
                    "V": V_f.tolist(),
                    "W": W_f.tolist()
                }
                found = True
                break
        
        # Also try structured initialization
        if not found:
            result = _try_structured_init(n, m, p, r, seed=seed)
            if result is not None:
                U_f, V_f, W_f = result
                best_rank = r
                best_result = {
                    "U": U_f.tolist(),
                    "V": V_f.tolist(),
                    "W": W_f.tolist()
                }
    
    return best_result
# EVOLVE-BLOCK-END
