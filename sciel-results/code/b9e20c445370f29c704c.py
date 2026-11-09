"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return a valid rank n*m*p decomposition (the safe fallback)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    Condition: for all i,k,j,j': sum_c W[k*n+i,c]*U[i*m+j,c]*V[j'*p+k,c] = delta_{j,j'}
    """
    r = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0
                    for c in range(r):
                        s += W[k * n + i][c] * U[i * m + j][c] * V[jp * p + k][c]
                    expected = 1 if j == jp else 0
                    if abs(s - expected) > tol:
                        return False
    return True


def _objective_and_grad(U, V, W, n, m, p):
    """
    Compute the smooth least-squares objective and gradients.
    Objective: sum over all (i,k,j,j') of (sum_c W[k*n+i,c]*U[i*m+j,c]*V[j'*p+k,c] - delta_{j,j'})^2
    """
    r = U.shape[1]
    # Reshape for efficient computation
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    
    obj = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    target = 1.0 if j == jp else 0.0
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    residual = s - target
                    obj += residual * residual
                    # Gradient contributions
                    for c in range(r):
                        dU = 2 * residual * W[k * n + i, c] * V[jp * p + k, c]
                        dV = 2 * residual * W[k * n + i, c] * U[i * m + j, c]
                        dW = 2 * residual * U[i * m + j, c] * V[jp * p + k, c]
                        grad_U[i * m + j, c] += dU
                        grad_V[jp * p + k, c] += dV
                        grad_W[k * n + i, c] += dW
    
    return obj, grad_U, grad_V, grad_W


def _gradient_descent(U, V, W, n, m, p, lr=0.01, max_iter=2000, tol=1e-8, deadline=None):
    """
    Run gradient descent to minimize the objective.
    Returns (U, V, W, final_obj)
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    lr_current = lr
    prev_obj = float('inf')
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        obj, gU, gV, gW = _objective_and_grad(U, V, W, n, m, p)
        
        if abs(prev_obj - obj) < tol:
            break
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr_current *= 0.9
        
        U -= lr_current * gU
        V -= lr_current * gV
        W -= lr_current * gW
        
        prev_obj = obj
    
    obj, _, _, _ = _objective_and_grad(U, V, W, n, m, p)
    return U, V, W, obj


def _random_init(n, m, p, r, seed):
    """Generate random initialization."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, r) * 0.5
    V = rng.randn(m * p, r) * 0.5
    W = rng.randn(p * n, r) * 0.5
    return U, V, W


def _try_rank(n, m, p, r, deadline, n_restarts=10, max_iter=1500):
    """
    Try to find a valid decomposition of rank r using gradient descent.
    Returns (U, V, W, success) or (None, None, None, False)
    """
    best_obj = float('inf')
    best_U = best_V = best_W = None
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        seed = 12345 + restart * 7919
        U, V, W = _random_init(n, m, p, r, seed)
        
        U, V, W, obj = _gradient_descent(U, V, W, n, m, p, lr=0.005, max_iter=max_iter, deadline=deadline)
        
        if obj < best_obj:
            best_obj = obj
            best_U, best_V, best_W = U, V, W
        
        # Check if we found a valid solution
        if obj < 1e-6:
            # Verify with exact check
            U_list = [[float(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[float(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[float(W[i, c]) for c in range(r)] for i in range(p * n)]
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-5):
                return U_list, V_list, W_list, True
    
    if best_U is not None and best_obj < 1e-4:
        # Close enough, try to polish
        U_list = [[float(best_U[i, c]) for c in range(r)] for i in range(n * m)]
        V_list = [[float(best_V[i, c]) for c in range(r)] for i in range(m * p)]
        W_list = [[float(best_W[i, c]) for c in range(r)] for i in range(p * n)]
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-5):
            return U_list, V_list, W_list, True
    
    return None, None, None, False


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 105
    
    # Start with schoolbook as the best known valid solution
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound up to best_rank - 1
    # Use gradient-based search with multiple restarts
    candidate_ranks = list(range(lower_bound, best_rank))
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Determine how many restarts to use based on time remaining
        time_left = deadline - time.time()
        if time_left < 5:
            break
        
        # For lower ranks, try more restarts; for higher ranks, fewer
        if r <= 40:
            n_restarts = min(20, int(time_left / 2))
            max_iter = 1000
        elif r <= 60:
            n_restarts = min(15, int(time_left / 3))
            max_iter = 800
        else:
            n_restarts = min(10, int(time_left / 5))
            max_iter = 600
        
        if n_restarts < 1:
            break
        
        U, V, W, success = _try_rank(n, m, p, r, deadline, n_restarts, max_iter)
        
        if success:
            best_U, best_V, best_W = U, V, W
            best_rank = r
            # Continue searching for even lower rank
    
    # Return the best valid decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
