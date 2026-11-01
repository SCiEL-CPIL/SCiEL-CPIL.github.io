"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    
    # The tensor T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # First, create the schoolbook solution as fallback (Rank 12)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Try to find lower rank decompositions
    # Known lower bound is 6, known rank is 8.
    # We try 8 first as it is the known minimal rank.
    deadline = time.time() + 110  # Leave 10s margin for return
    
    for r in [8, 7, 6]:
        if time.time() > deadline:
            break
        
        result = _find_decomposition(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            best_rank = r
            break  # We found the minimum rank, stop
    
    return best_result

def _find_decomposition(n, m, p, r, deadline):
    """
    Try to find a rank-r decomposition using vectorized operations and ALS.
    """
    np.random.seed(42)
    
    # Precompute index mappings for vectorized access
    # For each (i,j,k), we need:
    # U_idx = i*m + j, V_idx = j*p + k, W_idx = k*n + i
    
    max_restarts = 30
    
    for attempt in range(max_restarts):
        if time.time() > deadline:
            return None
        
        # Initialize with structured random values
        # Use different initialization strategies
        if attempt % 3 == 0:
            # Standard random
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
        elif attempt % 3 == 1:
            # Sparse initialization
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                # Assign random non-zero entries
                for _ in range(3):
                    U[np.random.randint(n * m), c] = np.random.choice([-1, 1])
                    V[np.random.randint(m * p), c] = np.random.choice([-1, 1])
                    W[np.random.randint(p * n), c] = np.random.choice([-1, 1])
        else:
            # Small random
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
        
        # Use ALS (Alternating Least Squares)
        U, V, W = _als_refine(n, m, p, r, U, V, W, deadline)
        
        # Check if we have a valid decomposition
        if _check_and_round(n, m, p, r, U, V, W):
            return _to_lists(n, m, p, r, U, V, W)
        
        # Try gradient descent polish
        U, V, W = _gd_polish(n, m, p, r, U, V, W, deadline)
        
        if _check_and_round(n, m, p, r, U, V, W):
            return _to_lists(n, m, p, r, U, V, W)
    
    return None

def _als_refine(n, m, p, r, U, V, W, deadline):
    """
    Alternating Least Squares refinement.
    """
    # Precompute all (i,j,k) indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for it in range(100):
        if time.time() > deadline:
            break
        
        # Update U: for each row i*m+j, solve least squares
        for i in range(n):
            for j in range(m):
                if time.time() > deadline:
                    return U, V, W
                # Collect equations where this (i,j) appears
                A_rows = []
                b_vals = []
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    # Equation: sum_c U[idx_u,c] * V[idx_v,c] * W[idx_w,c] = 1
                    # For fixed U[idx_u,:], this is linear in U[idx_u,:]
                    # V[idx_v,:] * W[idx_w,:] gives coefficients
                    coeff = V[idx_v, :] * W[idx_w, :]
                    A_rows.append(coeff)
                    b_vals.append(1.0)
                
                if A_rows:
                    A = np.array(A_rows)  # shape (p, r)
                    b = np.array(b_vals)  # shape (p,)
                    # Solve A @ U[idx_u,:] = b
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = sol
        
        # Update V: for each row j*p+k, solve least squares
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    return U, V, W
                A_rows = []
                b_vals = []
                for i in range(n):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    coeff = U[idx_u, :] * W[idx_w, :]
                    A_rows.append(coeff)
                    b_vals.append(1.0)
                
                if A_rows:
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = sol
        
        # Update W: for each row k*n+i, solve least squares
        for k in range(p):
            for i in range(n):
                if time.time() > deadline:
                    return U, V, W
                A_rows = []
                b_vals = []
                for j in range(m):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    coeff = U[idx_u, :] * V[idx_v, :]
                    A_rows.append(coeff)
                    b_vals.append(1.0)
                
                if A_rows:
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = sol
    
    return U, V, W

def _gd_polish(n, m, p, r, U, V, W, deadline):
    """
    Gradient descent polish with vectorized operations.
    """
    max_iters = 200
    lr = 0.01
    momentum = 0.9
    
    vel_U = np.zeros_like(U)
    vel_V = np.zeros_like(V)
    vel_W = np.zeros_like(W)
    
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute values vectorized
        vals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vals[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
        
        loss = np.sum((vals - 1.0)**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if loss < 1e-8:
                break
        
        # Compute gradients vectorized
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    err = vals[i, j, k] - 1.0
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    grad_U[idx_u, :] += 2 * err * V[idx_v, :] * W[idx_w, :]
                    grad_V[idx_v, :] += 2 * err * U[idx_u, :] * W[idx_w, :]
                    grad_W[idx_w, :] += 2 * err * U[idx_u, :] * V[idx_v, :]
        
        # Momentum update
        vel_U = momentum * vel_U - lr * grad_U
        vel_V = momentum * vel_V - lr * grad_V
        vel_W = momentum * vel_W - lr * grad_W
        
        U += vel_U
        V += vel_V
        W += vel_W
        
        # Adaptive learning rate
        if loss > best_loss * 1.1:
            lr *= 0.9
        else:
            lr = min(lr * 1.02, 0.05)
        
        # Prevent explosion
        if np.max(np.abs(U)) > 50 or np.max(np.abs(V)) > 50 or np.max(np.abs(W)) > 50:
            U = best_U.copy()
            V = best_V.copy()
            W = best_W.copy()
            vel_U = np.zeros_like(U)
            vel_V = np.zeros_like(V)
            vel_W = np.zeros_like(W)
            lr *= 0.5
    
    return best_U, best_V, best_W

def _check_and_round(n, m, p, r, U, V, W):
    """
    Check if decomposition is valid and round to half-integers.
    Returns True if valid.
    """
    # Check current loss
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                loss += (val - 1.0)**2
    
    if loss > 1e-6:
        return False
    
    # Try rounding to half-integers
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    loss_rounded = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                loss_rounded += (val - 1.0)**2
    
    if loss_rounded < 1e-9:
        U[:] = U_rounded
        V[:] = V_rounded
        W[:] = W_rounded
        return True
    
    # Try rounding to integers
    U_rounded = np.round(U)
    V_rounded = np.round(V)
    W_rounded = np.round(W)
    
    loss_rounded = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                loss_rounded += (val - 1.0)**2
    
    if loss_rounded < 1e-9:
        U[:] = U_rounded
        V[:] = V_rounded
        W[:] = W_rounded
        return True
    
    return False

def _to_lists(n, m, p, r, U, V, W):
    """
    Convert numpy arrays to lists.
    """
    U_list = [[float(U[i, c]) for c in range(r)] for i in range(n*m)]
    V_list = [[float(V[i, c]) for c in range(r)] for i in range(m*p)]
    W_list = [[float(W[i, c]) for c in range(r)] for i in range(p*n)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
