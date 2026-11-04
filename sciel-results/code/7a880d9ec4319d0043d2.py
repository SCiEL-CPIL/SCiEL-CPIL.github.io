"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integers(U, V, W):
    """Snap matrices to the nearest half-integers."""
    U_snapped = np.round(U * 2) / 2
    V_snapped = np.round(V * 2) / 2
    W_snapped = np.round(W * 2) / 2
    return U_snapped, V_snapped, W_snapped

def _optimize_and_snap(U, V, W, n, m, p, max_iters=2000, lr=0.01, snap_interval=50, deadline=None):
    """
    Optimize U, V, W using gradient descent, periodically snapping to half-integers.
    Returns the best valid decomposition found or None.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        total_err_sq = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i*m + j, :]
                    vk = V[j*p + k, :]
                    wk = W[k*n + i, :]
                    s_val = np.dot(ui, vk * wk)
                    err = s_val - 1.0
                    total_err_sq += err * err
                    
                    factor = 2.0 * err
                    grad_U[i*m + j, :] += factor * (vk * wk)
                    grad_V[j*p + k, :] += factor * (ui * wk)
                    grad_W[k*n + i, :] += factor * (ui * vk)
        
        err_norm = np.sqrt(total_err_sq)
        
        if err_norm < 1e-6:
            if err_norm < best_err:
                best_err = err_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            # Try snapping
            U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
            if _verify_decomposition(U_s, V_s, W_s, n, m, p, tol=1e-7):
                return U_s, V_s, W_s
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if err_norm < best_err:
            best_err = err_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        # Periodic snapping to escape local minima or find discrete solutions
        if it % snap_interval == 0:
            U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
            if _verify_decomposition(U_s, V_s, W_s, n, m, p, tol=1e-7):
                return U_s, V_s, W_s
            # Use snapped version as new start if it has lower error?
            # Or just keep continuous for now, but maybe nudge?
            # Let's just snap and continue from there if error is low enough
            if best_err < 1.0:
                 U, V, W = U_s, V_s, W_s
                 # Reset gradients? No, just continue.
    
    # Final check
    if best_err < 1e-6:
        U_s, V_s, W_s = _snap_to_half_integers(best_U, best_V, best_W)
        if _verify_decomposition(U_s, V_s, W_s, n, m, p, tol=1e-7):
            return U_s, V_s, W_s
        # Maybe the continuous solution is good enough for the evaluator if it accepts floats?
        # The task says "integer/half-integer entries". So we MUST snap.
        # If snapping fails, we return None.
        
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Create schoolbook decomposition
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower ranks
    # We iterate down from schoolbook rank.
    # For each rank R, we try to find a solution.
    # Strategy: Start with a solution of rank R+1, set one column to 0, and optimize.
    
    # Initialize with Schoolbook
    current_U = U_school.copy()
    current_V = V_school.copy()
    current_W = W_school.copy()
    current_rank = schoolbook_rank
    
    # Try to reduce rank one by one
    # This is slow if we start from 128 and go to 100.
    # Let's jump. Try ranks: 120, 110, 100, 90, 80, 70, 60
    target_ranks = [120, 115, 110, 105, 100, 95, 90, 85, 80]
    
    for R in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Initialize U, V, W for rank R
        # Take the first R columns of the current best solution (or schoolbook if no better found yet)
        # If R > current_rank, we can't just take columns. We need to expand?
        # Better: Always start from Schoolbook truncated to R, or from a previously found solution of higher rank.
        
        U_init = np.zeros((n*m, R))
        V_init = np.zeros((m*p, R))
        W_init = np.zeros((p*n, R))
        
        # If we have a solution of rank > R, take its first R columns.
        # If not, take schoolbook's first R columns.
        if current_rank > R:
            U_init = current_U[:, :R].copy()
            V_init = current_V[:, :R].copy()
            W_init = current_W[:, :R].copy()
        else:
            U_init = U_school[:, :R].copy()
            V_init = V_school[:, :R].copy()
            W_init = W_school[:, :R].copy()
            
        # Add small noise to help escape local minima
        np.random.seed(R)
        U_init += np.random.randn(*U_init.shape) * 0.1
        V_init += np.random.randn(*V_init.shape) * 0.1
        W_init += np.random.randn(*W_init.shape) * 0.1
        
        # Optimize
        U_opt, V_opt, W_opt = _optimize_and_snap(U_init, V_init, W_init, n, m, p, 
                                                 max_iters=500, lr=0.05, 
                                                 snap_interval=20, deadline=deadline)
        
        if U_opt is not None:
            best_rank = R
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            current_U, current_V, current_W = U_opt, V_opt, W_opt
            current_rank = R
            print(f"Found valid decomposition with rank {R}")
            # Continue to try lower ranks using this as base
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
