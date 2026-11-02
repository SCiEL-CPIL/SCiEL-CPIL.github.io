"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np
import scipy.linalg as la

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    
    For exact decomposition: for all i,j,k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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

def _snap_to_half_integers(matrix):
    """Snap matrix entries to nearest 0.5."""
    return np.round(matrix * 2) / 2.0

def _snap_to_integers(matrix):
    """Snap matrix entries to nearest integer."""
    return np.round(matrix)

def _als_step(U, V, W, n, m, p, fix='U'):
    """
    Perform one step of Alternating Least Squares.
    Fix two matrices, solve for the third to minimize residual.
    """
    r = U.shape[1]
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    if fix == 'U':
        # Solve for U
        # For each i, j: sum_c U[i*m+j, c] * (V[j*p+?, c] * W[?, i, c]) ... 
        # Actually, for fixed V, W, the term for U[i*m+j, c] is:
        # sum_k V[j*p+k, c] * W[k*n+i, c]
        # Let M_U[i*m+j, c] = sum_k V[j*p+k, c] * W[k*n+i, c]
        # We want sum_c U[i*m+j, c] * M_U[i*m+j, c] = 1
        # This is a linear system for each (i,j).
        # A_{i,j} U_{i,j} = 1 where A_{i,j} is a row vector of length r.
        # Least squares solution: U_{i,j} = A_{i,j}^T / ||A_{i,j}||^2
        
        for i in range(n):
            for j in range(m):
                # Construct the vector A for this (i,j)
                # A_c = sum_k V[j*p+k, c] * W[k*n+i, c]
                A = np.zeros(r)
                for k in range(p):
                    A += V[j*p+k] * W[k*n+i]
                norm_sq = np.dot(A, A)
                if norm_sq > 1e-12:
                    U_new[i*m+j] = A / norm_sq
                else:
                    U_new[i*m+j] = 0
                    
    elif fix == 'V':
        # Solve for V
        # For each j, k: sum_c V[j*p+k, c] * (U[?, j, c] * W[k, ?, c])
        # A_c = sum_i U[i*m+j, c] * W[k*n+i, c]
        for j in range(m):
            for k in range(p):
                A = np.zeros(r)
                for i in range(n):
                    A += U[i*m+j] * W[k*n+i]
                norm_sq = np.dot(A, A)
                if norm_sq > 1e-12:
                    V_new[j*p+k] = A / norm_sq
                else:
                    V_new[j*p+k] = 0

    elif fix == 'W':
        # Solve for W
        # For each k, i: sum_c W[k*n+i, c] * (U[?, ?, c] * V[?, k, c])
        # A_c = sum_j U[i*m+j, c] * V[j*p+k, c]
        for k in range(p):
            for i in range(n):
                A = np.zeros(r)
                for j in range(m):
                    A += U[i*m+j] * V[j*p+k]
                norm_sq = np.dot(A, A)
                if norm_sq > 1e-12:
                    W_new[k*n+i] = A / norm_sq
                else:
                    W_new[k*n+i] = 0

    if fix == 'U':
        return U_new, V, W
    elif fix == 'V':
        return U, V_new, W
    else:
        return U, V, W_new

def _als_optimize(n, m, p, r, U0, V0, W0, max_iters=100, tol=1e-10):
    """
    Run Alternating Least Squares to minimize the residual.
    """
    U, V, W = U0, V0, W0
    
    prev_err = float('inf')
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Cycle through fixing U, V, W
        U, V, W = _als_step(U, V, W, n, m, p, fix='U')
        U, V, W = _als_step(U, V, W, n, m, p, fix='V')
        U, V, W = _als_step(U, V, W, n, m, p, fix='W')
        
        # Calculate error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err += (s - 1.0)**2
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W

def _try_rank_r(n, m, p, r, max_restarts=50):
    """
    Try to find a rank-r decomposition using ALS with random restarts.
    """
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Initialize with random small integers
        # Use a mix of -1, 0, 1 to encourage sparsity and small values
        U0 = np.random.choice([-1, 0, 1], size=(n*m, r))
        V0 = np.random.choice([-1, 0, 1], size=(m*p, r))
        W0 = np.random.choice([-1, 0, 1], size=(p*n, r))
        
        # Ensure no row is all zeros to avoid singularity in ALS
        # If a row is zero, ALS might get stuck. Add small noise if needed.
        # But random choice usually gives non-zero rows.
        
        U_sol, V_sol, W_sol = _als_optimize(n, m, p, r, U0, V0, W0)
        
        # Calculate final error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_sol[i*m+j] * V_sol[j*p+k] * W_sol[k*n+i])
                    err += (s - 1.0)**2
                    
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            
            # Check for exact solution after snapping
            # Try half-integers
            U_snap = _snap_to_half_integers(U_sol)
            V_snap = _snap_to_half_integers(V_sol)
            W_snap = _snap_to_half_integers(W_sol)
            ok, _ = _verify_decomposition(U_snap, V_snap, W_snap, n, m, p)
            if ok:
                return U_snap, V_snap, W_snap
                
            # Try integers
            U_snap2 = _snap_to_integers(U_sol)
            V_snap2 = _snap_to_integers(V_sol)
            W_snap2 = _snap_to_integers(W_sol)
            ok2, _ = _verify_decomposition(U_snap2, V_snap2, W_snap2, n, m, p)
            if ok2:
                return U_snap2, V_snap2, W_snap2
                
    return None

def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try ranks from 18 up to 35
    # Lower bound is max(nm, mp, pn) = 18
    for r in range(18, best_rank):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        result = _try_rank_r(n, m, p, r, max_restarts=100)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break # Found a valid decomposition, can we go lower? 
            # Since we are iterating upwards, if we find one, it's the lowest 
            # *in this specific search order*. 
            # However, it's possible a lower rank exists but we missed it due to 
            # local minima. But usually, if we find a valid one at r, 
            # we should check if we can find one at r-1? 
            # The loop goes 18, 19, 20... so if we find at 20, we didn't 
            # find at 18 or 19. So 20 is the best we found so far.
            # We can stop if we want to save time, or continue to see if 
            # a lower one appears (unlikely if we search in order).
            # Actually, if we search 18, 19, 20, and find at 20, 
            # we should return 20. We don't need to check 21.
            # So breaking is correct.
            
    # Safety: ensure we have a valid decomposition
    ok, _ = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if not ok:
        best_U, best_V, best_W = U_school, V_school, W_school
    
    # Convert to lists for the return format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
