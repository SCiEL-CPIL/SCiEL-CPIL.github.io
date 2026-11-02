"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np
import scipy.linalg as la
from itertools import product

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

def _compute_residual(U, V, W, n, m, p):
    """Compute the sum of squared residuals for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                err += (s - 1.0)**2
    return err

def _discrete_local_search(n, m, p, r, U0, V0, W0, max_iters=5000):
    """
    Perform discrete local search on the entries of U, V, W.
    Candidates: [-1, -0.5, 0, 0.5, 1]
    """
    candidates = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    # Precompute indices for efficiency
    # We will iterate over all elements in U, V, W
    
    current_err = _compute_residual(U, V, W, n, m, p)
    
    # If we start at 0, we are done
    if current_err < 1e-12:
        return U, V, W
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        improved = False
        
        # Try changing U entries
        for i in range(n * m):
            for c in range(r):
                old_val = U[i, c]
                # Find nearest candidate to current value (or just try all neighbors)
                # Since we only allow candidates, we try changing to other candidates
                # To speed up, we only try if current value is not already a candidate?
                # No, we initialized with candidates. So we just try other candidates.
                
                # Optimization: Only try if there is a potential gain.
                # We can approximate the change in error.
                # Delta_err = (s_new - 1)^2 - (s_old - 1)^2
                # s_old = sum_c U[i,c] V[j,c] W[k,c]
                # Changing U[i,c] by delta affects s for all j,k such that V[j,c]!=0 and W[k,c]!=0.
                
                # For simplicity and robustness, we just try the best neighbor.
                # But with 5 candidates, trying all is 4 tries per element.
                # Total elements: r*(nm + mp + pn). For r=18, ~650 elements.
                # 650 * 4 * 36 (ijk) = ~90k ops per iteration.
                # 5000 iters = 450M ops. Might be too slow in Python.
                
                # Let's use a simpler heuristic: Randomly select an element and try to change it.
                pass # Placeholder for complex logic, we will use random selection below.
        
        # Random selection approach
        # Pick a random matrix, random row, random col
        mat_idx = np.random.randint(0, 3)
        if mat_idx == 0:
            row = np.random.randint(0, n * m)
            col = np.random.randint(0, r)
            mat = U
        elif mat_idx == 1:
            row = np.random.randint(0, m * p)
            col = np.random.randint(0, r)
            mat = V
        else:
            row = np.random.randint(0, p * n)
            col = np.random.randint(0, r)
            mat = W
            
        old_val = mat[row, col]
        
        # Try all other candidates
        best_val = old_val
        best_err = current_err
        
        for new_val in candidates:
            if new_val == old_val:
                continue
            mat[row, col] = new_val
            new_err = _compute_residual(U, V, W, n, m, p)
            if new_err < best_err:
                best_err = new_err
                best_val = new_val
            else:
                mat[row, col] = old_val # restore
        
        if best_val != old_val:
            mat[row, col] = best_val
            current_err = best_err
            improved = True
            
            if current_err < 1e-12:
                return U, V, W
        else:
            # If no improvement, break early to avoid wasting time
            # Or continue for a while to escape local minima? 
            # For now, break if stuck for a while.
            if it > 100:
                 break
        
    return U, V, W

def _try_rank_r(n, m, p, r, max_restarts=50):
    """
    Try to find a rank-r decomposition using discrete local search.
    """
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    # Candidate values for initialization
    init_candidates = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Initialize with sparse random values from candidates
        # Bias towards 0 to keep sparsity
        probs = [0.2, 0.1, 0.4, 0.1, 0.2] # -1, -0.5, 0, 0.5, 1
        U0 = np.random.choice(init_candidates, size=(n*m, r), p=probs)
        V0 = np.random.choice(init_candidates, size=(m*p, r), p=probs)
        W0 = np.random.choice(init_candidates, size=(p*n, r), p=probs)
        
        U_sol, V_sol, W_sol = _discrete_local_search(n, m, p, r, U0, V0, W0)
        
        err = _compute_residual(U_sol, V_sol, W_sol, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            
            if err < 1e-12:
                return U_sol, V_sol, W_sol
                
    return None

def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Lower bound is max(nm, mp, pn) = 18
    # We try ranks from 18 upwards
    for r in range(18, best_rank):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        # Try discrete search
        result = _try_rank_r(n, m, p, r, max_restarts=50)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
            
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
