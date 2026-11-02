"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np
from itertools import product
import scipy.optimize as optimize

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matmul tensor.
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

def _try_rank_reduction(n, m, p, target_rank, max_restarts=50, time_budget=10.0):
    """
    Try to find a rank-r decomposition for n x m x p matmul tensor.
    Uses warm starts from schoolbook solution with random perturbations.
    """
    n_eq = n * m * p
    n_vars = (n*m + m*p + p*n) * target_rank
    
    def objective(params):
        U = params[:n*m*target_rank].reshape(n*m, target_rank)
        V = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    residuals.append(s - 1.0)
        return residuals
    
    best_U, best_V, best_W = None, None, None
    best_loss = float('inf')
    
    # Get schoolbook solution for warm starting
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > time_budget:
            break
            
        # Create a warm start by taking the first 'target_rank' columns of schoolbook
        # and adding some noise
        U0 = U_school[:, :target_rank].copy()
        V0 = V_school[:, :target_rank].copy()
        W0 = W_school[:, :target_rank].copy()
        
        # Add small noise to escape local minima
        noise_scale = 0.1 * (1 + restart % 10)
        U0 += np.random.randn(n*m, target_rank) * noise_scale
        V0 += np.random.randn(m*p, target_rank) * noise_scale
        W0 += np.random.randn(p*n, target_rank) * noise_scale
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = optimize.least_squares(objective, x0, method='trf', max_nfev=200)
        except Exception:
            continue
            
        if res.cost < best_loss:
            best_loss = res.cost
            U_sol = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_sol = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            # Try snapping to half-integers
            U_snap = _snap_to_half_integers(U_sol)
            V_snap = _snap_to_half_integers(V_sol)
            W_snap = _snap_to_half_integers(W_sol)
            
            ok, _ = _verify_decomposition(U_snap, V_snap, W_snap, n, m, p)
            if ok:
                return U_snap, V_snap, W_snap
            
            # Also try integer snapping
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
    
    # Try to find a lower rank solution
    # Start from rank 35 and go down to 30
    for r in range(35, 29, -1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        result = _try_rank_reduction(n, m, p, r, max_restarts=20, time_budget=TIME_LIMIT - 15)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
            break  # Found a lower rank, stop searching for even lower
    
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
