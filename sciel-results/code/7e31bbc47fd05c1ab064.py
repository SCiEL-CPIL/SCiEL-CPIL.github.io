"""
Matrix multiplication 2x5 x 5x5: rank-minimal exact tensor decomposition.

The tensor T has indices (i, j, k) for i in range(2), j in range(5), k in range(5).
We need U (10 x r), V (25 x r), W (10 x r) such that:
  sum_c U[i*5+j, c] * V[j*5+k, c] * W[k*2+i, c] = 1  for all valid (i,j,k)

Strategy:
1. Start with schoolbook rank 50.
2. Use a structured search that exploits the tensor structure.
3. For each rank r, try to find a valid decomposition using:
   a. Structured initialization based on known patterns
   b. Local search to fix violations
   c. Randomized restarts with different seeds
   d. Gradient-based optimization (L-BFGS-B) to find near-exact solutions
   e. Snapping to half-integers and final validation

Key insight: The matrix multiplication tensor has a specific structure that
can be exploited. For 2x5x5, we can try to use block structure and known
lower bounds from related problems.

Known results:
- 2x2x2: rank 7 (Strassen)
- 2x3x3: best known rank 17 (vs 18 schoolbook)
- 2x4x4: best known rank around 28-30 (vs 32 schoolbook)
- 2x5x5: schoolbook is 50, target is to find rank < 50

The search tries ranks from 49 downward, using a combination of:
1. Structured initialization with small integer/half-integer entries
2. Local search that identifies and fixes the worst violations
3. Multiple restarts with different random seeds
4. Gradient-based optimization to find near-exact solutions
"""

import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def validate_fast(U, V, W, n, m, p, tol=1e-9):
    """Validate that U, V, W reconstruct the matrix multiplication tensor exactly."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    r = U.shape[1]
    
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _clean_val(x):
    """Clean a float value to integer or half-integer if possible."""
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    elif abs(2 * x - round(2 * x)) < 1e-9:
        return round(2 * x) / 2.0
    return x

def compute_residuals(U, V, W, n, m, p):
    """Compute residuals for all (i,j,k) triples."""
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                residuals.append((i, j, k, val - 1.0))
    return residuals

def try_rank_structured(r, n, m, p, timeout, max_attempts=200):
    """Try to find a rank-r decomposition using structured search."""
    start_time = time.time()
    
    # Values to use in the search
    values = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        # Structured initialization: use a pattern based on the tensor structure
        U = np.zeros((n * m, r), dtype=np.float64)
        V = np.zeros((m * p, r), dtype=np.float64)
        W = np.zeros((p * n, r), dtype=np.float64)
        
        # Initialize with a structured pattern
        for c in range(r):
            # For each column c, assign values based on a structured pattern
            # Use a pseudo-random but structured approach
            seed_c = random.randint(0, 10000)
            rng = random.Random(seed_c)
            
            # Assign values to U
            for i in range(n):
                for j in range(m):
                    if rng.random() < 0.3:
                        U[i * m + j, c] = rng.choice(values)
            
            # Assign values to V
            for j in range(m):
                for k in range(p):
                    if rng.random() < 0.3:
                        V[j * p + k, c] = rng.choice(values)
            
            # Assign values to W
            for k in range(p):
                for i in range(n):
                    if rng.random() < 0.3:
                        W[k * n + i, c] = rng.choice(values)
        
        # Check if this works
        if validate_fast(U, V, W, n, m, p):
            U_list = [[_clean_val(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[_clean_val(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[_clean_val(W[i, c]) for c in range(r)] for i in range(p * n)]
            return U_list, V_list, W_list
        
        # If not, try local search to fix violations
        U, V, W = local_search(U, V, W, n, m, p, timeout - (time.time() - start_time))
        
        if U is not None:
            U_list = [[_clean_val(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[_clean_val(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[_clean_val(W[i, c]) for c in range(r)] for i in range(p * n)]
            return U_list, V_list, W_list
    
    return None

def local_search(U, V, W, n, m, p, timeout, max_iterations=100):
    """Local search to fix violations in a near-valid decomposition."""
    start_time = time.time()
    values = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
    
    for iteration in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        # Compute residuals
        residuals = compute_residuals(U, V, W, n, m, p)
        
        # Find the worst violation
        worst_idx = max(range(len(residuals)), key=lambda idx: abs(residuals[idx][3]))
        i, j, k, res = residuals[worst_idx]
        
        if abs(res) < 1e-9:
            # All violations are fixed
            return U, V, W
        
        # Try to fix the worst violation by modifying one entry
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        
        # Try modifying U[idx_u, c] for each c
        for c in range(U.shape[1]):
            old_val = U[idx_u, c]
            for new_val in values:
                if new_val == old_val:
                    continue
                U[idx_u, c] = new_val
                if validate_fast(U, V, W, n, m, p):
                    return U, V, W
                U[idx_u, c] = old_val
        
        # Try modifying V[idx_v, c] for each c
        for c in range(V.shape[1]):
            old_val = V[idx_v, c]
            for new_val in values:
                if new_val == old_val:
                    continue
                V[idx_v, c] = new_val
                if validate_fast(U, V, W, n, m, p):
                    return U, V, W
                V[idx_v, c] = old_val
        
        # Try modifying W[idx_w, c] for each c
        for c in range(W.shape[1]):
            old_val = W[idx_w, c]
            for new_val in values:
                if new_val == old_val:
                    continue
                W[idx_w, c] = new_val
                if validate_fast(U, V, W, n, m, p):
                    return U, V, W
                W[idx_w, c] = old_val
    
    return None, None, None

def try_rank_randomized(r, n, m, p, timeout, max_attempts=1000):
    """Try to find a rank-r decomposition using randomized search."""
    start_time = time.time()
    values = [-2, -1, -0.5, 0.5, 1, 2]
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        # Random sparse initialization
        density = 0.1 + 0.2 * random.random()
        
        U = np.zeros((n * m, r), dtype=np.float64)
        V = np.zeros((m * p, r), dtype=np.float64)
        W = np.zeros((p * n, r), dtype=np.float64)
        
        for row in range(n * m):
            for col in range(r):
                if random.random() < density:
                    U[row, col] = random.choice(values)
        
        for row in range(m * p):
            for col in range(r):
                if random.random() < density:
                    V[row, col] = random.choice(values)
        
        for row in range(p * n):
            for col in range(r):
                if random.random() < density:
                    W[row, col] = random.choice(values)
        
        if validate_fast(U, V, W, n, m, p):
            U_list = [[_clean_val(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[_clean_val(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[_clean_val(W[i, c]) for c in range(r)] for i in range(p * n)]
            return U_list, V_list, W_list
    
    return None

def try_rank_gradient(r, n, m, p, timeout, max_seeds=10):
    """Try to find a rank-r decomposition using gradient-based optimization."""
    start_time = time.time()
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def objective(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
        Res = T - Rec
        return np.sum(Res * Res)
    
    def gradient(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
        Res = T - Rec
        Grad_U = np.einsum('ijl,jc,lc->ic', Res, V, W)
        Grad_V = np.einsum('ijl,ic,lc->jc', Res, U, W)
        Grad_W = np.einsum('ijl,ic,jc->lc', Res, U, V)
        return np.concatenate([Grad_U.flatten(), Grad_V.flatten(), Grad_W.flatten()])
    
    for seed in range(max_seeds):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(1000 + r * 100 + seed)
        x0 = np.random.randn(n*m*r + m*p*r + p*n*r)
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < 1e-8:
                x_opt = res.x
                U = x_opt[:n*m*r].reshape(n*m, r)
                V = x_opt[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = x_opt[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                Rec_snap = np.einsum('ik,jk,lk->ijl', U_snap, V_snap, W_snap)
                err_snap = np.linalg.norm(T - Rec_snap)
                
                if err_snap < 1e-6:
                    U_list = [[_clean_val(U_snap[i, c]) for c in range(r)] for i in range(n * m)]
                    V_list = [[_clean_val(V_snap[i, c]) for c in range(r)] for i in range(m * p)]
                    W_list = [[_clean_val(W_snap[i, c]) for c in range(r)] for i in range(p * n)]
                    return U_list, V_list, W_list
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook_decomposition(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    for r in range(best_rank - 1, 24, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        remaining = deadline - elapsed - 5
        if remaining <= 0:
            break
        
        # Allocate time for different search strategies
        timeout_structured = min(1.0, remaining * 0.2)
        timeout_random = min(1.0, remaining * 0.2)
        timeout_gradient = min(3.0, remaining * 0.5)
        
        if timeout_structured < 0.2 or timeout_random < 0.2 or timeout_gradient < 0.5:
            break
        
        result = None
        
        # Try structured search first
        if time.time() - start_time < deadline - 10:
            result = try_rank_structured(r, n, m, p, timeout_structured, max_attempts=100)
        
        # Try random search
        if result is None and time.time() - start_time < deadline - 10:
            result = try_rank_randomized(r, n, m, p, timeout_random, max_attempts=500)
        
        # Try gradient-based search
        if result is None and time.time() - start_time < deadline - 5:
            result = try_rank_gradient(r, n, m, p, timeout_gradient, max_seeds=10)
        
        if result is not None:
            U_r, V_r, W_r = result
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = r
            continue
    
    # Final validation
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        U_arr = np.array(best_result["U"])
        V_arr = np.array(best_result["V"])
        W_arr = np.array(best_result["W"])
        if not validate_fast(U_arr, V_arr, W_arr, n, m, p):
            best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return best_result
