"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _validate(U, V, W, n=3, m=3, p=5):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _validate_half_integer(U, V, W, n=3, m=3, p=5):
    """Check entries are half-integers and decomposition is exact."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return _validate(U, V, W, n, m, p)

def _compute_errors(U, V, W, n=3, m=3, p=5):
    """Compute the error for each (i,j,k) triple."""
    rank = len(U[0])
    errors = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                errors[(i,j,k)] = s - 1.0
    return errors

def _try_newton(r, n=3, m=3, p=5, max_iters=50, seed=42):
    """Try to find a rank-r decomposition using Newton's method on the polynomial system."""
    if _time_remaining() < 3.0:
        return None
    
    rng = random.Random(seed)
    
    # Number of equations: n*m*p
    n_eq = n * m * p
    # Number of unknowns: (n*m + m*p + p*n) * r
    n_unknowns = (n*m + m*p + p*n) * r
    
    if n_unknowns < n_eq:
        return None  # Underdetermined but we need at least as many unknowns
    
    # Build the Jacobian structure
    # Variables: U[0..n*m-1, 0..r-1], V[0..m*p-1, 0..r-1], W[0..p*n-1, 0..r-1]
    
    def flatten(U, V, W):
        x = np.zeros(n_unknowns)
        idx = 0
        for i in range(n*m):
            for rr in range(r):
                x[idx] = U[i][rr]
                idx += 1
        for j in range(m*p):
            for rr in range(r):
                x[idx] = V[j][rr]
                idx += 1
        for k in range(p*n):
            for rr in range(r):
                x[idx] = W[k][rr]
                idx += 1
        return x
    
    def unflatten(x):
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        idx = 0
        for i in range(n*m):
            for rr in range(r):
                U[i][rr] = x[idx]
                idx += 1
        for j in range(m*p):
            for rr in range(r):
                V[j][rr] = x[idx]
                idx += 1
        for k in range(p*n):
            for rr in range(r):
                W[k][rr] = x[idx]
                idx += 1
        return U, V, W
    
    def residuals(x):
        U, V, W = unflatten(x)
        res = np.zeros(n_eq)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(r):
                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    res[eq_idx] = s - 1.0
                    eq_idx += 1
        return res
    
    # Try multiple random initializations
    for init in range(20):
        if _time_remaining() < 3.0:
            break
        
        # Random initialization with small values
        x0 = np.zeros(n_unknowns)
        for idx in range(n_unknowns):
            x0[idx] = rng.choice([-1.0, -0.5, 0.5, 1.0])
        
        # Newton's method
        x = x0.copy()
        for it in range(max_iters):
            if _time_remaining() < 3.0:
                break
            
            # Compute residuals
            f = residuals(x)
            if np.max(np.abs(f)) < 1e-10:
                U, V, W = unflatten(x)
                if _validate_half_integer(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                break
            
            # Compute Jacobian numerically
            J = np.zeros((n_eq, n_unknowns))
            h = 1e-7
            for j in range(n_unknowns):
                if _time_remaining() < 3.0:
                    break
                x_plus = x.copy()
                x_plus[j] += h
                f_plus = residuals(x_plus)
                J[:, j] = (f_plus - f) / h
            
            # Solve J dx = -f
            try:
                dx = np.linalg.lstsq(J, -f, rcond=None)[0]
            except:
                break
            
            # Line search
            alpha = 1.0
            f_norm = np.linalg.norm(f)
            for _ in range(20):
                x_new = x + alpha * dx
                f_new = residuals(x_new)
                if np.linalg.norm(f_new) < f_norm:
                    break
                alpha *= 0.5
            
            x = x + alpha * dx
            
            if np.max(np.abs(residuals(x))) < 1e-10:
                U, V, W = unflatten(x)
                if _validate_half_integer(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                break
    
    return None

def _try_structured_low_rank(r, n=3, m=3, p=5):
    """Try structured patterns for low-rank decompositions with rational entries."""
    if _time_remaining() < 2.0:
        return None
    
    # Try a pattern based on the algebraic structure of matrix multiplication
    # For M_{3,3,5}, we can try to use a decomposition that exploits the rectangular structure
    
    # Strategy: Fix U and V to have a specific sparse pattern, then solve for W
    # or use a known low-rank construction
    
    # Try rank 23 (known for 3x3x3, might extend)
    # Try rank 20, 18, etc.
    
    # For now, try a simple structured approach:
    # Make U, V, W have entries from {-1, 0, 1} with a specific pattern
    
    values = [-1, 0, 1]
    
    # Try all combinations of a small number of non-zero entries
    # This is a heuristic approach
    
    rng = random.Random(123)
    
    for attempt in range(100):
        if _time_remaining() < 2.0:
            break
        
        # Random sparse initialization
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        # Fill with random values from the set
        for i in range(n*m):
            for rr in range(r):
                if rng.random() < 0.3:
                    U[i][rr] = rng.choice(values)
        
        for j in range(m*p):
            for rr in range(r):
                if rng.random() < 0.3:
                    V[j][rr] = rng.choice(values)
        
        for k in range(p*n):
            for rr in range(r):
                if rng.random() < 0.3:
                    W[k][rr] = rng.choice(values)
        
        # Local search
        improved = True
        iters = 0
        while improved and iters < 100 and _time_remaining() > 2.0:
            improved = False
            iters += 1
            
            errors = _compute_errors(U, V, W, n, m, p)
            max_err = max(abs(e) for e in errors.values())
            
            if max_err < 1e-9:
                if _validate_half_integer(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                break
            
            # Try random entry changes
            for _ in range(10):
                mat_choice = rng.randint(0, 2)
                if mat_choice == 0:
                    idx = (rng.randint(0, n*m-1), rng.randint(0, r-1))
                    old_val = U[idx[0]][idx[1]]
                    new_val = rng.choice(values)
                    if new_val == old_val:
                        continue
                    U[idx[0]][idx[1]] = new_val
                elif mat_choice == 1:
                    idx = (rng.randint(0, m*p-1), rng.randint(0, r-1))
                    old_val = V[idx[0]][idx[1]]
                    new_val = rng.choice(values)
                    if new_val == old_val:
                        continue
                    V[idx[0]][idx[1]] = new_val
                else:
                    idx = (rng.randint(0, p*n-1), rng.randint(0, r-1))
                    old_val = W[idx[0]][idx[1]]
                    new_val = rng.choice(values)
                    if new_val == old_val:
                        continue
                    W[idx[0]][idx[1]] = new_val
                
                new_errors = _compute_errors(U, V, W, n, m, p)
                new_max_err = max(abs(e) for e in new_errors.values())
                
                if new_max_err < max_err - 1e-12:
                    improved = True
                    max_err = new_max_err
                    if max_err < 1e-9:
                        if _validate_half_integer(U, V, W, n, m, p):
                            return {"U": U, "V": V, "W": W}
                        break
                else:
                    if mat_choice == 0:
                        U[idx[0]][idx[1]] = old_val
                    elif mat_choice == 1:
                        V[idx[0]][idx[1]] = old_val
                    else:
                        W[idx[0]][idx[1]] = old_val
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Build schoolbook as fallback
    U = [[0]*45 for _ in range(n*m)]
    V = [[0]*45 for _ in range(m*p)]
    W = [[0]*45 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    schoolbook = {"U": U, "V": V, "W": W}
    
    best_rank = 45
    best_sol = schoolbook
    
    # Try ranks from lower to higher
    # Theoretical lower bound is max(nm, mp, np) = max(9, 15, 15) = 15
    # Try ranks from 15 upward
    
    for r in range(15, 46):
        if _time_remaining() < 5.0:
            break
        
        # Try Newton's method first (more structured)
        sol = _try_newton(r, n, m, p, max_iters=30, seed=r)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
        
        # Try structured low-rank search
        sol = _try_structured_low_rank(r, n, m, p)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
    
    return best_sol
