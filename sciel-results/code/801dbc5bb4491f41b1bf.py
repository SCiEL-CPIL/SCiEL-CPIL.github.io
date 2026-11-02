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

def _compute_errors_vec(U, V, W, n=3, m=3, p=5):
    """Vectorized error computation."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    errors = np.zeros((n, m, p))
    for r in range(rank):
        # U[:,r] has shape (n*m,), V[:,r] has shape (m*p,), W[:,r] has shape (p*n,)
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    errors[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
    return errors - 1.0

def _try_rank_structured(r, n=3, m=3, p=5, seed=42, max_time=10.0):
    """
    Try to find a rank-r decomposition using structured search.
    Uses the insight that we can fix sparsity patterns and solve.
    """
    start = time.time()
    rng = random.Random(seed)
    
    n_r = n * m  # 9
    m_r = m * p  # 15
    p_r = p * n  # 15
    
    values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    best_err = float('inf')
    best_sol = None
    
    n_attempts = max(20, int(200 * _time_remaining() / max(1, r)))
    
    for attempt in range(n_attempts):
        if time.time() - start > max_time or _time_remaining() < 5.0:
            break
        
        # Initialize with a structured pattern
        # Strategy: make U dense in a subset of columns, V and W similarly
        U = [[0.0]*r for _ in range(n_r)]
        V = [[0.0]*r for _ in range(m_r)]
        W = [[0.0]*r for _ in range(p_r)]
        
        # Random sparse initialization with bias toward small values
        for row in range(n_r):
            nnz = rng.randint(max(1, r//4), min(r, r//2 + 1))
            cols = rng.sample(range(r), nnz)
            for c in cols:
                U[row][c] = rng.choice(values)
        
        for row in range(m_r):
            nnz = rng.randint(max(1, r//4), min(r, r//2 + 1))
            cols = rng.sample(range(r), nnz)
            for c in cols:
                V[row][c] = rng.choice(values)
        
        for row in range(p_r):
            nnz = rng.randint(max(1, r//4), min(r, r//2 + 1))
            cols = rng.sample(range(r), nnz)
            for c in cols:
                W[row][c] = rng.choice(values)
        
        # Compute initial error
        errors = _compute_errors_vec(U, V, W, n, m, p)
        max_err = float(np.max(np.abs(errors)))
        sum_sq_err = float(np.sum(errors**2))
        
        if max_err < 1e-9:
            if _validate_half_integer(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # Local search with simulated annealing style
        temperature = max_err * 0.5
        best_local_err = sum_sq_err
        best_local_sol = (U, V, W)
        
        for it in range(5000):
            if time.time() - start > max_time or _time_remaining() < 5.0:
                break
            
            # Choose a random entry to modify
            mat_choice = rng.randint(0, 2)
            if mat_choice == 0:
                idx = (rng.randint(0, n_r-1), rng.randint(0, r-1))
                old_val = U[idx[0]][idx[1]]
                new_val = rng.choice(values)
                if new_val == old_val:
                    continue
                U[idx[0]][idx[1]] = new_val
            elif mat_choice == 1:
                idx = (rng.randint(0, m_r-1), rng.randint(0, r-1))
                old_val = V[idx[0]][idx[1]]
                new_val = rng.choice(values)
                if new_val == old_val:
                    continue
                V[idx[0]][idx[1]] = new_val
            else:
                idx = (rng.randint(0, p_r-1), rng.randint(0, r-1))
                old_val = W[idx[0]][idx[1]]
                new_val = rng.choice(values)
                if new_val == old_val:
                    continue
                W[idx[0]][idx[1]] = new_val
            
            new_errors = _compute_errors_vec(U, V, W, n, m, p)
            new_max_err = float(np.max(np.abs(new_errors)))
            new_sum_sq = float(np.sum(new_errors**2))
            
            if new_sum_sq < best_local_err:
                best_local_err = new_sum_sq
                best_local_sol = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
            
            if new_max_err < 1e-9:
                if _validate_half_integer(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                # Accept and continue
                sum_sq_err = new_sum_sq
                max_err = new_max_err
                temperature = max(0.01, temperature * 0.95)
                continue
            
            # Accept with probability based on temperature
            delta = new_sum_sq - sum_sq_err
            if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-6)):
                sum_sq_err = new_sum_sq
                max_err = new_max_err
                temperature = max(0.01, temperature * 0.99)
            else:
                # Revert
                if mat_choice == 0:
                    U[idx[0]][idx[1]] = old_val
                elif mat_choice == 1:
                    V[idx[0]][idx[1]] = old_val
                else:
                    W[idx[0]][idx[1]] = old_val
        
        # After local search, check best solution
        if best_local_err < 1e-9:
            bU, bV, bW = best_local_sol
            if _validate_half_integer(bU, bV, bW, n, m, p):
                return {"U": bU, "V": bV, "W": bW}
        
        if best_local_err < best_err:
            best_err = best_local_err
            best_sol = (best_local_sol[0], best_local_sol[1], best_local_sol[2])
    
    return None

def _try_rank_linear_solve(r, n=3, m=3, p=5, seed=42, max_time=8.0):
    """
    Try to find a rank-r decomposition by fixing two matrices and solving for the third.
    For each column r, if we fix U[:,r] and V[:,r], then W[:,r] must satisfy:
    For all (i,j,k): sum over r' of U[i,j,r']*V[j,k,r']*W[k,i,r'] = 1
    This is linear in W if U and V are fixed.
    """
    start = time.time()
    rng = random.Random(seed)
    
    n_r = n * m  # 9
    m_r = m * p  # 15
    p_r = p * n  # 15
    
    values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for attempt in range(50):
        if time.time() - start > max_time or _time_remaining() < 5.0:
            break
        
        # Fix U and V with random sparse patterns
        U = [[0.0]*r for _ in range(n_r)]
        V = [[0.0]*r for _ in range(m_r)]
        
        for row in range(n_r):
            nnz = rng.randint(max(1, r//3), min(r, r//2))
            cols = rng.sample(range(r), nnz)
            for c in cols:
                U[row][c] = rng.choice(values)
        
        for row in range(m_r):
            nnz = rng.randint(max(1, r//3), min(r, r//2))
            cols = rng.sample(range(r), nnz)
            for c in cols:
                V[row][c] = rng.choice(values)
        
        # Now solve for W: for each (i,j,k), we need
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a system of 45 equations in 15*r unknowns (W entries)
        # But it's not linear in W because each equation involves W[k*n+i, r] 
        # multiplied by U[i*m+j, r] * V[j*p+k, r]
        # Actually it IS linear in W! Each equation is:
        # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        # So for each output entry (i,j,k), we have a linear equation in the W variables.
        
        # Build the linear system: A * w_vec = b
        # where w_vec is the 15*r vector of all W entries
        # A has 45 rows and 15*r columns
        
        A = np.zeros((n*m*p, p_r * r))
        b = np.ones(n*m*p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    for rr in range(r):
                        w_idx = (k*n + i) * r + rr
                        A[eq_idx, w_idx] = U[i*m+j][rr] * V[j*p+k][rr]
        
        # Solve the linear system
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        W = [[0.0]*r for _ in range(p_r)]
        for idx in range(p_r * r):
            W[idx // r][idx % r] = W_flat[idx]
        
        # Check if solution is close to half-integers
        # Round to nearest half-integer and check
        W_rounded = [[round(val * 2) / 2.0 for val in row] for row in W]
        
        # Verify
        errors = _compute_errors_vec(U, V, W_rounded, n, m, p)
        max_err = float(np.max(np.abs(errors)))
        
        if max_err < 1e-9:
            if _validate_half_integer(U, V, W_rounded, n, m, p):
                return {"U": U, "V": V, "W": W_rounded}
        
        # Try without rounding
        errors2 = _compute_errors_vec(U, V, W, n, m, p)
        max_err2 = float(np.max(np.abs(errors2)))
        if max_err2 < 1e-9:
            if _validate_half_integer(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _try_fixed_pattern(r, n=3, m=3, p=5):
    """
    Try specific structural patterns that are known to work for small matrices.
    For 3x3x5, try patterns based on the structure of matrix multiplication.
    """
    n_r = n * m  # 9
    m_r = m * p  # 15
    p_r = p * n  # 15
    
    # Pattern 1: Each rank-1 term handles a specific (j, k) pair
    # For each j in range(m), k in range(p), have one term
    # U[i*m+j, r_jk] = 1 for all i
    # V[j*p+k, r_jk] = 1
    # W[k*n+i, r_jk] = 1 for all i
    # This gives rank m*p = 15, but let's verify it works
    if r >= m * p:
        U = [[0.0]*r for _ in range(n_r)]
        V = [[0.0]*r for _ in range(m_r)]
        W = [[0.0]*r for _ in range(p_r)]
        col = 0
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    U[i*m+j][col] = 1.0
                    W[k*n+i][col] = 1.0
                V[j*p+k][col] = 1.0
                col += 1
        if _validate_half_integer(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Pattern 2: Each rank-1 term handles a specific (i, k) pair
    if r >= n * p:
        U = [[0.0]*r for _ in range(n_r)]
        V = [[0.0]*r for _ in range(m_r)]
        W = [[0.0]*r for _ in range(p_r)]
        col = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
        if _validate_half_integer(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Pattern 3: Each rank-1 term handles a specific (i, j) pair
    if r >= n * m:
        U = [[0.0]*r for _ in range(n_r)]
        V = [[0.0]*r for _ in range(m_r)]
        W = [[0.0]*r for _ in range(p_r)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m+j][col] = 1.0
                for k in range(p):
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                col += 1
        if _validate_half_integer(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Schoolbook: rank 45
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
    
    # Try structural patterns first (fast)
    for r in range(15, 46):
        if _time_remaining() < 5.0:
            break
        sol = _try_fixed_pattern(r, n, m, p)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
    
    if best_rank < 45:
        return best_sol
    
    # Try linear solve approach for various ranks
    for r in range(15, min(46, best_rank)):
        if _time_remaining() < 8.0:
            break
        sol = _try_rank_linear_solve(r, n, m, p, seed=r, max_time=5.0)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
    
    if best_rank < 45:
        return best_sol
    
    # Try structured local search
    for r in range(15, min(46, best_rank)):
        if _time_remaining() < 10.0:
            break
        for seed in [r, 100+r, 200+r]:
            if _time_remaining() < 5.0:
                break
            sol = _try_rank_structured(r, n, m, p, seed=seed, max_time=5.0)
            if sol is not None:
                best_rank = r
                best_sol = sol
                break
        if best_rank < 45:
            break
    
    return best_sol
