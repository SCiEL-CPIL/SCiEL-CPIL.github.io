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
    """Compute the error vector for all tensor entries."""
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

def _try_rank(r, n=3, m=3, p=5, max_iters=5000, seed=42):
    """Try to find a rank-r decomposition using random restarts and local search."""
    if _time_remaining() < 2.0:
        return None
    
    rng = random.Random(seed)
    
    # Try different value sets
    value_sets = [
        [-2, -1, -0.5, 0, 0.5, 1, 2],
        [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
        [-1, 0, 1],
        [-2, -1, 0, 1, 2],
    ]
    
    best_solution = None
    
    for vs_idx, values in enumerate(value_sets):
        if _time_remaining() < 3.0:
            break
        
        n_restarts = max(10, min(100, int(200 * _time_remaining() / max(1, 15-r))))
        
        for restart in range(n_restarts):
            if _time_remaining() < 3.0:
                break
            
            # Random initialization with values from the set
            U = [[rng.choice(values) for _ in range(r)] for _ in range(n*m)]
            V = [[rng.choice(values) for _ in range(r)] for _ in range(m*p)]
            W = [[rng.choice(values) for _ in range(r)] for _ in range(p*n)]
            
            # Local search: try flipping individual entries
            improved = True
            iters = 0
            while improved and iters < max_iters and _time_remaining() > 3.0:
                improved = False
                iters += 1
                
                # Compute current error
                errors = _compute_errors(U, V, W, n, m, p)
                max_err = max(abs(e) for e in errors.values())
                if max_err < 1e-9:
                    if _validate_half_integer(U, V, W, n, m, p):
                        return {"U": U, "V": V, "W": W}
                    break
                
                # Try random entry changes
                for _ in range(20):
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
                    
                    # Check if error decreased
                    new_errors = _compute_errors(U, V, W, n, m, p)
                    new_max_err = max(abs(e) for e in new_errors.values())
                    
                    if new_max_err < max_err - 1e-12:
                        improved = True
                        max_err = new_max_err
                        errors = new_errors
                        if max_err < 1e-9:
                            if _validate_half_integer(U, V, W, n, m, p):
                                return {"U": U, "V": V, "W": W}
                            break
                    else:
                        # Revert
                        if mat_choice == 0:
                            U[idx[0]][idx[1]] = old_val
                        elif mat_choice == 1:
                            V[idx[0]][idx[1]] = old_val
                        else:
                            W[idx[0]][idx[1]] = old_val
            
            if max_err < 1e-9:
                if _validate_half_integer(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    return None

def _try_structural(r, n=3, m=3, p=5):
    """Try structural patterns for low-rank decompositions."""
    if _time_remaining() < 2.0:
        return None
    
    # Pattern: some entries of U, V, W are fixed to 0 or 1
    # Try sparse patterns
    
    for mask_u in range(1 << (n*m)):
        if _time_remaining() < 2.0:
            break
        for mask_v in range(1 << (m*p)):
            if _time_remaining() < 2.0:
                break
            for mask_w in range(1 << (p*n)):
                if _time_remaining() < 2.0:
                    break
                
                # Count active entries
                au = bin(mask_u).count('1')
                av = bin(mask_v).count('1')
                aw = bin(mask_w).count('1')
                
                if au > r or av > r or aw > r:
                    continue
                
                # Build sparse U, V, W
                U = [[0]*r for _ in range(n*m)]
                V = [[0]*r for _ in range(m*p)]
                W = [[0]*r for _ in range(p*n)]
                
                col_u = 0
                for i in range(n*m):
                    if mask_u & (1 << i):
                        U[i][col_u] = 1
                        col_u += 1
                
                col_v = 0
                for j in range(m*p):
                    if mask_v & (1 << j):
                        V[j][col_v] = 1
                        col_v += 1
                
                col_w = 0
                for k in range(p*n):
                    if mask_w & (1 << k):
                        W[k][col_w] = 1
                        col_w += 1
                
                if _validate(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    return None

def _try_systematic_search(r, n=3, m=3, p=5):
    """Systematic search over small value sets with optimized local search."""
    if _time_remaining() < 5.0:
        return None
    
    # Use numpy for faster error computation
    U_np = np.zeros((n*m, r))
    V_np = np.zeros((m*p, r))
    W_np = np.zeros((p*n, r))
    
    # Precompute index mappings
    u_indices = [(i*m+j, i, j) for i in range(n) for j in range(m)]
    v_indices = [(j*p+k, j, k) for j in range(m) for k in range(p)]
    w_indices = [(k*n+i, k, i) for k in range(p) for i in range(n)]
    
    def compute_max_err():
        """Compute max error using numpy."""
        max_e = 0.0
        for (ui, i, j) in u_indices:
            for (vj, j2, k) in v_indices:
                if j2 != j:
                    continue
                (wi, k2, i2) = (k*n+i, k, i)
                s = np.dot(U_np[ui], V_np[vj] * W_np[wi])
                e = abs(s - 1.0)
                if e > max_e:
                    max_e = e
        return max_e
    
    # Try a focused search with values in {-1, 0, 1}
    values = [-1, 0, 1]
    
    # Start from a structured initialization
    # Set up the schoolbook pattern as a starting point if r >= 45
    # For smaller r, use random init
    
    rng = random.Random(12345 + r)
    
    n_restarts = 50
    for restart in range(n_restarts):
        if _time_remaining() < 5.0:
            return None
        
        # Initialize
        for i in range(n*m):
            for c in range(r):
                U_np[i, c] = rng.choice(values)
        for j in range(m*p):
            for c in range(r):
                V_np[j, c] = rng.choice(values)
        for k in range(p*n):
            for c in range(r):
                W_np[k, c] = rng.choice(values)
        
        # Local search
        improved = True
        iters = 0
        max_iters = 1000
        while improved and iters < max_iters and _time_remaining() > 5.0:
            improved = False
            iters += 1
            
            max_err = compute_max_err()
            if max_err < 1e-9:
                # Convert to lists and validate
                U_list = U_np.tolist()
                V_list = V_np.tolist()
                W_list = W_np.tolist()
                if _validate_half_integer(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                break
            
            # Try random moves
            for _ in range(30):
                if _time_remaining() < 5.0:
                    break
                
                mat_choice = rng.randint(0, 2)
                if mat_choice == 0:
                    row = rng.randint(0, n*m-1)
                    col = rng.randint(0, r-1)
                    old = U_np[row, col]
                    new = rng.choice(values)
                    if new == old:
                        continue
                    U_np[row, col] = new
                elif mat_choice == 1:
                    row = rng.randint(0, m*p-1)
                    col = rng.randint(0, r-1)
                    old = V_np[row, col]
                    new = rng.choice(values)
                    if new == old:
                        continue
                    V_np[row, col] = new
                else:
                    row = rng.randint(0, p*n-1)
                    col = rng.randint(0, r-1)
                    old = W_np[row, col]
                    new = rng.choice(values)
                    if new == old:
                        continue
                    W_np[row, col] = new
                
                new_err = compute_max_err()
                
                if new_err < max_err - 1e-12:
                    improved = True
                    max_err = new_err
                    if max_err < 1e-9:
                        U_list = U_np.tolist()
                        V_list = V_np.tolist()
                        W_list = W_np.tolist()
                        if _validate_half_integer(U_list, V_list, W_list, n, m, p):
                            return {"U": U_list, "V": V_list, "W": W_list}
                        break
                else:
                    # Revert
                    if mat_choice == 0:
                        U_np[row, col] = old
                    elif mat_choice == 1:
                        V_np[row, col] = old
                    else:
                        W_np[row, col] = old
        
        if max_err < 1e-9:
            U_list = U_np.tolist()
            V_list = V_np.tolist()
            W_list = W_np.tolist()
            if _validate_half_integer(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
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
    
    # Try ranks from 14 upward
    for r in range(14, 46):
        if _time_remaining() < 5.0:
            break
        
        # First try systematic search with numpy acceleration
        sol = _try_systematic_search(r, n, m, p)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
        
        # Then try random/local search
        sol = _try_rank(r, n, m, p, max_iters=2000, seed=r)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
        
        # Try with different seeds
        for s in [100+r, 200+r, 300+r]:
            if _time_remaining() < 5.0:
                break
            sol = _try_rank(r, n, m, p, max_iters=1000, seed=s)
            if sol is not None:
                best_rank = r
                best_sol = sol
                break
        if sol is not None:
            break
    
    return best_sol
