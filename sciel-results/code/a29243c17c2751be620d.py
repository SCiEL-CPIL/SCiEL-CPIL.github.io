"""Tensor decomposition search for matrix multiplication (3,3,7)."""
import random
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
DEADLINE = time.time() + 130  # safety margin before 150s kill

def _idx_W(k, i, n):
    return k * n + i

def _idx_U(i, j, m):
    return i * m + j

def _idx_V(j, k, p):
    return j * p + k

def _tensor_rank_error(U, V, W, n, m, p):
    """Compute max absolute error over all (i,j,k) in reconstruction."""
    rank = len(U[0]) if U else 0
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                reconstructed = 0
                u_val = U[_idx_U(i, j, m)]
                v_val = V[_idx_V(j, k, p)]
                w_val = W[_idx_W(k, i, n)]
                for r in range(rank):
                    reconstructed += u_val[r] * v_val[r] * w_val[r]
                err = abs(reconstructed - 1)
                if err > max_err:
                    max_err = err
    return max_err

def _try_fix_rational(U, V, W, n, m, p, max_denom=6):
    """Try to find exact rational (half-integer) values matching the decomposition."""
    n, m, p = n, m, p
    rank = len(U[0]) if U else 0
    
    # For each (i,j,k), we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # Use rational approximation from the real values
    def to_frac(x, max_denom=max_denom):
        if x == 0:
            return Fraction(0)
        # Try denominators
        best = None
        best_err = float('inf')
        for d in range(1, max_denom + 1):
            num = round(x * d)
            f = Fraction(num, d)
            err = abs(float(f) - x)
            if err < best_err:
                best_err = err
                best = f
            if err == 0:
                break
        return best
    
    U_r = [[to_frac(x) for x in row] for row in U]
    V_r = [[to_frac(x) for x in row] for row in V]
    W_r = [[to_frac(x) for x in row] for row in W]
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U_r[_idx_U(i,j,m)][r] * V_r[_idx_V(j,k,p)][r] * W_r[_idx_W(k,i,n)][r]
                if s != 1:
                    return None
    return {"U": U_r, "V": V_r, "W": W_r}

def _local_refine(U, V, W, n, m, p, steps=500, step_scale=0.1):
    """Coordinate descent to reduce decomposition error."""
    rank = len(U[0]) if U else 0
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    best_err = _tensor_rank_error(U, V, W, n, m, p)
    if best_err == 0:
        return U, V, W, 0
    
    step_size = step_scale
    for step in range(steps):
        if time.time() > DEADLINE - 5:
            break
        improved = False
        # Pick random coordinate
        tensor = random.choice(['U', 'V', 'W'])
        if tensor == 'U':
            row = random.randint(0, n*m - 1)
            col = random.randint(0, rank - 1)
            old = U[row][col]
        elif tensor == 'V':
            row = random.randint(0, m*p - 1)
            col = random.randint(0, rank - 1)
            old = V[row][col]
        else:
            row = random.randint(0, p*n - 1)
            col = random.randint(0, rank - 1)
            old = W[row][col]
        
        # Try a few perturbations
        for _ in range(20):
            delta = random.uniform(-step_size, step_size)
            new_val = old + delta
            if tensor == 'U':
                U[row][col] = new_val
            elif tensor == 'V':
                V[row][col] = new_val
            else:
                W[row][col] = new_val
            
            err = _tensor_rank_error(U, V, W, n, m, p)
            if err < best_err - 1e-12:
                best_err = err
                old = new_val
                improved = True
                if best_err == 0:
                    return U, V, W, 0
            else:
                if tensor == 'U':
                    U[row][col] = old
                elif tensor == 'V':
                    V[row][col] = old
                else:
                    W[row][col] = old
        
        if not improved:
            step_size *= 0.8
    
    return U, V, W, best_err

def _random_decomposition(rank, n, m, p):
    """Generate a random rank-r decomposition attempt."""
    U = [[random.uniform(-2, 2) for _ in range(rank)] for _ in range(n*m)]
    V = [[random.uniform(-2, 2) for _ in range(rank)] for _ in range(m*p)]
    W = [[random.uniform(-2, 2) for _ in range(rank)] for _ in range(p*n)]
    return U, V, W

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb, sb_rank = _construct_schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = sb_rank
    
    # Try a structured low-rank approach: use symmetry
    # For 3x3 * 3x7, try rank values below 63
    # Known lower bounds: for (n,m,p), rank >= n*m*p/something
    # Let's search for exact integer/half-integer decompositions
    
    random.seed(42)
    
    # Strategy 1: Try to find exact rational decompositions at various ranks
    # by random initialization + local refinement + rational fixup
    for target_rank in [36, 35, 34, 33, 32, 30, 28, 27, 24, 21, 18, 15]:
        if time.time() > deadline - 10:
            break
        
        found_exact = False
        attempts = 0
        while time.time() < deadline - 10 and attempts < 300:
            attempts += 1
            U, V, W = _random_decomposition(target_rank, n, m, p)
            U, V, W, err = _local_refine(U, V, W, n, m, p, steps=200, step_scale=0.3)
            
            if err < 1e-6:
                # Try to fix to exact rationals
                result = _try_fix_rational(U, V, W, n, m, p, max_denom=4)
                if result is not None:
                    best_rank = target_rank
                    best_result = result
                    found_exact = True
                    break
            
            if attempts % 50 == 0:
                pass
        
        if found_exact:
            break
    
    # Strategy 2: If no low rank found, try slightly higher ranks with more attempts
    if best_rank == sb_rank:
        for target_rank in [45, 42, 40, 38]:
            if time.time() > deadline - 10:
                break
            found_exact = False
            attempts = 0
            while time.time() < deadline - 10 and attempts < 500:
                attempts += 1
                U, V, W = _random_decomposition(target_rank, n, m, p)
                U, V, W, err = _local_refine(U, V, W, n, m, p, steps=300, step_scale=0.2)
                
                if err < 1e-6:
                    result = _try_fix_rational(U, V, W, n, m, p, max_denom=8)
                    if result is not None:
                        best_rank = target_rank
                        best_result = result
                        found_exact = True
                        break
        
            if found_exact:
                break
    
    return best_result
