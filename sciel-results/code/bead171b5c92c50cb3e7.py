"""Matrix multiplication 3x4x5 - rank-minimal decomposition search."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition using rational arithmetic with Fractions."""
    nx, ny, nz = n*m, m*p, p*n
    R = len(U[0])
    
    # Convert to Fractions for exact arithmetic
    U_frac = [[Fraction(u) for u in row] for row in U]
    V_frac = [[Fraction(v) for v in row] for row in V]
    W_frac = [[Fraction(w) for w in row] for row in W]
    
    # Build the decomposition tensor
    for x in range(nx):
        for y in range(ny):
            total = Fraction(0)
            for r in range(R):
                total += U_frac[x][r] * V_frac[y][r] * W_frac[0][r] if False else U_frac[x][r] * V_frac[y][r] * W_frac[y if False else 0][r]
            # Actually let me just compute properly
    # Simpler: compute D[x,y] = sum_r U[x,r]*V[y,r]*W[?][r]
    # Wait, W is indexed by z. So D[x,y,z] = sum_r U[x,r]*V[y,r]*W[z,r]
    # We need D[x,y,z] = T[x,y,z] for all x,y,z
    
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[x][r] * V_frac[y][r] * W_frac[z][r]
                
                # Determine expected value
                i = x // m
                j = x % m
                j2 = y // p
                k = y % p
                k2 = z // n
                i2 = z % n
                
                expected = Fraction(1) if (i == i2 and j == j2 and k == k2) else Fraction(0)
                
                if total != expected:
                    return False
    return True

def _compute_error_fast(V, W, T, nx, ny, nz, R):
    """Fast error computation given V, W. Returns (U, err)."""
    U = np.zeros((nx, R))
    err = 0.0
    
    for x in range(nx):
        # Build A_x: (ny*nz, R)
        A_x = np.zeros((ny*nz, R))
        b_x = np.zeros(ny*nz)
        for y in range(ny):
            for z in range(nz):
                idx = y*nz + z
                for r in range(R):
                    A_x[idx, r] = V[y, r] * W[z, r]
                b_x[idx] = T[x, y, z]
        
        # Solve least squares
        if R > 0:
            U_x, residuals, rank, s = np.linalg.lstsq(A_x, b_x, rcond=None)
            U[x, :] = U_x
        
        # Compute error for this x
        for y in range(ny):
            for z in range(nz):
                pred = sum(U[x, r] * V[y, r] * W[z, r] for r in range(R))
                diff = pred - T[x, y, z]
                err += diff * diff
    
    return U, err

def _try_rank_with_fixed_structure(n, m, p, R, time_limit, seed=None):
    """
    Try to find a rank-R decomposition using structured random search.
    Uses a 'fix V, solve W' approach to reduce search space.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    start_time = time.time()
    nx, ny, nz = n*m, m*p, p*n
    
    # Target tensor (1s at valid positions)
    T = np.zeros((nx, ny, nz))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                T[x, y, z] = 1.0
    
    def compute_error_fast(V, W):
        return _compute_error_fast(V, W, T, nx, ny, nz, R)
    
    def try_round_and_verify(U, V, W):
        """Try rounding U to half-integers and verify."""
        # Round to nearest half-integer
        U_int = np.round(U * 2) / 2.0
        if _verify_decomposition(U_int.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U_int.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Also try rounding to integers
        U_int2 = np.round(U).astype(int)
        if _verify_decomposition(U_int2.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U_int2.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None
    
    # Strategy: Fix V to sparse patterns, solve for W, then check if U can be half-integers
    # V has shape (ny, R) = (20, R)
    # W has shape (nz, R) = (15, R)
    
    best_err = float('inf')
    best_result = None
    
    # Multiple random restarts
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize V with sparse random pattern
        V = np.zeros((ny, R))
        for c in range(R):
            # Each column of V: pick 1-3 random rows to be non-zero
            num_nonzero = random.randint(1, min(3, ny))
            rows = random.sample(range(ny), num_nonzero)
            for r in rows:
                V[r, c] = random.choice([-1.0, 1.0])
        
        # Solve for W given V
        # For each z: sum_r V[y,r]*W[z,r] = ? 
        # Actually we need: for all x,y,z: sum_r U[x,r]*V[y,r]*W[z,r] = T[x,y,z]
        # Given V, we want to find W and U.
        # This is still underdetermined. Let's fix V and solve for W using least squares.
        
        # Approach: For each (y,z) pair, define M[y,z] = V[y,:] * W[z,:] (dot product)
        # Then we need: for all x, sum_r U[x,r]*V[y,r]*W[z,r] = T[x,y,z]
        # Which is: for all x,y,z: U[x,:] dot (V[y,:] * W[z,:]) = T[x,y,z]
        
        # Let's parameterize: fix V, then for each z, W[z,:] is a vector in R-dim.
        # The constraint is: for each x, the matrix A_x where A_x[y*nz+z] = V[y,:]*W[z,:] 
        # must satisfy A_x @ U[x,:] ≈ b_x
        
        # Simpler: use iterative approach. Fix V, randomly init W, solve U, 
        # then update W based on residuals.
        
        W = np.zeros((nz, R))
        for c in range(R):
            num_nonzero = random.randint(1, min(3, nz))
            rows = random.sample(range(nz), num_nonzero)
            for r in rows:
                W[r, c] = random.choice([-1.0, 1.0])
        
        U, err = compute_error_fast(V, W)
        
        if err < 1e-9:
            result = try_round_and_verify(U, V, W)
            if result:
                return result
        
        best_err = min(best_err, err)
        
        # Local search: perturb V or W entries
        local_no_improve = 0
        local_max_no_improve = 50
        
        while time.time() - start_time < time_limit and local_no_improve < local_max_no_improve:
            # Randomly perturb one entry of V or W
            if random.random() < 0.5:
                mat = V
                rows = ny
            else:
                mat = W
                rows = nz
            
            r = random.randint(0, rows - 1)
            c = random.randint(0, R - 1)
            
            old_val = mat[r, c]
            new_val = random.choice([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            if new_val == old_val:
                continue
            
            mat[r, c] = new_val
            
            U_new, err_new = compute_error_fast(V, W)
            
            if err_new < err - 1e-9:
                err = err_new
                local_no_improve = 0
                best_err = min(best_err, err)
                
                if err < 1e-9:
                    result = try_round_and_verify(U_new, V, W)
                    if result:
                        return result
            else:
                if random.random() > 0.1:
                    mat[r, c] = old_val
                local_no_improve += 1
    
    return best_result

def _generate_schoolbook(n, m, p):
    """Generate the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Constructs the minimal rank tensor decomposition for 3x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safe margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try ranks from schoolbook-1 downwards
    # For 3x4x5, known lower bound is around 48-55, schoolbook is 60
    # Try a few ranks below schoolbook
    ranks_to_try = [schoolbook_rank - 1, schoolbook_rank - 2, schoolbook_rank - 3, schoolbook_rank - 4]
    
    # Allocate time: give more time to higher ranks (easier to find)
    time_allocations = [30.0, 25.0, 20.0, 15.0]
    
    for idx, (R, time_limit) in enumerate(zip(ranks_to_try, time_allocations)):
        if time.time() > deadline - time_limit:
            break
        
        # Try multiple seeds for diversity
        for seed in [42, 123, 456]:
            if time.time() > deadline - 5:
                break
            remaining = min(time_limit, deadline - time.time() - 2)
            if remaining < 2:
                break
            res = _try_rank_with_fixed_structure(n, m, p, R, remaining, seed=seed)
            if res is not None:
                return res
    
    # Fallback to schoolbook
    return _generate_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
