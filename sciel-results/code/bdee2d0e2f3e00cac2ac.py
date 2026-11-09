"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition satisfies T[i,j,k] = 1 for all valid (i,j,k)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _discrete_repair(U, V, W, n, m, p, deadline):
    """
    Attempt to repair a near-miss decomposition by moving entries to the nearest half-integers.
    Uses a greedy steepest descent approach on the discrete lattice of half-integers.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Initial rounding to half-integers
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    def calc_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0)**2
        return err

    err = calc_error(U, V, W)
    if err < 1e-8:
        return U, V, W

    # Precompute contributions for faster gradient estimation
    # We will iterate and try to flip individual entries to the "best" neighbor.
    # Neighbors for a value x in {0, 0.5, 1.0, ...} are x-0.5, x+0.5.
    # We also allow 0 as a neighbor if current is not 0.
    
    max_iter = 5000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if err < 1e-8:
            break
            
        best_improvement = 1e-8
        best_change = None # (matrix, i, j, new_val)
        
        # Try changing U
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    old_val = U[i*m+j, c]
                    # Candidates: old_val +/- 0.5, and 0 if old_val != 0
                    candidates = set([old_val - 0.5, old_val + 0.5])
                    if abs(old_val) > 1e-9:
                        candidates.add(0.0)
                    
                    for new_val in candidates:
                        if abs(new_val) > 10.0: # Sanity check
                            continue
                        U[i*m+j, c] = new_val
                        new_err = calc_error(U, V, W)
                        U[i*m+j, c] = old_val
                        
                        if new_err < err - best_improvement:
                            best_improvement = err - new_err
                            best_change = ('U', i*m+j, c, new_val)
        
        # Try changing V
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    old_val = V[j*p+k, c]
                    candidates = set([old_val - 0.5, old_val + 0.5])
                    if abs(old_val) > 1e-9:
                        candidates.add(0.0)
                    
                    for new_val in candidates:
                        if abs(new_val) > 10.0:
                            continue
                        V[j*p+k, c] = new_val
                        new_err = calc_error(U, V, W)
                        V[j*p+k, c] = old_val
                        
                        if new_err < err - best_improvement:
                            best_improvement = err - new_err
                            best_change = ('V', j*p+k, c, new_val)
        
        # Try changing W
        for k in range(p):
            for i in range(n):
                for c in range(r):
                    old_val = W[k*n+i, c]
                    candidates = set([old_val - 0.5, old_val + 0.5])
                    if abs(old_val) > 1e-9:
                        candidates.add(0.0)
                    
                    for new_val in candidates:
                        if abs(new_val) > 10.0:
                            continue
                        W[k*n+i, c] = new_val
                        new_err = calc_error(U, V, W)
                        W[k*n+i, c] = old_val
                        
                        if new_err < err - best_improvement:
                            best_improvement = err - new_err
                            best_change = ('W', k*n+i, c, new_val)
        
        if best_change is not None:
            mat_name, idx, c, new_val = best_change
            if mat_name == 'U':
                U[idx, c] = new_val
            elif mat_name == 'V':
                V[idx, c] = new_val
            else:
                W[idx, c] = new_val
            err -= best_improvement
        else:
            # Stuck in local minimum. Try a random perturbation?
            # Or break and try a new restart.
            break
            
    if err < 1e-8:
        return U, V, W
    return None

def _try_rank(r, n, m, p, deadline, seed=42):
    """Try to find a rank-r decomposition using random restarts and discrete repair."""
    rng = np.random.RandomState(seed)
    
    # We perform several random restarts
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Random initialization
        # Using small integers/half-integers to start close to the solution manifold
        # Uniform random in {-1, 0, 1} or {-0.5, 0, 0.5}
        U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
        V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
        W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
        
        # Run discrete repair
        result = _discrete_repair(U, V, W, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    deadline = time.time() + 145
    
    # Try ranks from 12 up to 23
    # 2x3x4 rank is likely around 12-14.
    for target_rank in range(12, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(target_rank, n, m, p, deadline, seed=target_rank)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
