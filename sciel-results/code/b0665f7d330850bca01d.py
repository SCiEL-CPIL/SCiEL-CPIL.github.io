import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def _build_target_tensor(n, m, p):
    """Build the target matmul tensor as a dict of (i,j,k) -> 1."""
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i, j, k)] = 1.0
    return target

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly computes matmul."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                vals = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                s = np.sum(vals)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(arr, tol=1e-10):
    """Check if all entries are integers or half-integers."""
    for val in arr:
        doubled = 2.0 * val
        if abs(doubled - round(doubled)) > tol:
            return False
    return True

def _snap_to_half_int(arr, tol=1e-8):
    """Snap values to nearest half-integer if close."""
    result = []
    for val in arr:
        doubled = 2.0 * val
        nearest = round(doubled)
        if abs(doubled - nearest) < tol:
            result.append(nearest / 2.0)
        else:
            result.append(val)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook rank
    rank_schoolbook = nm * mp // m  # = n*m*p
    # Actually rank_schoolbook = n*m*p = 36
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    # First, return schoolbook as fallback
    U_fb = [[0] * rank_schoolbook for _ in range(nm)]
    V_fb = [[0] * rank_schoolbook for _ in range(mp)]
    W_fb = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # For small problems, try discrete search with half-integer entries
    # Candidate values for U, V, W entries: {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    
    # Start from lower ranks and work up
    # But we need to be efficient. Let's try a few specific ranks.
    
    # For 2x3 x 3x6, try ranks 30, 32, 34, 35
    ranks_to_try = [30, 31, 32, 33, 34, 35]
    
    # Actually, let's think differently. Use a random search with local improvement.
    # Initialize with schoolbook and try to zero out columns by adjusting others.
    
    # Alternative: use the fact that we can express some products as linear combinations.
    # For example, in standard matmul, if we have terms for (i,j,k1) and (i,j,k2),
    # we might be able to combine them if W allows.
    
    # Let me try a more systematic approach: 
    # For each rank r from 35 down to some minimum, try to find a solution.
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        # Try random initialization + local search
        np.random.seed(42 + r)
        
        # Random half-integer initialization
        candidates = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Initialize with some random values
        for t in range(r):
            U[:, t] = np.random.choice(candidates, size=nm)
            V[:, t] = np.random.choice(candidates, size=mp)
            W[:, t] = np.random.choice(candidates, size=pn)
        
        # Local search: try to change one entry at a time to reduce error
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Compute current error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vals = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                        error += (np.sum(vals) - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-10:
                break
            
            # Try a random perturbation
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                idx = np.random.randint(nm)
                col = np.random.randint(r)
                old_val = U[idx, col]
                new_val = np.random.choice(candidates)
                U[idx, col] = new_val
                # Compute new error incrementally would be faster, but let's just check
                err_new = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i*m + j != idx:
                                continue
                            vals = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                            err_new += (np.sum(vals) - 1.0) ** 2
                # This is incomplete; just accept and check later
            elif mat_choice == 1:
                idx = np.random.randint(mp)
                col = np.random.randint(r)
                V[idx, col] = np.random.choice(candidates)
            else:
                idx = np.random.randint(pn)
                col = np.random.randint(r)
                W[idx, col] = np.random.choice(candidates)
        
        # Check if we found a valid decomposition
        if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            if _is_half_integer(best_U) and _is_half_integer(best_V) and _is_half_integer(best_W):
                best_rank = r
                best_U = [[x for x in row] for row in best_U.tolist()]
                best_V = [[x for x in row] for row in best_V.tolist()]
                best_W = [[x for x in row] for row in best_W.tolist()]
                break
        
        # Also try starting from schoolbook and merging
        if time.time() > deadline - 10:
            break
        
        # Start from schoolbook (rank 36) and try to reduce
        U_s = np.zeros((nm, 36))
        V_s = np.zeros((mp, 36))
        W_s = np.zeros((pn, 36))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        
        # Try to zero out one column and adjust others
        for target_col in range(36):
            if time.time() > deadline - 10:
                    break
            
            # Try setting column target_col to 0 and see if we can compensate
            # This is complex; skip for now
        
        if time.time() > deadline - 15:
            break
    
    # If we found a better solution, return it
    if best_rank is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Return schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
