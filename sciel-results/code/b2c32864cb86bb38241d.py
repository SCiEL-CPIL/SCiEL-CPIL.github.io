import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 150.0  # Evaluator kills at 150s, keep margin
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 10.0  # 10s safety margin

def _verify_fast(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Entries must be integer or half-integer.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    try:
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
    except:
        return False

    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    # Check integer/half-integer constraint
    for arr in [Ua, Va, Wa]:
        # 2 * x should be integer
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    # Compute Tensor
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Note: W index is k*n+i, so W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _snap_to_half(x):
    return round(2 * x) / 2.0

def _try_rank_altsolve(n, m, p, rank, max_iter=500, seed=None, tol=1e-10, init_type="random"):
    """
    Try to find a valid decomposition of given rank using alternating linear solve.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    if init_type == "random":
        U[:] = rng.uniform(-1, 1, (n*m, rank))
        V[:] = rng.uniform(-1, 1, (m*p, rank))
        W[:] = rng.uniform(-1, 1, (p*n, rank))
    elif init_type == "sparse":
        # Initialize with sparse structure similar to schoolbook
        # Randomly assign 1s to a few positions
        for c in range(rank):
            i = rng.randint(0, n*m-1)
            j = rng.randint(0, m*p-1)
            k = rng.randint(0, p*n-1)
            U[i, c] = 1.0
            V[j, c] = 1.0
            W[k, c] = 1.0
        # Add some noise
        U += rng.uniform(-0.1, 0.1, (n*m, rank))
        V += rng.uniform(-0.1, 0.1, (m*p, rank))
        W += rng.uniform(-0.1, 0.1, (p*n, rank))
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        # Snap to half-integers
        Ua_snapped = np.vectorize(_snap_to_half)(best_U)
        Va_snapped = np.vectorize(_snap_to_half)(best_V)
        Wa_snapped = np.vectorize(_snap_to_half)(best_W)
        
        U_out = Ua_snapped.tolist()
        V_out = Va_snapped.tolist()
        W_out = Wa_snapped.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
        
        # Try raw values (in case snapping broke it, though unlikely if error < tol)
        U_out_raw = best_U.tolist()
        V_out_raw = best_V.tolist()
        W_out_raw = best_W.tolist()
        
        if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
            return U_out_raw, V_out_raw, W_out_raw
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    # Lower bound for rank: max(n*m, m*p, n*p) = 30
    min_rank = max(n*m, m*p, n*p)
    
    # We search for ranks from min_rank to best_rank - 1
    # If we find one, we update best.
    # We prioritize lower ranks.
    
    # To save time, we don't search all ranks if we don't have time.
    # We search a subset of ranks or just try to improve on 90.
    
    # Strategy:
    # 1. Try a few random initializations for rank 90 to see if we can find a "better" 90? No, rank is rank.
    # 2. Try to find rank < 90.
    
    # Given the difficulty, we try a few specific ranks close to 90 and some lower ones.
    # Actually, if rank 90 is the minimum, we should return it.
    # If we suspect rank 89 exists, we should try it.
    
    # Let's try ranks from 30 to 89, but with a very tight time budget per rank.
    # If we don't find anything, we return 90.
    
    # Optimized search:
    # Try ranks in order of likelihood? No, minimal rank is the goal.
    # But if 30-89 are impossible, we waste time.
    # We will try a few random ranks in [30, 89] and the immediate neighbors of 90.
    
    # Actually, the most efficient use of time is to try to find ANY valid decomposition
    # with rank < 90. If we fail, we return 90.
    
    # We will iterate through a few candidate ranks.
    # Candidate ranks: 30, 31, ..., 35, 85, 86, 87, 88, 89.
    # This covers the lower bound and the vicinity of the schoolbook rank.
    
    candidate_ranks = list(range(min_rank, min_rank + 5)) + list(range(best_rank - 5, best_rank))
    # Remove duplicates and sort
    candidate_ranks = sorted(list(set(candidate_ranks)))
    
    for rank in candidate_ranks:
        if _remaining() < 10.0:
            break
            
        # Try a few seeds for this rank
        for attempt in range(3):
            if _remaining() < 5.0:
                break
            
            # Use different initializations
            init_type = "random" if attempt % 2 == 0 else "sparse"
            
            result = _try_rank_altsolve(n, m, p, rank, max_iter=200, seed=rank*100+attempt, init_type=init_type)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    # If we found a rank, we might want to check if we can go lower?
                    # But time is limited. We found a valid one.
                    # We can break out of the attempt loop and try to find a lower rank?
                    # The loop iterates ranks in increasing order.
                    # So if we found rank R, we can break the outer loop if we want to return the first found?
                    # No, we want the MINIMAL rank.
                    # So we should continue checking lower ranks?
                    # But we are iterating increasing. So if we found rank 35, we should check 30-34?
                    # We already checked 30-34 if they were in candidate_ranks.
                    # So if we find a valid one, we can update best and break?
                    # Not necessarily, a lower rank might be found later?
                    # No, we iterate increasing. So the first valid one found is the minimal among the candidates.
                    # But we didn't check all ranks.
                    # If we found rank 85, maybe rank 30 works?
                    # We checked 30-34 first. If they failed, and 85 works, 85 is the best found.
                    # We can break.
                    break
        
        if best_rank < best_rank: # This condition is always false in this scope, logic error in thought
            pass
        if best_rank < n*m*p:
            # Found a better rank than schoolbook. 
            # We can break if we found a very low rank, or continue?
            # Since we iterate increasing, the first found is the minimal among candidates.
            # We can break to save time for final verification?
            # Or continue to see if an even lower rank (not in candidates) works?
            # Unlikely. We break.
            break
    
    # Final check
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
    
    return best
