import time
import numpy as np
import random

DIMENSIONS = [3, 4, 8]

def _build_schoolbook(n, m, p):
    """Build schoolbook tensor decomposition for n x m * m x p matrix multiplication."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_tests=5):
    """Verify that the decomposition correctly computes C = A @ B."""
    random.seed(42)
    rank = len(U[0]) if U else 0
    for _ in range(num_tests):
        A = np.random.randint(-5, 5, size=(n, m))
        B = np.random.randint(-5, 5, size=(m, p))
        
        true_C = A @ B
        
        # Compute using decomposition
        # u[i, t] = sum_j U[i*m+j, t] * A[i, j]
        # v[k, t] = sum_j V[j*p+k, t] * B[j, k]
        # c[i, k] = sum_t W[k*n+i, t] * u[i, t] * v[k, t]
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Reshape A and B to align with U and V indices
        # A is (n, m). U is (n*m, rank).
        # We need u[i, t] = sum_j U[i*m+j, t] A[i, j]
        # This is a tensor contraction.
        # Let A_reshaped be (n, m). 
        # We can compute u by:
        # u = np.einsum('ij,imt->it', A, U_arr.reshape(n, m, rank))
        # Similarly v = np.einsum('jk,jpt->kt', B, V_arr.reshape(m, p, rank))
        # Then c = np.einsum('ikt,kt,it->ik', W_arr.reshape(p, n, rank), v, u)
        # Wait, W index is k*n+i. So W_arr is (p*n, rank).
        # Reshape W to (p, n, rank).
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        u = np.einsum('ij,imt->it', A, U_reshaped)
        v = np.einsum('jk,jpt->kt', B, V_reshaped)
        
        # c[i, k] = sum_t W_reshaped[k, i, t] * u[i, t] * v[k, t]
        c = np.einsum('kit,it,kt->ik', W_reshaped, u, v)
        
        if not np.allclose(c, true_C, atol=1e-5):
            return False
    return True

def _try_find_decomposition(n, m, p, target_rank, time_limit):
    """Attempt to find a decomposition of rank target_rank using gradient descent."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # Build true tensor
    T_true = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_true[idx_u, idx_v, idx_w] = 1.0

    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    # Try a few random initializations
    for seed in range(20):
        if time.time() - start_time > time_limit:
            break
        
        rng = np.random.RandomState(seed)
        # Initialize with small random values
        scale = 0.5
        U = rng.randn(nm, target_rank) * scale
        V = rng.randn(mp, target_rank) * scale
        W = rng.randn(pn, target_rank) * scale
        
        # Gradient descent with momentum
        lr = 0.05
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
            
            # Compute reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            
            err = np.max(np.abs(R))
            if err < 1e-6:
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            # Gradients
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            # Update with momentum
            vU = momentum * vU - lr * grad_U
            vV = momentum * vV - lr * grad_V
            vW = momentum * vW - lr * grad_W
            
            U += vU
            V += vV
            W += vW
            
            # Adaptive learning rate
            if step % 100 == 99 and err > 0.1:
                lr *= 0.9
            
            if step % 50 == 0 and err < 1e-3:
                lr *= 0.5

        if best_err < 1e-6:
            # Verify the best found so far
            if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
                return best_U, best_V, best_W

    return None

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x8."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    # 1. Build schoolbook decomposition (rank = 96)
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # 2. Try to find a lower rank decomposition
    # Start from a lower bound. 
    # For 3x4x8, let's try ranks from 48 to 95.
    # We allocate more time to lower ranks as they are harder but more valuable.
    # However, finding any rank < 96 is a win.
    
    # Strategy: Try a few specific ranks.
    # If we find rank 95, we stop.
    # If we have time, try lower.
    
    ranks_to_try = list(range(48, 96))
    
    # We will try ranks in increasing order.
    # If we find a valid decomposition, we update best and continue to see if we can go lower.
    # But finding a lower rank is harder.
    # So maybe we should try to find the *lowest* possible rank within the time limit.
    # Or just find *any* rank < 96?
    # The score is 1.0 * (96 / rank). So lower rank is better.
    # If we find rank 95, score is 96/95 = 1.01.
    # If we find rank 48, score is 96/48 = 2.0.
    
    # Let's try to find the best rank possible.
    # We'll iterate ranks from low to high.
    # If we find a solution for rank r, we store it and continue to r-1?
    # No, if we find rank 50, we know rank 50 works.
    # We should check if rank 49 works.
    
    # Let's try ranks from 48 up to 95.
    # We will spend a fixed time budget for each rank, or decreasing budget?
    # Lower ranks are harder.
    
    # Let's just try a few ranks with a time limit.
    # We'll try ranks: 60, 50, 45, 40, 35, 30.
    # If we find a solution, we record it.
    # At the end, we return the best (lowest rank) solution found.
    
    best_U, best_V, best_W = U, V, W
    best_rank_found = best_rank
    
    # We will try a set of candidate ranks.
    # Since 96 is the schoolbook, we want < 96.
    # Let's try: 90, 80, 70, 60, 50, 45, 40.
    # We'll allocate time based on remaining time.
    
    candidates = [90, 85, 80, 75, 70, 65, 60, 55, 50, 48]
    
    for r in candidates:
        if time.time() - start_time > deadline - 10:
            break
            
        # Allocate time for this rank
        # If we have 50s left, and 5 ranks left, 10s each?
        # Let's just use a fixed time limit per rank, say 5s.
        # If we find a solution, we break? No, we want the *lowest* rank.
        # So if we find rank 50, we should try 48.
        # So we should iterate from low to high?
        # If we iterate 48, 50, 55...
        # If 48 fails, we try 50.
        # If 50 succeeds, we have a solution.
        # Do we try 49? Yes.
        
        # So we should iterate from low to high.
        # But if 48 takes 10s and fails, we have less time for 50.
        
        # Let's iterate from low to high.
        # 48, 49, 50, ...
        
        pass # Placeholder for logic below

    # Revised Strategy:
    # Iterate r from 48 to 95.
    # For each r, try to find a decomposition with a time limit.
    # If found, update best.
    # Continue to next r (lower) ? No, r is increasing.
    # We want the lowest r.
    # So we should check r=48, then 49, etc.
    # If we find r=50, we know 50 works.
    # We should check if 49 works.
    # So we should iterate r from 48 up to 95.
    # If we find a solution for r, we record it.
    # We do NOT stop, because a lower r might exist?
    # No, if we iterate 48, 49, 50...
    # If 48 fails, 49 fails, 50 succeeds.
    # We have solution for 50.
    # We have already checked 48 and 49.
    # So we can stop?
    # Yes, if we iterate in increasing order of rank, the first success is the lowest rank.
    # BUT, finding a decomposition is stochastic.
    # It might fail for 48 due to bad initialization, but succeed with another seed.
    # So we should try multiple seeds for each rank.
    
    # Let's do this:
    # For r in range(48, 96):
    #    try to find decomposition with time_limit = 2s (or adaptive)
    #    if found:
    #       return it (since it's the lowest rank found so far, and we're going up)
    #       Actually, we want to be sure.
    #       If we find rank 50, we can return it immediately?
    #       Yes, because we checked all lower ranks (48, 49) and failed.
    #       So 50 is the best we found.
    #       Can we find 49 with more time?
    #       We spent 2s on 48 and 2s on 49.
    #       If we have time, we could try 49 again with more time.
    #       But for simplicity, let's just return the first success.
    
    # Let's implement this.
    
    for r in range(48, 96):
        if time.time() - start_time > deadline - 5:
            break
            
        # Time limit for this rank
        # Give more time to lower ranks?
        # Or equal time?
        # Let's give 2 seconds per rank.
        time_limit = 2.0
        
        result = _try_find_decomposition(n, m, p, r, time_limit)
        if result:
            U, V, W = result
            # Verify again to be safe
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
    # If no lower rank found, return schoolbook
    return {"U": U, "V": V, "W": W}
