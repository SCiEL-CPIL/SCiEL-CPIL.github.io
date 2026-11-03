"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    Uses optimization to find low-rank decompositions.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p  # 80
    
    def validate(U, V, W):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                    s = np.sum(prod)
                    if abs(s - 1.0) > 1e-5:
                        return False
        return True

    def error_func(x):
        """Calculate the sum of squared errors for the tensor decomposition."""
        # x contains flattened U, V, W
        r = len(x) // (n*m + m*p + p*n)
        U_size = n * m * r
        V_size = m * p * r
        W_size = p * n * r
        
        U = x[:U_size].reshape(n * m, r)
        V = x[U_size:U_size+V_size].reshape(m * p, r)
        W = x[U_size+V_size:].reshape(p * n, r)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err += (val - 1.0) ** 2
        return err

    # Try to find lower rank decompositions using optimization
    # We search from a known good upper bound (49 for 4x4x4, maybe similar for 4x4x5) up to 79
    # We use random restarts and L-BFGS-B
    
    import scipy.optimize as opt
    
    # Target ranks to try. 49 is a strong candidate based on 4x4x4 Strassen.
    # We'll try 49, 50, 51... up to 79
    # But to be safe and fast, we'll prioritize lower ranks.
    target_ranks = list(range(49, 80))
    
    # Shuffle to try different orders if time allows, but lower is better
    # Actually, let's just go in order. If we find one at 49, we stop.
    # If we don't find one at 49, we try 50, etc.
    
    for r in target_ranks:
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations for this rank
        max_attempts = 2 if r < 60 else 1
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            np.random.seed(42 + r + attempt)
            U0 = np.random.randn(n * m, r) * 0.5
            V0 = np.random.randn(m * p, r) * 0.5
            W0 = np.random.randn(p * n, r) * 0.5
            
            # Flatten
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Optimize
            try:
                res = opt.minimize(error_func, x0, method='L-BFGS-B', 
                                   options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.success and res.fun < 1e-6:
                    # Extract solution
                    U_size = n * m * r
                    V_size = m * p * r
                    W_size = p * n * r
                    
                    U_opt = res.x[:U_size].reshape(n * m, r)
                    V_opt = res.x[U_size:U_size+V_size].reshape(m * p, r)
                    W_opt = res.x[U_size+V_size:].reshape(p * n, r)
                    
                    # Final validation
                    if validate(U_opt, V_opt, W_opt):
                        best_result = {'U': U_opt.tolist(), 'V': V_opt.tolist(), 'W': W_opt.tolist()}
                        best_rank = r
                        break # Found a valid decomposition for this rank, move to next lower rank? No, we are iterating up. 
                             # Wait, we want the MINIMAL rank. So if we iterate 49, 50, 51... and find one at 50, 
                             # we should keep searching for 49? No, 49 was already tried.
                             # So if we find a solution at r, it's the best so far because we started from low r.
                             # But we might find a better one later? No, we are going 49, 50, 51...
                             # So the first one we find is the minimal among the ones we tried.
                             # However, it's possible that 49 failed but 48 would work? 
                             # We didn't try 48. The known bound is 49. Let's assume 49 is the floor.
                             # If we find a solution at 50, we should stop? 
                             # The goal is rank-minimal. If we found one at 50, and 49 failed, 
                             # it's possible 48 works? Unlikely given bounds. 
                             # But to be safe, we can continue searching lower if we had time? 
                             # No, we iterated UP. So if we find one at 50, we don't know about 49.
                             # We should iterate DOWN? No, lower rank is harder to find.
                             # Standard practice: Start from known upper bound and go down? 
                             # Or start from lower bound and go up?
                             # If we start from 49 and go up, the first success is the best we found.
                             # If we start from 79 and go down, the first success is the best we found.
                             # Going down is safer because we know 80 works.
                             # Let's reverse the loop: 79 down to 49.
                             # If we find one at 60, we continue to 59, 58...
                             # This ensures we find the lowest rank possible within the search space.
            except Exception:
                pass
        
        # If we found a solution at this rank, we can break if we are iterating down?
        # Wait, my loop is `for r in target_ranks` where `target_ranks` is 49..79.
        # If I find a solution at 50, I should break? 
        # No, because 49 might work. But I already tried 49.
        # So if I try 49, 50, 51... and 49 fails, 50 succeeds, then 50 is the best.
        # I can break because any higher rank is worse.
        if best_rank == r:
             break # Found the lowest rank in the sequence so far? 
             # Wait, if I break, I stop searching. 
             # Since I am iterating 49, 50, 51..., if I find one at 50, 
             # I know 49 failed. So 50 is the best. I can break.
             pass

    return best_result
# EVOLVE-BLOCK-END
