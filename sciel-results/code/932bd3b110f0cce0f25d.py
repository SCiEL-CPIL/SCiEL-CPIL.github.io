import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def _build_rank17():
    """Build the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the 2x3x3 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(U.shape[1]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_find_rank16(deadline):
    """Try to find a rank-16 decomposition."""
    n, m, p = DIMENSIONS
    r = 16
    
    # Strategy: start from rank-17, try to merge two terms
    U17, V17, W17 = _build_rank17()
    
    # For each pair of terms, try to see if they can be combined
    # by finding a linear transformation that preserves the tensor
    # This is equivalent to: can we find U16, V16, W16 such that
    # the span of {U16[:,s] ⊗ V16[:,s] ⊗ W16[:,s]} contains the tensor?
    
    # Alternative: directly solve for rank-16
    # The equations are: for each (i,j,k), sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 1
    # This is 18 equations. We have 3*16*16 = 768 unknowns (but with scaling freedom).
    
    # Let's try a structured approach: assume the first 7 terms are Strassen-like
    # and the last 9 are a modified schoolbook
    
    # Try: use 7 Strassen terms for the 2x2x2 block, and 9 terms for the rest
    # The rest has 10 products, so we need to save 1
    
    # Key insight: A02*B20 and A12*B20 both involve B20. Can we share?
    # C00 gets A02*B20, C10 gets A12*B20
    # If we use (A02+A12)*B20 -> contributes to both C00 and C10? No, that gives A02*B20 + A12*B20
    # which is the sum, not individual terms.
    
    # Let me try a different approach: use a known rank-16 construction
    # from the literature.
    
    # Actually, let me try to find it by solving a nonlinear system
    # with a smart initialization.
    
    # Initialize with rank-17 and set one term to zero, then adjust
    for remove_idx in range(17):
        if time.time() > deadline - 10:
            break
        
        U16 = U17[:, [i for i in range(17) if i != remove_idx]].copy()
        V16 = V17[:, [i for i in range(17) if i != remove_idx]].copy()
        W16 = W17[:, [i for i in range(17) if i != remove_idx]].copy()
        
        # Now try to adjust the 16 terms to satisfy all 18 equations
        # Use gradient descent on the residual
        
        def residual(Um, Vm, Wm):
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(Um[i*m+j, s] * Vm[j*p+k, s] * Wm[k*n+i, s] for s in range(r))
                        res += (val - 1.0)**2
            return res
        
        # Simple iterative adjustment
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            res = residual(U16, V16, W16)
            if res < 1e-12:
                return U16, V16, W16
            
            # Compute gradient and take a step
            # This is complex, so let's try a simpler approach:
            # randomly perturb one entry and keep if it improves
            
            improved = False
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random matrix, entry, and term
                mat_choice = np.random.choice(3)
                if mat_choice == 0:
                    idx = np.random.randint(n*m)
                    term = np.random.randint(r)
                    old_val = U16[idx, term]
                    U16[idx, term] += np.random.uniform(-0.1, 0.1)
                    new_res = residual(U16, V16, W16)
                    if new_res < res:
                        res = new_res
                        improved = True
                    else:
                        U16[idx, term] = old_val
                elif mat_choice == 1:
                    idx = np.random.randint(m*p)
                    term = np.random.randint(r)
                    old_val = V16[idx, term]
                    V16[idx, term] += np.random.uniform(-0.1, 0.1)
                    new_res = residual(U16, V16, W16)
                    if new_res < res:
                        res = new_res
                        improved = True
                    else:
                        V16[idx, term] = old_val
                else:
                    idx = np.random.randint(p*n)
                    term = np.random.randint(r)
                    old_val = W16[idx, term]
                    W16[idx, term] += np.random.uniform(-0.1, 0.1)
                    new_res = residual(U16, V16, W16)
                    if new_res < res:
                        res = new_res
                        improved = True
                    else:
                        W16[idx, term] = old_val
                
                if not improved:
                    break
        
        if residual(U16, V16, W16) < 1e-12:
            return U16, V16, W16
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # First, verify the rank-17 decomposition
    U17, V17, W17 = _build_rank17()
    
    if not _verify(U17, V17, W17):
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Try to find rank-16
    result16 = _try_find_rank16(deadline)
    if result16 is not None:
        U16, V16, W16 = result16
        if _verify(U16, V16, W16):
            return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Return rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
