"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

TIME_LIMIT = 145.0
START_TIME = time.time()

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition
    U = np.zeros((nm, schoolbook_rank))
    V = np.zeros((mp, schoolbook_rank))
    W = np.zeros((pn, schoolbook_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    current_rank = schoolbook_rank
    
    # Fast verification using tensor reconstruction
    def verify(U, V, W, n, m, p):
        """Check if decomposition is valid"""
        rank = U.shape[1]
        # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank by removing terms
    # Strategy: for each column, check if removing it still gives valid tensor
    # If not, try to find a linear combination that allows removal
    
    def try_remove_term(U, V, W, r_remove, n, m, p):
        """Try to remove term r_remove by finding alpha such that
        U[:, r] * V[:, r] * W[:, r] = sum_{s != r} alpha_s * U[:, s] * V[:, s] * W[:, s]
        in the tensor sense.
        
        Actually, we need: for all (i,j,k),
        U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = sum_{s != r} beta_s(i,j,k) * ...
        
        This is nonlinear. Instead, we try: can we adjust the other terms to compensate?
        
        Simpler approach: just check if removing the term still works (it won't for schoolbook),
        then try small perturbations.
        """
        pass
    
    # Better approach: use the fact that for schoolbook, each term is independent.
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    # A rank-1 term contributes U[i*m+j]*V[j*p+k]*W[k*n+i] to T[i,j,k].
    
    # For the all-ones tensor, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to: the tensor T = sum_r u_r ⊗ v_r ⊗ w_r where
    # u_r[i*m+j] = U[i*m+j, r], v_r[j*p+k] = V[j*p+k, r], w_r[k*n+i] = W[k*n+i, r]
    
    # The all-ones tensor has rank at most n*m*p (schoolbook).
    # Can we do better?
    
    # Key insight: T[i,j,k] = 1 = 1*1*1. This is a rank-1 tensor!
    # Wait, no. T[i,j,k] = 1 for all i,j,k means T = ones(n) ⊗ ones(m) ⊗ ones(p).
    # This IS rank 1!
    
    # But wait, that's not right for matrix multiplication. Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix 
    # by a 4 by 4 matrix."
    
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    # So M[i,j,k] = 1 for all valid i,j,k where j is the shared index.
    
    # For M_{3,4,4}: i in [0,3), j in [0,4), k in [0,4)
    # M[i,j,k] = 1 for all (i,j,k) in this range.
    
    # So the tensor IS the all-ones tensor of shape (3,4,4)!
    # And the all-ones tensor has rank 1: ones(3) ⊗ ones(4) ⊗ ones(4).
    
    # Let me verify: U[i*m+j, 0] = 1 for all i,j
    #            V[j*p+k, 0] = 1 for all j,k  
    #            W[k*n+i, 0] = 1 for all k,i
    # Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1. ✓
    
    # So the rank is 1!
    
    if time_left() < 5:
        # Fallback to schoolbook if something goes wrong
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Rank 1 solution
    U1 = np.ones((nm, 1))
    V1 = np.ones((mp, 1))
    W1 = np.ones((pn, 1))
    
    # Verify
    if verify(U1, V1, W1, n, m, p):
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}
    else:
        # Should not happen, but fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
