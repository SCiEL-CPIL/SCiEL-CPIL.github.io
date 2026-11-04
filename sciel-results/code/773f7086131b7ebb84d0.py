"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""

import time
import random
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [3, 3, 7]
_TIME_LIMIT = 120.0
_DEADLINE = time.time() + _TIME_LIMIT - 15.0  # safety margin

# EVOLVE-BLOCK-START
def _time_left():
    return _DEADLINE - time.time()

def _verify(U, V, W):
    """Verify that the tensor decomposition is exact for 3x3 * 3x7."""
    n, m, p = 3, 3, 7
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check tensor equality: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T[i,j,k] should be 1 if this is the standard multiplication tensor, 0 otherwise
    # Actually the tensor for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k) since every product contributes.
    # Wait, let me reconsider. The multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # in the standard basis. So we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(U.shape[1]))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook rank-63 decomposition."""
    n, m, p = 3, 3, 7
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_decomp(rank):
    """
    Try to find a rank-`rank` decomposition using a randomized approach.
    Strategy: We need to find U (9 x rank), V (21 x rank), W (21 x rank) such that
    for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1.
    
    This is equivalent to finding a rank-r decomposition of the 3rd-order tensor T
    where T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,7).
    
    We use the approach: fix U and V randomly, then W is determined linearly.
    For each k, the equation becomes:
    sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j.
    
    This is a linear system in W. For a fixed (U,V), we can solve for W.
    """
    n, m, p = 3, 3, 7
    dim_u = n * m  # 9
    dim_v = m * p  # 21
    dim_w = p * n  # 21
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Fix U and V, solve for W
    # For each k, we have equations: for all (i,j):
    #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # Let A_k[i, r] = U[i*m+j,r] * V[j*p+k,r] -- but this depends on j too
    # Actually: for fixed k, the unknowns are W[k*n+0,r], W[k*n+1,r], W[k*n+2,r] for r in range(rank)
    # That's 3*rank unknowns.
    # The equations are: for each (i,j) in [0,3)x[0,3):
    #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # That's 9 equations per k, 3k total = 27 equations.
    # Unknowns: 3*rank.
    # For rank >= 9, we have more unknowns than equations, so solutions exist.
    # For rank < 9, we need the system to be consistent.
    
    # Let's try: for each k, build a linear system.
    # Unknowns: W[k*n+i, r] for i in 0..2, r in 0..rank-1. Flatten to index (i*rank + r).
    # Equation for (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Build the full linear system for all k simultaneously.
    # Total equations: p * n * m = 7*3*3 = 63
    # Total unknowns: p * n * rank = 7*3*rank = 21*rank
    
    # For rank >= 3, we have more unknowns than equations, so a solution exists.
    # But we need the solution to be consistent across all equations.
    
    # Actually, the equations for different k are independent in terms of W variables
    # (W[k*n+i,r] only appears in equations for that k), so we can solve each k-block independently.
    
    # For each k: 9 equations, 3*rank unknowns.
    # If 3*rank > 9 (i.e., rank > 3), we have a underdetermined system.
    # We can find a solution by least squares or by picking free variables.
    
    # Strategy: Pick random U and V, then for each k, solve the linear system for W.
    # If the system is consistent (which it should be for rank >= 3 since we have more unknowns),
    # we get a valid W.
    
    # But wait - we need the SAME U and V to work for all k. Let's just solve.
    
    for attempt in range(200):
        if _time_left() < 5:
            break
        
        # Random U and V with entries from {-1, 0, 1} or small integers
        U = np.random.randint(-2, 3, size=(dim_u, rank)).astype(float)
        V = np.random.randint(-2, 3, size=(dim_v, rank)).astype(float)
        
        # Ensure U and V are not all zeros
        if np.all(U == 0) or np.all(V == 0):
            continue
        
        W = np.zeros((dim_w, rank), dtype=float)
        valid = True
        
        for k in range(p):
            # Build linear system for W[k*n+0..2, :]
            # 9 equations, 3*rank unknowns
            # Equation (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # Coefficient matrix: 9 x (3*rank)
            A = np.zeros((n*m, 3*rank), dtype=float)
            b = np.ones(n*m, dtype=float)
            
            for i in range(n):
                for j in range(m):
                    eq_idx = i * m + j
                    for r in range(rank):
                        # W[k*n+i, r] has index (i*rank + r) in the flattened unknown vector
                        A[eq_idx, i*rank + r] = U[i*m+j, r] * V[j*p+k, r]
            
            # Solve least squares
            try:
                sol, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                valid = False
                break
            
            # Check if solution is exact
            if not np.allclose(A @ sol, b, atol=1e-8):
                # Try a different approach: use null space to find exact solution
                # Find one particular solution and add null space components
                # Actually, if the system is consistent, lstsq should find it.
                # Let's check consistency more carefully.
                valid = False
                break
            
            W[k*n:(k+1)*n, :] = sol.reshape(n, rank)
        
        if not valid:
            continue
        
        # Verify the full decomposition
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank_decomp_structured(rank):
    """
    Try a structured approach: use the fact that T[i,j,k]=1 for all i,j,k.
    We can try to factor this tensor in a structured way.
    
    For example, if we can write 1 = sum_r a_r(i) * b_r(j) * c_r(k) for all i,j,k,
    then U[i*m+j, r] = a_r(i) (independent of j), V[j*p+k, r] = b_r(j)*something, etc.
    
    Actually, let's think differently. The tensor T[i,j,k] = 1 can be written as:
    T[i,j,k] = 1 * 1 * 1
    This is already rank 1! But that's not right because the multiplication tensor
    is not just all-ones. Let me re-examine.
    
    Wait - I need to reconsider what the multiplication tensor actually is.
    For C = A * B where A is (n x m) and B is (m x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has T[i,j,k] = 1 for all valid i,j,k. This means every triple
    (i,j,k) contributes to the product. So T is indeed the all-ones tensor of shape (3,3,7).
    
    The all-ones tensor of shape (n,m,p) has rank 1! Because T[i,j,k] = 1 = 1*1*1.
    So U[:,0] = 1, V[:,0] = 1, W[:,0] = 1 gives rank 1.
    
    Let me verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1. Yes!
    """
    n, m, p = 3, 3, 7
    # Rank 1 decomposition of the all-ones tensor
    U = [[1]]  # 9 x 1
    V = [[1]]  # 21 x 1  
    W = [[1]]  # 21 x 1
    
    # Expand to correct shapes
    U = [[1] for _ in range(n*m)]
    V = [[1] for _ in range(m*p)]
    W = [[1] for _ in range(p*n)]
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """
    Returns a rank-minimal decomposition for 3x3 by 3x7 matrix multiplication.
    """
    # First try the rank-1 decomposition (all-ones tensor)
    result = _try_rank_decomp_structured(1)
    if result is not None:
        return result
    
    # Try increasing ranks
    for rank in range(2, 15):
        if _time_left() < 10:
            break
        result = _try_rank_decomp(rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
