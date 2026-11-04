import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check shapes
    if U.shape[0] != n*m:
        return False
    if V.shape[0] != m*p:
        return False
    if W.shape[0] != p*n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # For each output element C[i,k], check that:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must hold for all i, j, k
    
    for i in range(n):
        for k in range(p):
            # Build the left-hand side: sum_j A[i,j]*B[j,k]
            lhs = np.zeros((n*m, m*p))
            for j in range(m):
                lhs[i*m+j, j*p+k] = 1.0
            
            # Build the right-hand side: sum_r U[:,r] * V[:,r] * W[k*n+i, r]
            rhs = np.zeros((n*m, m*p))
            for r in range(rank):
                rhs += np.outer(U[:, r], V[:, r]) * W[k*n+i, r]
            
            # Check if lhs == rhs
            if not np.allclose(lhs, rhs, atol=1e-10):
                return False
    
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            if val != 0:
                U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            if val != 0:
                V[j * p + k][r] = val
        for (k, i), val in w_dict.items():
            if val != 0:
                W[k * n + i][r] = val
    
    return {"U": U, "V": V, "W": W}

def _try_algebraic_search(n, m, p, target_rank, deadline, seed=42, max_attempts=50):
    """
    Try to find a decomposition by fixing U and V, solving for W linearly,
    then verifying all constraints.
    """
    rng = np.random.RandomState(seed)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Random U and V with various scales
        scale = rng.choice([0.5, 1.0, 2.0, 5.0])
        U = rng.randn(n*m, target_rank) * scale
        V = rng.randn(m*p, target_rank) * scale
        
        # For each (i,k), solve for W[k*n+i, :]
        W = np.zeros((p*n, target_rank))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                # Build the m x target_rank matrix: A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A = np.zeros((m, target_rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                
                # Solve A @ w = ones(m) for w = W[k*n+i, :]
                try:
                    w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    # Check if the solution is good
                    if rank_a < m or np.linalg.norm(A @ w - np.ones(m)) > 1e-10:
                        feasible = False
                        break
                    W[k*n+i, :] = w
                except:
                    feasible = False
                    break
            
            if not feasible:
                break
        
        if not feasible:
            continue
        
        # Now check all constraints
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_structured_search(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using structured initialization.
    """
    lower_bound = max(n*p, n*m, m*p)  # 15
    schoolbook_rank = n * m * p  # 45
    
    # Try ranks from lower bound upward
    for target_rank in range(lower_bound, min(schoolbook_rank, 35) + 1):
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        for seed in range(15):
            if time.time() > deadline:
                break
            
            result = _try_algebraic_search(n, m, p, target_rank, deadline, seed=seed, max_attempts=15)
            if result is not None:
                return result
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix 
    by a 3x5 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    deadline = time.time() + 130.0  # Safety margin
    
    # Try to find a better decomposition
    result = _try_structured_search(n, m, p, deadline)
    
    if result is None:
        # Fallback to schoolbook
        result = _schoolbook(n, m, p)
    else:
        # Verify the result
        if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            # Fallback to schoolbook
            result = _schoolbook(n, m, p)
    
    return result
# EVOLVE-BLOCK-END
