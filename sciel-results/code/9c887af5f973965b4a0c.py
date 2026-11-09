"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 110  # 110s deadline for safety
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try lower ranks
    lower_bound = max(nm, mp, pn)
    for rank in range(lower_bound, rank_sb):
        if time.time() > deadline - 20:
            break
        result = _search_decomp(n, m, p, rank, deadline - 15)
        if result is not None:
            best_result = result
            break
    
    return best_result

def _search_decomp(n, m, p, rank, time_limit):
    """Search for a rank-r exact decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + time_limit
    
    # The constraint: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # Strategy: Set W[k*n+i, r] = 1 for all r initially, then solve for U and V.
    # With W=1: sum_r U[i*m+j,r]*V[j*p+k,r] = 1 for all i,j,k
    # This means U[i*m+j,:] dot V[j*p+k,:] = 1 for all i,j,k
    # For fixed j: U[i*m+j,:] dot V[j*p+k,:] = 1 for all i,k
    # This means V[j*p+k,:] must be orthogonal to all U[i*m+j,:] except in a specific way...
    # Actually this is very restrictive.
    
    # Alternative: Use a structured approach. Let's try to find a decomposition
    # where U, V, W have simple patterns.
    
    # Try: U[i*m+j, r] = a[i,j,r], V[j*p+k, r] = b[j,k,r], W[k*n+i, r] = c[k,i,r]
    # with sum_r a[i,j,r]*b[j,k,r]*c[k,i,r] = 1
    
    # Let's try rank = nm + mp + pn - something, or use known techniques.
    
    # For now, try a randomized search with verification
    np.random.seed(12345 + rank)
    
    for attempt in range(1000):
        if time.time() > deadline:
            return None
        
        # Initialize with small integer/half-integer values
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Random initialization with values from {-1, 0, 1}
        for r in range(rank):
            # Sparse initialization
            nnz = max(1, int(rank * 0.1))
            idx_u = np.random.choice(nm, size=min(nnz, nm), replace=False)
            idx_v = np.random.choice(mp, size=min(nnz, mp), replace=False)
            idx_w = np.random.choice(pn, size=min(nnz, pn), replace=False)
            U[idx_u, r] = np.random.choice([-1, 1], size=len(idx_u))
            V[idx_v, r] = np.random.choice([-1, 1], size=len(idx_v))
            W[idx_w, r] = np.random.choice([-1, 1], size=len(idx_w))
        
        # Verify
        if _verify(n, m, p, U, V, W):
            # Convert to lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

def _verify(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    nm, mp, pn = n * m, m * p, p * n
    # For all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-END
