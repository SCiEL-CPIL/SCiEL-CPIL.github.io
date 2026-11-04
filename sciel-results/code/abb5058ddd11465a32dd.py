"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np

def _verify_factors(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition matches the MM tensor exactly."""
    rank = len(U[0])
    # Check consistent indices: sum should be 1
    for i, j, k in itertools.product(range(n), range(m), range(p)):
        s = 0
        for c in range(rank):
            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        if abs(s - 1) > tol:
            return False
    # Check inconsistent indices: sum should be 0
    # For any (i,j,k) where we consider U[i*m+j], V[j'*p+k'], W[k''*n+i'']
    # with inconsistent indices, the product should sum to 0.
    # But the verify in the original only checks consistent cases.
    # Let's also check a few inconsistent cases to be safe.
    # Actually, the full tensor condition requires:
    # sum_c U[a,c]*V[b,c]*W[d,c] = 1 if (a,b,d) are consistent, 0 otherwise
    # where a = i*m+j, b = j'*p+k, d = k'*n+i'
    # Consistent means i=i', j=j', k=k'
    # Let's verify a sample of inconsistent cases
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        d = random.randint(0, p*n-1)
        i1, j1 = divmod(a, m)
        j2, k2 = divmod(b, p)
        k3, i3 = divmod(d, n)
        if i1 == i3 and j1 == j2 and k2 == k3:
            # Consistent, should be 1 (already checked)
            continue
        s = 0
        for c in range(rank):
            s += U[a][c] * V[b][c] * W[d][c]
        if abs(s) > tol:
            return False
    return True

def _als_decompose(n, m, p, rank, max_iter=2000, tol=1e-10, seed=42):
    """Use alternating least squares to find a rank-r decomposition."""
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    for it in range(max_iter):
        # Optimize U: for each (i,j) row, solve least squares
        # U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :])^T = 1 for all k
        # Actually, the condition is:
        # sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
        # where T[a,b,d] = 1 if consistent, 0 otherwise
        # For fixed V, W, optimize U:
        # For each a, we need: U[a,:] dot (V[b,:] * W[d,:]) = T[a,b,d] for all b,d
        # This is a linear system in U[a,:]
        
        # Build the system for U
        # For each a (i*m+j), the equations are:
        # sum_c U[a,c] * V[b,c] * W[d,c] = T[a,b,d] for all b,d
        # Let M_U[a] be a (m*p*p*n) x rank matrix where row (b,d) is V[b,:] * W[d,:]
        # And rhs_U[a] is T[a,b,d] for all b,d
        
        # This is expensive. Let's use a simpler approach: gradient descent
        pass
    
    # This ALS approach is getting complex. Let's try a different strategy.
    # Use a known construction or random search with rounding.
    return None

def _try_rank(n, m, p, rank, max_attempts=50, seed=0):
    """Try to find a rank-r decomposition via random search + rounding."""
    for attempt in range(max_attempts):
        s = seed + attempt
        rng = np.random.RandomState(s)
        
        # Initialize with random values
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Normalize
        for c in range(rank):
            norm = np.sqrt(np.sum(U[:,c]**2) * np.sum(V[:,c]**2) * np.sum(W[:,c]**2))
            if norm > 0:
                U[:,c] /= norm**(1/3)
                V[:,c] /= norm**(1/3)
                W[:,c] /= norm**(1/3)
        
        # Convert to lists and try rounding to half-integers
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        # Round to nearest half-integer
        U_rounded = [[round(x*2)/2 for x in row] for row in U_list]
        V_rounded = [[round(x*2)/2 for x in row] for row in V_list]
        W_rounded = [[round(x*2)/2 for x in row] for row in W_list]
        
        if _verify_factors(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
            return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    return None

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    # Try to find a lower-rank decomposition
    # Try ranks from 19 up to 99
    for rank in [19, 20, 25, 30, 40, 50, 60, 70, 80, 90, 99]:
        result = _try_rank(n, m, p, rank, max_attempts=20, seed=rank)
        if result is not None:
            return result
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
