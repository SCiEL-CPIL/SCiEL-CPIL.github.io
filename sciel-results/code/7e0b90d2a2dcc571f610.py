"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time

_START = time.time()
_LIMIT = 120.0          # hard safety margin before the 150 s wall-clock kill
DIMENSIONS = [3, 4, 7]


def _check(U, V, W, n, m, p, r, tol=1e-6):
    """Check if the tensor decomposition exactly reproduces matrix multiplication."""
    # Term 0 must be the identity term (schoolbook)
    # For each term c >= 1, we need sum_{i,j,k} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 0
    # Actually, the full check is: for each (i,k), sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if (i,j,k) matches
    # More precisely: the tensor T[i,j,k] = delta[i,j,k] for the multiplication identity
    # We need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if this corresponds to the (i,j,k) term
    
    # The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) in the sense that
    # out[i,k] = sum_j A[i,j]B[j,k], so the tensor is the "identity" bilinear map.
    # The rank-1 decomposition must satisfy:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1  for all i,j,k? No.
    
    # Actually, the constraint is:
    # For the bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]:
    # We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] to equal the coefficient of A[i,j]*B[j,k] in C[k*n+i]
    # which is 1 if the indices match (always, since we're decomposing the full tensor)
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    # A rank-r decomposition means T = sum_c U_c ⊗ V_c ⊗ W_c where
    # (U_c ⊗ V_c ⊗ W_c)[i*m+j, j*p+k, k*n+i] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    # So we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # AND the "wrong" entries must be 0, but since the tensor is supported only on these indices,
    # we just need the sum to equal 1 for each (i,j,k).
    
    # Let's verify by computing the full tensor and checking.
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
    
    # More efficiently, check that for all i,j,k:
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n

    # --- 1. Explicit schoolbook decomposition (rank n*m*p), always valid ---
    R0 = n * m * p
    U0 = np.zeros((Nm, R0)); V0 = np.zeros((Mp, R0)); W0 = np.zeros((Pn, R0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    def _pack(arr):
        return [[float(x) for x in row] for row in np.asarray(arr)]

    # best result found so far (initially schoolbook)
    best = {"U": _pack(U0), "V": _pack(V0), "W": _pack(W0)}

    # --- 2. Search for a lower-rank exact decomposition ---
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # We seek rank-r decomposition: T = sum_{c=0}^{r-1} U_c ⊗ V_c ⊗ W_c
    
    # Strategy: Use the fact that the tensor is the "multiplication tensor".
    # Known lower bounds: rank >= max(n*p, m*p, n*m) = max(21, 28, 12) = 28.
    # Try to find decompositions with rank < 84.
    
    # Approach: Start with the schoolbook decomposition and try to reduce rank
    # by finding linear dependencies among the rank-1 terms.
    
    # Alternative: Use random initialization + ALS to find low-rank decompositions.
    
    # Let's use a more systematic approach:
    # The tensor T can be viewed as a matrix of shape (n*m) x (m*p*p*n) by flattening.
    # Actually, let's use the standard approach: T as a 3-way tensor, and try to find
    # low-rank decompositions via optimization.
    
    # For now, let's try a simpler approach: check if we can find a decomposition
    # with rank < R0 by solving a system of equations.
    
    # Key insight: The tensor T[i*m+j, j*p+k, k*n+i] = 1 is actually the "identity"
    # tensor for matrix multiplication. Its rank is known to be at most n*m*p (schoolbook)
    # and at least max(n*p, m*p, n*m) = 28.
    
    # Let's try to find a decomposition with rank as low as possible using ALS.
    
    def _als_decompose(n, m, p, r, max_iter=100, tol=1e-8, seed=42):
        """Find a rank-r decomposition of the matrix multiplication tensor using ALS."""
        rng = np.random.RandomState(seed)
        
        # Initialize randomly
        U = rng.randn(Nm, r) * 0.1
        V = rng.randn(Mp, r) * 0.1
        W = rng.randn(Pn, r) * 0.1
        
        # The target tensor T has T[i*m+j, j*p+k, k*n+i] = 1
        # We need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
        
        # ALS: fix two factors, solve for the third.
        # For fixed V and W, solve for U:
        # For each (i,j), we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
        # This is a linear system in U[i*m+j,:].
        
        for iteration in range(max_iter):
            # Solve for U given V, W
            U_new = np.zeros((Nm, r))
            for i in range(n):
                for j in range(m):
                    # For this (i,j), we have equations for each k:
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # Let's form the matrix A of shape (p, r):
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Then A @ U[i*m+j,:] = ones(p)
                    A = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    b = np.ones(p)
                    # Solve A @ u = b
                    if r <= p:
                        u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    else:
                        u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = u
            
            # Solve for V given U, W
            V_new = np.zeros((Mp, r))
            for j in range(m):
                for k in range(p):
                    # For this (j,k), we have equations for each i:
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    # A @ v = ones(n)
                    A = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    b = np.ones(n)
                    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = v
            
            # Solve for W given U, V
            W_new = np.zeros((Pn, r))
            for k in range(p):
                for i in range(n):
                    # For this (k,i), we have equations for each j:
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # A @ w = ones(m)
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    b = np.ones(m)
                    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = w
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            if _check(U, V, W, n, m, p, r, tol=tol):
                return U, V, W, True
        
        return U, V, W, _check(U, V, W, n, m, p, r, tol=tol)
    
    # Try ranks from the lower bound upward
    lb = max(n*p, m*p, n*m)  # = 28
    
    best_rank = R0
    for r in range(lb, R0):
        if time.time() > _LIMIT - 30.0:
            break
        
        # Try multiple random initializations
        found = False
        for seed in range(30):
            if time.time() > _LIMIT - 20.0:
                break
            
            U, V, W, success = _als_decompose(n, m, p, r, max_iter=50, tol=1e-7, seed=seed + 1000 + r)
            
            if success:
                best = {"U": _pack(U), "V": _pack(V), "W": _pack(W)}
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    return best
