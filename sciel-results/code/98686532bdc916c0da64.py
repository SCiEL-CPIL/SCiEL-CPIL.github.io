"""Matrix multiplication 3,3,5 rank-minimal tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n=3, m=3, p=5):
    """Check if U, V, W form an exact decomposition.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Convention: W[k*n+i, r]
    """
    rank = U.shape[1]
    # Build tensor T[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For each (i,j,k), the sum over r of the product should be 1
    # For each (i,j,k,r'), the tensor entry is the product
    # Actually, the decomposition says:
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Wait, that's not right. Let me re-read.
    
    # Standard tensor decomposition for matrix multiplication:
    # (AB)_{i,k} = sum_{j=1}^m A_{i,j} B_{j,k}
    # = sum_{r=1}^R sum_{j=1}^m (U_{i,j,r} * A_{i,j}) * (V_{j,k,r} * B_{j,k}) * W_{k,i,r}
    
    # Actually the standard form is:
    # (AB)_{i,k} = sum_r U_{i,*,r} (A) * V_{*,k,r} (B) * W_{r}
    # where U_{i,*,r} means: sum_j U[i*m+j, r] * A_{i,j}
    # and V_{*,k,r} means: sum_j V[j*p+k, r] * B_{j,k}
    
    # So for each r, we have a rank-1 term in the "extended" sense.
    # The full tensor is: T[i,j,k] = delta_{jk} ... no wait.
    
    # The matrix multiplication tensor M has entries M[i,j,k] = 1 if this corresponds to (AB)_{i,k} using A_{i,j} B_{j,k}.
    # Actually M[i,j,k] is the coefficient of A_{i,j} * B_{j,k} in (AB)_{i,k}, which is 1.
    
    # A tensor decomposition of rank R means:
    # M[i,j,k] = sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # for all i,j,k.
    
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _to_integer_or_half(U, V, W, tol=1e-8):
    """Try to round entries to nearest integer or half-integer. Returns (success, U', V', W')."""
    def round_matrix(M):
        M2 = M.copy()
        for i in range(M2.shape[0]):
            for j in range(M2.shape[1]):
                # Try nearest integer
                ri = round(M2[i,j])
                # Try nearest half-integer
                rh = round(2*M2[i,j]) / 2.0
                if abs(M2[i,j] - ri) < abs(M2[i,j] - rh):
                    M2[i,j] = ri
                else:
                    M2[i,j] = rh
        return M2
    
    U2 = round_matrix(U)
    V2 = round_matrix(V)
    W2 = round_matrix(W)
    
    if _verify(U2, V2, W2):
        return True, U2, V2, W2
    return False, None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = 3, 3, 5
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 45
    
    # Best result found
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Start with schoolbook
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Known lower bound for matmul 3x3 * 3x5:
    # For 3x3*3x3, the rank is 23 (Landsberg). For 3x3*3x5, it's at least...
    # Let's just search from a reasonable range.
    # Actually, let's try to find a decomposition with rank < 45.
    
    # Strategy: use random initialization and gradient-like optimization
    # to minimize the error, then try to round.
    
    def try_random_rank(R, n_trials=5):
        nonlocal best_rank, best_U, best_V, best_W
        for trial in range(n_trials):
            if time.time() > deadline:
                return False
            rng = np.random.default_rng(trial * 1000 + R)
            U = rng.standard_normal((n*m, R))
            V = rng.standard_normal((m*p, R))
            W = rng.standard_normal((p*n, R))
            
            # Normalize columns to avoid scale issues
            for r in range(R):
                norm = np.sqrt(np.sum(U[:,r]**2) * np.sum(V[:,r]**2) * np.sum(W[:,r]**2))
                if norm > 1e-12:
                    scale = norm ** (1.0/3.0)
                    U[:,r] /= scale
                    V[:,r] /= scale
                    W[:,r] /= scale
            
            # Simple local optimization: coordinate descent
            # For each r, given fixed U[:,r'] for r'!=r, V[:,r'] for r'!=r, W[:,r'] for r'!=r,
            # the residual for (i,j,k) is: 1 - sum_{r'!=r} U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r']
            # We want to choose U[:,r], V[:,r], W[:,r] to minimize sum_{i,j,k} (residual_{ijk} + U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
            # This is a hard problem, but we can do simple updates.
            
            # Actually, let's just do a few rounds of simple averaging.
            # For each (i,j,k), the target is 1. The current contribution from all r is:
            # C[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Error[i,j,k] = 1 - C[i,j,k]
            # We can try to distribute the error across r.
            
            max_iter = 200
            for it in range(max_iter):
                if time.time() > deadline:
                    return False
                
                # Compute current contributions
                C = np.zeros((n, m, p))
                for r in range(R):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                C[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                error = 1.0 - C
                max_err = np.max(np.abs(error))
                
                if max_err < 1e-10:
                    # Success!
                    if R < best_rank:
                        best_rank = R
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    return True
                
                # Simple update: for each r, adjust one matrix at a time
                # Let's do a simple gradient step on the error.
                # dU[i*m+j, r] = -lambda * V[j*p+k, r] * W[k*n+i, r] * error[i,j,k] summed over... 
                # This is getting complicated. Let's just do a crude random perturbation.
                
                # Pick a random r and a random entry in one of the matrices, perturb it.
                r = rng.integers(0, R)
                mat_idx = rng.integers(0, 3)
                if mat_idx == 0:
                    idx = (rng.integers(0, n*m),)
                    U[idx[0], r] += 0.1 * rng.standard_normal()
                elif mat_idx == 1:
                    idx = (rng.integers(0, m*p),)
                    V[idx[0], r] += 0.1 * rng.standard_normal()
                else:
                    idx = (rng.integers(0, p*n),)
                    W[idx[0], r] += 0.1 * rng.standard_normal()
            
            # Try to round
            success, U2, V2, W2 = _to_integer_or_half(U, V, W)
            if success:
                if R < best_rank:
                    best_rank = R
                    best_U = U2
                    best_V = V2
                    best_W = W2
                return True
        
        return False
    
    # Try ranks below schoolbook
    # Known: for 3x3x3, rank 23 is optimal. For 3x3x5, let's try a range.
    for R in range(23, schoolbook_rank):
        if time.time() > deadline:
            break
        if try_random_rank(R, n_trials=3):
            break
    
    # If we found something, return it
    if best_U is not None and best_rank < schoolbook_rank:
        # Convert to lists for the interface
        return {
            "U": [[float(x) for x in row] for row in best_U],
            "V": [[float(x) for x in row] for row in best_V],
            "W": [[float(x) for x in row] for row in best_W]
        }
    
    # Fallback: return schoolbook
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
