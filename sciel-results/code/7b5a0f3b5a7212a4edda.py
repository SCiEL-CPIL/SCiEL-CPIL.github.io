"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time
import itertools

_START = time.time()
_LIMIT = 140.0  # safety margin before 150s wall-clock kill
DIMENSIONS = [3, 4, 7]


def _check_decomposition(U, V, W, n, m, p, r, tol=1e-8):
    """Check if the decomposition is exact.
    For each rank-1 term c, we need:
    sum_{i,j,k} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * delta_{i,j,k} = 0
    where delta is 1 only for the 'identity' pattern.
    
    Actually, the condition is:
    The full tensor T = sum_c U_c ⊗ V_c ⊗ W_c must equal the matmul tensor.
    For the matmul tensor, T[i,j,k] = 1 if the indices form a valid matmul path, else 0.
    But in our convention, the matmul tensor has T[i,j,k] = 1 for all (i,j,k) in range,
    and the decomposition must reproduce A@B where A[i,j]=U[i*m+j,0], etc.
    
    Wait - let me re-read. The decomposition is:
    C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    This must hold for ALL i,j,k. So for each (i,j,k):
    sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 if this is a valid matmul term, else 0.
    
    Actually no - the matmul tensor T[i,j,k] = 1 for all valid (i,j,k) indices.
    The condition is that the tensor product sum equals the matmul tensor.
    
    Let me think differently. The matmul tensor has entries T[i,j,k] where:
    - i ranges over [0,n), j over [0,m), k over [0,p)
    - T[i,j,k] = 1 (it's the structure tensor of matrix multiplication)
    
    A rank-r decomposition means:
    T[i,j,k] = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for all i,j,k.
    
    So we need: for all (i,j,k) in {0..n-1} x {0..m-1} x {0..p-1}:
    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    """
    # Build the matmul tensor and check
    # For efficiency, check each (i,j,k) triple
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
                s = 0.0
                for r in range(r):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _check_fast(U_np, V_np, W_np, n, m, p, r, tol=1e-8):
    """Fast numpy check."""
    # U_np shape: (n*m, r), V_np: (m*p, r), W_np: (p*n, r)
    # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # Vectorize: for fixed j, we can check all (i,k) pairs
    for j in range(m):
        # U indices: i*m+j for i in range(n) -> shape (n,)
        # V indices: j*p+k for k in range(p) -> shape (p,)
        # W indices: k*n+i for (k,i) -> shape (p*n,)
        u_col = U_np[np.arange(n)*m + j, :]  # (n, r)
        v_col = V_np[j*p + np.arange(p), :]   # (p, r)
        # For each (i,k): sum_r u_col[i,r] * v_col[k,r] * W[k*n+i,r]
        # W[k*n+i] for all k,i: reshape W to (p, n, r)
        W_reshaped = W_np.reshape(p, n, r)  # W_reshaped[k,i,:] = W[k*n+i,:]
        # Compute: for each (i,k): sum_r u_col[i,r]*v_col[k,r]*W_reshaped[k,i,r]
        # = sum_r (u_col[i,r]*W_reshaped[k,i,r]) * v_col[k,r]
        # Let's do: product[i,k,r] = u_col[i,r] * v_col[k,r] * W_reshaped[k,i,r]
        # Then sum over r for each (i,k)
        # u_col: (n, 1, r), v_col: (1, p, r), W_reshaped: (p, n, r) -> transpose to (n, p, r)
        W_t = W_reshaped.transpose(1, 0, 2)  # (n, p, r)
        prod = u_col[:, np.newaxis, :] * v_col[np.newaxis, :, :] * W_t  # (n, p, r)
        sums = prod.sum(axis=2)  # (n, p)
        if np.any(np.abs(sums - 1.0) > tol):
            return False
    return True


def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
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
    return U, V, W, rank


def _try_structured_search(n, m, p, deadline):
    """Try structured low-rank decompositions."""
    best_rank = n * m * p
    best_decomp = None
    
    # Lower bound: max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    lower_bound = max(n * p, m * p, n * m)
    
    # Try ranks from lower_bound upward
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Strategy 1: Try random integer/half-integer entries
        # Use multiple random seeds
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(r * 1000 + attempt)
            
            # Generate random matrices with small integer/half-integer entries
            scale = 2  # entries in {-1, -0.5, 0, 0.5, 1}
            U_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
            
            if _check_fast(U_rand, V_rand, W_rand, n, m, p, r):
                best_rank = r
                best_decomp = (U_rand.tolist(), V_rand.tolist(), W_rand.tolist())
                break
        
        if best_decomp is not None:
            break
        
        # Strategy 2: Try structured patterns
        # Pattern: use Kronecker-like structures
        if r >= n * p:  # at least 21
            # Try: U[i*m+j, r] = a[i,j,r'], V[j*p+k, r] = b[j,k,r'], W[k*n+i, r] = c[k,i,r']
            # where r' ranges over some structured set
            pass
    
    return best_rank, best_decomp


def _try_nullspace_search(n, m, p, deadline):
    """Use the null space approach from inspiration 2.
    
    The idea: if we have a valid decomposition of rank R0 (schoolbook),
    any additional rank-1 term that is orthogonal to the matmul tensor
    can be added without breaking the decomposition.
    
    But more usefully: we can try to find a lower-rank decomposition by
    solving the system of equations.
    
    The constraint is: for all (i,j,k):
    sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    """
    best_rank = n * m * p
    best_decomp = None
    
    # The number of constraints is n*m*p = 84
    # For rank r, we have r*(n*m + m*p + p*n) = r*(12+28+21) = 61r unknowns
    # For r >= 2, 61r >= 122 > 84, so the system is underdetermined for r >= 2
    # But the system is nonlinear (trilinear), so this is a tensor rank problem
    
    # Try small ranks with random initialization and local search
    lower_bound = max(n * p, m * p, n * m)  # 28
    
    for r in range(lower_bound, min(lower_bound + 20, n * m * p)):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            np.random.seed(100000 + r * 1000 + attempt)
            
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Try to refine using simple iterations
            # This is a non-convex problem, so we'll just check if random works
            if _check_fast(U, V, W, n, m, p, r):
                best_rank = r
                best_decomp = (U.tolist(), V.tolist(), W.tolist())
                found = True
                break
        
        if found:
            break
    
    return best_rank, best_decomp


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _LIMIT - 5.0  # 5s safety margin
    
    # Step 1: Build schoolbook decomposition (always valid)
    U0, V0, W0, R0 = _build_schoolbook(n, m, p)
    
    best_rank = R0
    best_U, best_V, best_W = U0, V0, W0
    
    # Step 2: Try structured search for lower ranks
    if time.time() < deadline:
        rank_found, decomp_found = _try_structured_search(n, m, p, deadline)
        if decomp_found is not None and rank_found < best_rank:
            best_rank = rank_found
            best_U, best_V, best_W = decomp_found
    
    # Step 3: Try nullspace/random search if time remains
    if time.time() < deadline:
        rank_found, decomp_found = _try_nullspace_search(n, m, p, deadline)
        if decomp_found is not None and rank_found < best_rank:
            best_rank = rank_found
            best_U, best_V, best_W = decomp_found
    
    # Ensure we return lists (not numpy arrays)
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
    if isinstance(best_V, np.ndarray):
        best_V = best_V.tolist()
    if isinstance(best_W, np.ndarray):
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
