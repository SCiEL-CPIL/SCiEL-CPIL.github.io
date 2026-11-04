"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    Strategy:
    1. Use known results from algebraic complexity theory. The minimum rank for 3x3 by 3x7
       is known to be at most the schoolbook rank (63), but potentially lower.
    2. We try to find a decomposition using structured patterns based on Strassen-like ideas.
    3. For small matrices, we can use exhaustive search or structured ansatz over rational numbers.
    
    Since finding the true minimum rank is NP-hard in general, we use a heuristic approach:
    - Start with schoolbook (rank 63)
    - Try to find lower rank decompositions using linear algebra over rationals
    - Verify exactness
    
    For 3x3 * 3x7, the tensor has 3*3*7 = 63 entries. The schoolbook decomposition has rank 63.
    It is known that for matrix multiplication, the rank can sometimes be reduced.
    
    Let's use a systematic approach: try to find a rank-r decomposition by solving the
    system of equations for the tensor entries. We'll start with small ranks and increase.
    
    However, given the 120s timeout, we need to be smart. Let's use known bounds and
    structured decompositions.
    
    Actually, for this specific case, let's use the fact that we can group operations.
    A 3x3 times 3x7 can be seen as 7 separate 3x3 multiplications, but that's not helpful.
    
    Let's try a different approach: use the LLL algorithm or rational reconstruction to find
    a low-rank decomposition.
    
    Given the constraints, let's implement a search that tries to find a decomposition
    with rank < 63 by exploiting linearity. We'll use numpy for linear algebra.
    
    The tensor T[i,j,k] = delta (for the multiplication tensor) where:
    T[i,j,k] = 1 if i is row of A, j is col of A/row of B, k is col of B
    Actually, the multiplication tensor is:
    (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    The tensor T has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    T[i,j,k] = 1 if we're computing (A*B)_{i,k} using A_{i,j} and B_{j,k}
    
    Wait, the tensor is defined such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = (A*B)_{i,k} when A and B are inputs
    
    Actually, the standard formulation:
    The bilinear map is: (A, B) -> A*B
    where A is n x m, B is m x p, result is n x p.
    
    The tensor T is of shape (n*m, m*p, n*p) with:
    T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = i'*n+k' and i=i', j=j, k=k'
    i.e., T[i*m+j, j*p+k, i*n+k] = 1 for all i,j,k
    
    So the tensor has exactly n*m*p = 63 non-zero entries, each being 1.
    
    We want to find rank-r decomposition: T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{n*p}
    
    This is equivalent to finding U (n*m x r), V (m*p x r), W (n*p x r) such that
    for all a, b, c: T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]
    
    Let's try to find such a decomposition with minimal r.
    
    Given the 120s limit, let's try a few approaches:
    1. Check if r=63 works (schoolbook) - yes, by construction
    2. Try to find r < 63 using a randomized approach or structured ansatz
    
    For now, let's implement a solver that tries to find a decomposition with a given rank
    using a greedy approach or by solving the nonlinear system.
    
    Actually, let's use a known result: the rank of the 3x3x3 matrix multiplication tensor
    is 7 (Strassen). For 3x3x7, we can potentially do better than 63.
    
    Let's try to construct a decomposition by grouping the k-index.
    For each k, we have a 3x3 multiplication. If we use Strassen for each, that's 7*7=49.
    But we can share computations across k.
    
    Let's implement a search over small ranks using a linear programming approach over
    a finite field, as suggested.
    """
    n, m, p = DIMENSIONS
    
    # Build the tensor T of shape (n*m, m*p, n*p)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = i*n+k for some i,j,k
    T = np.zeros((n * m, m * p, n * p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try to find a decomposition with rank r
    # We'll try r from a lower bound up to n*m*p
    # Lower bound: max(n*m, m*p, n*p) = max(9, 21, 21) = 21
    # Upper bound: n*m*p = 63
    
    # For each rank r, try to find U, V, W
    # This is a hard problem. Let's use a heuristic.
    
    # Approach: Use the fact that we can express the multiplication as a sum of
    # rank-1 tensors. We'll try to find a structured decomposition.
    
    # For 3x3x7, let's try to use a decomposition based on the 3x3x3 Strassen algorithm
    # applied to each block, with some sharing.
    
    # Actually, let's just return the schoolbook decomposition for now, since finding
    # a lower rank decomposition is non-trivial and may not finish in 120s.
    
    # But let's try a simple improvement: group by j (the shared index).
    # For each j, we have a 3x1 times 1x7 multiplication, which is rank 3*7=21.
    # Total: 3*21 = 63. Same as schoolbook.
    
    # Let's try a different grouping. For each i and k, we have a 1x3 times 3x1.
    # That's rank 3 per (i,k) pair, total 3*7*3 = 63. Same.
    
    # Let's use a randomized search to see if we can find a lower rank decomposition.
    # We'll try ranks from 30 to 63 and see if we can find a valid one.
    
    import random
    import time
    
    start_time = time.time()
    time_limit = 110  # Leave some margin
    
    # Try to find a decomposition with rank r
    # We'll use a simple iterative approach: initialize U, V, W randomly,
    # then try to minimize the error.
    
    best_rank = n * m * p  # Schoolbook rank
    best_decomp = None
    
    # First, set up the schoolbook as baseline
    U_school = np.zeros((n * m, n * m * p), dtype=float)
    V_school = np.zeros((m * p, n * m * p), dtype=float)
    W_school = np.zeros((n * p, n * m * p), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    # Verify schoolbook
    T_recon = np.zeros_like(T, dtype=float)
    for t in range(n * m * p):
        T_recon += np.outer(np.outer(U_school[:, t], V_school[:, t]), W_school[:, t])
    
    # The schoolbook should be exact
    assert np.allclose(T_recon, T.astype(float)), "Schoolbook verification failed"
    
    # Try to find a lower rank decomposition
    # We'll use a simple approach: try random initializations and use gradient descent
    # or a simple iterative method.
    
    # Given the time constraint, let's try a few random searches for lower ranks.
    
    def try_rank(r, num_trials=100):
        """Try to find a rank-r decomposition."""
        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                return None
            
            # Initialize U, V, W randomly
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(n * p, r)
            
            # Simple iterative refinement
            # We want to minimize sum_{a,b,c} (T[a,b,c] - sum_t U[a,t]*V[b,t]*W[c,t])^2
            # This is a non-convex problem. Let's use a simple alternating minimization.
            
            for it in range(50):
                if time.time() - start_time > time_limit:
                    return None
                
                # Update U: for each a, U[a,:] should satisfy
                # sum_t U[a,t] * (V[:,t] * W[:,t])[b,c] = T[a,b,c] for all b,c
                # This is a linear system: U[a,:] @ (V*W) = T[a,:,:].flatten()
                # where (V*W)[t, (b,c)] = V[b,t] * W[c,t]
                
                # Build the matrix M_U of shape (m*p*n*p, r)
                # M_U[(b,c), t] = V[b,t] * W[c,t]
                M_U = np.zeros((m * p * n * p, r))
                for b in range(m * p):
                    for c in range(n * p):
                        for t in range(r):
                            M_U[b * n * p + c, t] = V[b, t] * W[c, t]
                
                # For each a, solve U[a,:] @ M_U = T[a,:,:].flatten()
                for a in range(n * m):
                    target = T[a, :, :].flatten()
                    # Solve least squares
                    try:
                        U[a, :] = np.linalg.lstsq(M_U, target, rcond=None)[0]
                    except:
                        pass
                
                # Similarly for V and W
                M_V = np.zeros((n * m * n * p, r))
                for a in range(n * m):
                    for c in range(n * p):
                        for t in range(r):
                            M_V[a * n * p + c, t] = U[a, t] * W[c, t]
                
                for b in range(m * p):
                    target = T[:, b, :].reshape(n * m, n * p).flatten()
                    try:
                        V[b, :] = np.linalg.lstsq(M_V, target, rcond=None)[0]
                    except:
                        pass
                
                M_W = np.zeros((n * m * m * p, r))
                for a in range(n * m):
                    for b in range(m * p):
                        for t in range(r):
                            M_W[a * m * p + b, t] = U[a, t] * V[b, t]
                
                for c in range(n * p):
                    target = T[:, :, c].reshape(n * m, m * p).flatten()
                    try:
                        W[c, :] = np.linalg.lstsq(M_W, target, rcond=None)[0]
                    except:
                        pass
            
            # Check if the decomposition is exact
            T_recon = np.zeros_like(T, dtype=float)
            for t in range(r):
                T_recon += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
            
            # Check exactness with tolerance
            if np.allclose(T_recon, T.astype(float), atol=1e-10):
                # Verify with exact arithmetic using fractions
                from fractions import Fraction
                U_frac = [[Fraction(U[i, t]).limit_denominator(10**6) for t in range(r)] for i in range(n * m)]
                V_frac = [[Fraction(V[i, t]).limit_denominator(10**6) for t in range(r)] for i in range(m * p)]
                W_frac = [[Fraction(W[i, t]).limit_denominator(10**6) for t in range(r)] for i in range(n * p)]
                
                exact_ok = True
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(n * p):
                            val = sum(U_frac[a][t] * V_frac[b][t] * W_frac[c][t] for t in range(r))
                            expected = T[a, b, c]
                            if abs(val - expected) > 1e-10:
                                exact_ok = False
                                break
                        if not exact_ok:
                            break
                    if not exact_ok:
                        break
                if exact_ok:
                    return (U_frac, V_frac, W_frac)
        
        return None
    
    # Try ranks from 30 to 62
    for r in range(30, 63):
        if time.time() - start_time > time_limit:
            break
        
        result = try_rank(r, num_trials=20)
        if result is not None:
            U_frac, V_frac, W_frac = result
            best_rank = r
            best_decomp = (U_frac, V_frac, W_frac)
            break
    
    if best_decomp is not None:
        U_frac, V_frac, W_frac = best_decomp
        # Convert to lists for return
        U_list = [[float(x) for x in row] for row in U_frac]
        V_list = [[float(x) for x in row] for row in V_frac]
        W_list = [[float(x) for x in row] for row in W_frac]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        U_list = [[float(x) for x in row] for row in U_school]
        V_list = [[float(x) for x in row] for row in V_school]
        W_list = [[float(x) for x in row] for row in W_school]
        return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
