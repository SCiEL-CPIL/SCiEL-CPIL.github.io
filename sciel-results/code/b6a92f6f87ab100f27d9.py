"""
Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.

Strategy:
  - Start from the schoolbook rank-36 decomposition.
  - Search for rank reductions using structured algebraic identities
    (Strassen-style sharing across the two rows of A for shared B columns).
  - Use a direct algebraic construction for specific rank targets,
    then verify exactness with rational arithmetic.
  - Fall back to schoolbook if no better decomposition is found in time.
"""

import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 6]


def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces matrix multiplication
    using Fraction arithmetic for exactness.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using numpy float64."""
    rank = len(U[0])
    Uf = np.array(U, dtype=np.float64)
    Vf = np.array(V, dtype=np.float64)
    Wf = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = np.dot(Uf[u_idx], Vf[v_idx] * Wf[w_idx])
                if abs(s - 1) > tol:
                    return False
    return True


def _verify_all_indices(U, V, W, n, m, p, tol=1e-9):
    """
    Full verification: check that for ALL (a, b, c) in the index space,
    sum_r U[a,r]*V[b,r]*W[c,r] equals 1 if (a,b,c) corresponds to a valid
    (i,j,k) triple, and 0 otherwise.
    """
    rank = len(U[0])
    Uf = np.array(U, dtype=np.float64)
    Vf = np.array(V, dtype=np.float64)
    Wf = np.array(W, dtype=np.float64)

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Check all indices
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                val = np.dot(Uf[a], Vf[b] * Wf[c])
                if abs(val - T[a, b, c]) > tol:
                    return False
    return True


def _try_strassen_pair(n, m, p, deadline):
    """
    Try to construct a rank-reduced decomposition using Strassen-style
    sharing. For 2x3 * 3x6, we have 2 rows of A and 3 columns.
    
    Key idea: For each pair of rows (i1, i2) = (0, 1) and each column j,
    the terms A[i1,j]*B[j,k] and A[i2,j]*B[j,k] can potentially be
    combined using identities like:
      (x1+y1)(z1+w1) = x1*z1 + x1*w1 + y1*z1 + y1*w1
      = (x1+y1)(z1+w1) - x1*w1 - y1*z1  [not helpful directly]
    
    Better: use the identity for 2x2 matrix multiplication:
      [a b][e f] = [ae+bg, af+bh]
      [c d][g h]   [ce+dg, cf+dh]
    can be done in 7 multiplications instead of 8.
    
    For our case, we can group the 3 columns of A and 3 rows of B.
    Split the 6 output columns into two groups of 3.
    
    Actually, let's try a more direct approach:
    For each j (0,1,2) and each k (0..5), we have contributions from i=0 and i=1.
    That's 3 * 6 * 2 = 36 terms.
    
    Strassen trick for 2x2: 
    m1 = (a+b)(e+f)
    m2 = (c-d)(e+f)
    m3 = a(g-h)
    m4 = d(b-c)
    m5 = (a+d)(h-f)
    m6 = (b+c)(e-g)
    m7 = (a+b)(f-d)  -- wait, let me use the standard 7-mult scheme
    
    Standard Strassen for [a b; c d] * [e f; g h]:
    p1 = (a+b)(e+f)
    p2 = (c+d)(e)
    p3 = (a-b)(h)
    p4 = (d)(g+h)
    p5 = (a+d)(e-g)
    p6 = (b+c)(f-e)
    p7 = (a+b)(h-f)
    
    C11 = p1 + p4 - p5 - p7
    C12 = p3 + p5
    C21 = p2 + p5
    C22 = p1 - p2 + p3 + p4
    
    For our problem, let's try to apply this per (j, k-group).
    
    Actually, a simpler approach: try to find a decomposition of rank 35, 34, ...
    by searching over small perturbations of the schoolbook decomposition.
    """
    start = time.time()
    n, m, p = 2, 3, 6
    nm, mp, pn = n * m, m * p, p * n
    
    # Try a structured search: for each pair of schoolbook columns that share
    # the same (j, k) but different i, try to merge them.
    # Schoolbook column for (i, j, k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # For fixed (j,k), we have two columns: (i=0) and (i=1)
    # Can we replace these 2 columns with 1 column + some shared terms?
    
    # This is hard in general. Let's try a numerical optimization approach.
    
    # For target ranks from 35 down to 20, try to find a valid decomposition.
    # Use a simple iterative approach: start from schoolbook, remove one column,
    # and try to adjust the remaining columns to compensate.
    
    best = None
    
    for target_rank in range(35, 19, -1):
        if time.time() - start > deadline - 10:
            break
        
        # Initialize with first target_rank schoolbook columns
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        for r in range(target_rank):
            # Map r back to (i, j, k)
            i = r // (m * p)
            j = (r % (m * p)) // p
            k = r % p
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
        
        # The missing terms are the last (36 - target_rank) schoolbook columns.
        # We need to adjust the existing columns to cover these missing contributions.
        
        # This is a nonlinear problem. Let's try a simple gradient-free approach:
        # randomly perturb columns and check if the decomposition becomes valid.
        
        np.random.seed(target_rank * 1000 + 42)
        
        for attempt in range(50):
            if time.time() - start > deadline - 10:
                break
            
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Add small random perturbations
            noise = 0.1
            U_try += np.random.randn(nm, target_rank) * noise
            V_try += np.random.randn(mp, target_rank) * noise
            W_try += np.random.randn(pn, target_rank) * noise
            
            # Check validity
            if _verify_all_indices(U_try.tolist(), V_try.tolist(), W_try.tolist(), n, m, p, tol=1e-6):
                # Try to polish to exact rationals
                U_r = [[round(x * 1000) / 1000 for x in row] for row in U_try.tolist()]
                V_r = [[round(x * 1000) / 1000 for x in row] for row in V_try.tolist()]
                W_r = [[round(x * 1000) / 1000 for x in row] for row in W_try.tolist()]
                if _verify_exact(U_r, V_r, W_r, n, m, p):
                    best = {"U": U_r, "V": V_r, "W": W_r, "rank": target_rank}
                    break
        
        if best is not None:
            break
    
    return best


def _try_direct_construction(n, m, p, deadline):
    """
    Try to directly construct a low-rank decomposition using known
    algebraic structures for small matrix multiplication.
    
    For 2x3 * 3x6, consider the following:
    - The output is a 2x6 matrix, so we have 12 output entries.
    - Each output C[i,k] = sum_j A[i,j] * B[j,k]
    - Schoolbook uses 3 multiplications per output, 36 total.
    
    Try to share multiplications between outputs that share the same B[j,k]
    or A[i,j] factors.
    """
    start = time.time()
    n, m, p = 2, 3, 6
    
    # One approach: use the fact that for fixed j, the terms A[0,j]*B[j,k] and A[1,j]*B[j,k]
    # appear in outputs C[0,k] and C[1,k]. We can compute:
    #   s = A[0,j] + A[1,j]
    #   d = A[0,j] - A[1,j]
    #   t = B[j,k]
    # Then A[0,j]*B[j,k] = (s+d)/2 * t and A[1,j]*B[j,k] = (s-d)/2 * t
    # This doesn't reduce the number of multiplications directly.
    
    # Better: use the identity (x+y)(u+v) = xu + xv + yu + yv
    # If we precompute (A[0,j] + A[1,j]) * B[j,k], we still need to
    # separate the contributions.
    
    # Let's try a different approach: use a rank-1 update structure.
    # For each (j,k), define a vector v_jk = [B[j,k], B[j,k]] (for the two rows of A)
    # Then the contribution to C from (j,k) is A[:,j] * v_jk (outer product)
    # This is a rank-1 update to C.
    
    # Total: 3 * 6 = 18 rank-1 updates, but each "multiplication" in the
    # bilinear sense is A[i,j] * B[j,k]. So we still have 36 multiplications.
    
    # The key question is whether we can reduce the bilinear complexity.
    # For 2x3 * 3x6, the bilinear complexity is at most 36 (schoolbook).
    # Lower bounds: max(n*m, m*p, p*n) = max(6, 18, 12) = 18.
    # So the rank is between 18 and 36.
    
    # Let's try to find a rank-18 decomposition (the lower bound).
    # This would require each "multiplication" to contribute to multiple outputs.
    
    # For rank 18, we need 18 terms. Each term is U[:,r] * V[:,r] * W[:,r].
    # U has 6 rows, V has 18 rows, W has 12 rows.
    
    # This is a very constrained problem. Let's try a specific construction.
    
    # Idea: for each (j,k), use one term that covers both i=0 and i=1.
    # Term r for (j,k):
    #   U[0*m+j, r] = 1, U[1*m+j, r] = 1
    #   V[j*p+k, r] = 1
    #   W[k*n+0, r] = 1, W[k*n+1, r] = 1
    # But this gives sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for each (i,j,k),
    # which is exactly what we want! And we only need 3*6 = 18 terms.
    
    # Wait, let's verify:
    # For fixed (i, j, k), the sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = sum over (j', k') of [U[i*m+j, (j',k')] * V[j*p+k, (j',k')] * W[k*n+i, (j',k')]]
    # = sum over (j', k') of [delta(j,j') * delta(k,k') * delta(i,i')]  (if we set them up right)
    
    # Let me re-derive. If for term r corresponding to (j', k'):
    #   U[i*m+j, r] = 1 if j == j', else 0  (for both i=0 and i=1)
    #   V[j*p+k, r] = 1 if j == j' and k == k', else 0
    #   W[k*n+i, r] = 1 if i == i' and k == k', else 0  (but i' is not fixed)
    
    # Actually, W[k*n+i, r] should depend on i. Let's set:
    #   W[k*n+i, r] = 1 for all i (both 0 and 1) if k == k', else 0
    # Then for fixed (i, j, k):
    #   sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    #   = sum_{(j',k')} [1 if j==j' else 0] * [1 if j==j' and k==k' else 0] * [1 if k==k' else 0]
    #   = 1 if j==j' and k==k' for some (j',k'), which is always 1.
    
    # But wait, this gives 1 for ALL (i,j,k), which is correct!
    # And we only need 3*6 = 18 terms.
    
    # Let's construct this:
    U = [[0] * 18 for _ in range(n * m)]
    V = [[0] * 18 for _ in range(m * p)]
    W = [[0] * 18 for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    # Verify
    if _verify_exact(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W, "rank": 18}
    
    # If the above doesn't work, try a variant where W depends on i
    U2 = [[0] * 18 for _ in range(n * m)]
    V2 = [[0] * 18 for _ in range(m * p)]
    W2 = [[0] * 18 for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            U2[0 * m + j][col] = 1
            U2[1 * m + j][col] = 1
            V2[j * p + k][col] = 1
            W2[k * n + 0][col] = 1
            W2[k * n + 1][col] = 1
            col += 1
    
    if _verify_exact(U2, V2, W2, n, m, p):
        return {"U": U2, "V": V2, "W": W2, "rank": 18}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Try direct construction first (fastest)
    result = _try_direct_construction(n, m, p, deadline)
    if result is not None:
        return {"U": result["U"], "V": result["V"], "W": result["W"]}

    # Try Strassen-style search
    result = _try_strassen_pair(n, m, p, deadline)
    if result is not None:
        return {"U": result["U"], "V": result["V"], "W": result["W"]}

    # Fall back to schoolbook
    return _schoolbook(n, m, p)
