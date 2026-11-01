"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_kronecker_strassen(n, m, p, time_budget, seed=42):
    """
    Try to exploit the structure: 2x3 * 3x10.
    
    Idea: We can view the multiplication as:
    A (2x3) * B (3x10) = C (2x10)
    
    We can try to decompose this using a Kronecker-like structure.
    Note that 2x3 * 3x10 can be seen as a block structure.
    
    One approach: Use the fact that we can multiply (2x3) by (3x10) by
    first reshaping or using a decomposition of the 3x10 part.
    
    Actually, let's think about it differently. The tensor is T[i,j,k] = 1 for all i,j,k.
    
    A key insight: We can use the identity:
    (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    For small dimensions, we can try explicit constructions.
    
    For 2x3 * 3x10:
    C_{0,k} = A_{0,0}B_{0,k} + A_{0,1}B_{1,k} + A_{0,2}B_{2,k}
    C_{1,k} = A_{1,0}B_{0,k} + A_{1,1}B_{1,k} + A_{1,2}B_{2,k}
    
    This is 2*3*10 = 60 multiplications in schoolbook.
    
    Can we do better? Let's try a recursive/structured approach.
    
    One idea: Group the columns of B. Since p=10 is even, we can split into two 5-column blocks.
    But that doesn't directly help with the 3-row dimension.
    
    Another idea: Use a decomposition based on the 3x10 matrix multiplication.
    For 1x3 * 3x10, the rank is 3 (since it's just a linear combination).
    For 2x3 * 3x10, we need to handle the 2 rows.
    
    Let's try a different approach: use the fact that we can write the multiplication
    as a sum of rank-1 terms, but with a clever grouping.
    
    Actually, let's try a direct construction. We want to find U, V, W such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    One simple construction: use the schoolbook but with some sharing.
    
    Let's try: for each (i,j), create a term that handles all k at once.
    But that would require V[j*p+k, c] to depend on k, which it does.
    
    Alternative: Use the fact that n=2 is small.
    For i=0: C_{0,k} = A_{0,0}B_{0,k} + A_{0,1}B_{1,k} + A_{0,2}B_{2,k}
    For i=1: C_{1,k} = A_{1,0}B_{0,k} + A_{1,1}B_{1,k} + A_{1,2}B_{2,k}
    
    We can try to share computations between the two rows.
    
    Let's try a specific construction with rank < 60.
    
    Actually, let's use a random search with a bias towards structured solutions.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    # Try a few candidate ranks
    candidate_ranks = list(range(40, n*m*p + 1, 5))
    candidate_ranks.sort()
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        # Try random search at this rank
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Generate random ±1 matrices with some structure
            # Bias towards having some zero entries to reduce complexity
            U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
            W = rng.randint(0, 2, size=(p*n, r)) * 2 - 1
            
            # Make some entries zero (sparse)
            if attempt % 2 == 0:
                U[rng.random((n*m, r)) < 0.3] = 0
                V[rng.random((m*p, r)) < 0.3] = 0
                W[rng.random((p*n, r)) < 0.3] = 0
            
            # Verify
            ok = True
            for i in range(n):
                if not ok:
                    break
                for j in range(m):
                    if not ok:
                        break
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        if abs(s - 1.0) > 1e-10:
                            ok = False
                            break
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
        if best is not None:
            break
    
    return best

def _try_linear_program(n, m, p, time_budget, seed=42):
    """
    Try to solve a linear program to find a valid decomposition.
    This is more systematic than random search.
    
    The constraints are:
    For all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is a system of nonlinear equations (products of variables).
    However, we can fix two of the matrices and solve for the third linearly.
    
    Strategy: Fix U and V randomly, then solve for W linearly.
    For each (i,k), we have:
    sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1
    
    Wait, that's not quite right. Let me re-derive.
    
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    For fixed (i,k), this is:
    sum_c W[k*n+i, c] * (sum_j ... ) 
    
    No, that's not separable in that way.
    
    Actually, for fixed (i,j,k), it's a single equation.
    If we fix U and V, then for each (i,k), we have m equations (one for each j):
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for j=0,...,m-1
    
    This is a linear system in W[k*n+i, :].
    If the matrix M_{j,c} = U[i*m+j, c] * V[j*p+k, c] has full rank m,
    we can solve for W[k*n+i, :].
    
    Let's try this.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    # Try a few candidate ranks
    candidate_ranks = list(range(30, n*m*p + 1, 5))
    candidate_ranks.sort()
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Fix U and V randomly
            U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
            
            # Solve for W
            W = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    # Build the linear system for W[k*n+i, :]
                    # For each j: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    # Solve A * W[k*n+i, :]^T = b
                    # Use least squares
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = sol
                        # Check if the solution is exact
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                # Verify the full solution
                ok = True
                for i in range(n):
                    if not ok:
                        break
                    for j in range(m):
                        if not ok:
                            break
                        for k in range(p):
                            prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-10:
                                ok = False
                                break
                if ok:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break
        if best is not None:
            break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try linear program approach
    time_left = deadline - time.time()
    if time_left > 10:
        result = _try_linear_program(n, m, p, time_left, seed=12345)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    # Try Kronecker/Strassen approach
    time_left = deadline - time.time()
    if time_left > 10 and best_rank == n * m * p:
        result = _try_kronecker_strassen(n, m, p, time_left, seed=99999)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    return best
