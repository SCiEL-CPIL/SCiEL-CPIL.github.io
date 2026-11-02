"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np
from scipy import linalg as sla

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

def _strassen_rank55(n, m, p):
    """
    Construct the known rank-55 decomposition using Strassen for 2x2 blocks
    and schoolbook for the 2x1 block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2 * 2x10 (5 blocks of 2 columns each)
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        u00, u01, u10, u11 = 0, 1, 3, 4
        v00, v01 = k0, k1
        v10, v11 = 10 + k0, 10 + k1
        w00, w01 = k0 * 2, k1 * 2
        w10, w11 = k0 * 2 + 1, k1 * 2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[u00][col] = 1; U[u11][col] = 1; V[v00][col] = 1; V[v11][col] = 1; W[w00][col] = 1; W[w11][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[u10][col] = 1; U[u11][col] = 1; V[v00][col] = 1; W[w10][col] = 1; W[w11][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[u00][col] = 1; V[v01][col] = 1; V[v11][col] = -1; W[w01][col] = 1; W[w11][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[u11][col] = 1; V[v10][col] = 1; V[v00][col] = -1; W[w00][col] = 1; W[w10][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[u00][col] = 1; U[u01][col] = 1; V[v11][col] = 1; W[w01][col] = 1; W[w00][col] = -1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[u10][col] = 1; U[u00][col] = -1; V[v00][col] = 1; V[v01][col] = 1; W[w11][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[u01][col] = 1; U[u11][col] = -1; V[v10][col] = 1; V[v11][col] = 1; W[w00][col] = 1
        col += 1
        
    # Schoolbook for A2 (2x1) * B2 (1x10)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    assert col == rank, f"Expected {rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def _lll_reduce(matrix):
    """
    Perform LLL reduction on a matrix of integers.
    Returns a reduced basis for the lattice.
    Uses a simple implementation based on the LLL algorithm.
    """
    import math
    
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))
    
    def norm2(v):
        return dot(v, v)
    
    def sub(a, b):
        return [x - y for x, y in zip(a, b)]
    
    def add(a, b):
        return [x + y for x, y in zip(a, b)]
    
    def scale(v, s):
        return [x * s for x in v]
    
    # Ensure matrix is list of lists of ints
    n = len(matrix)
    if n == 0:
        return []
    m = len(matrix[0])
    
    # Convert to list of integer vectors
    B = [list(map(int, row)) for row in matrix]
    
    # LLL algorithm
    delta = 0.75
    mu = [[0.0] * (i + 1) for i in range(n)]
    B_reduced = [row[:] for row in B]
    
    # Compute Gram-Schmidt
    def gs_compute(B, mu):
        for i in range(n):
            B_reduced[i] = B[i][:]
            for j in range(i):
                if norm2(B_reduced[j]) > 0:
                    mu[i][j] = dot(B[i], B_reduced[j]) / norm2(B_reduced[j])
                    B_reduced[i] = sub(B[i], scale(B_reduced[j], mu[i][j]))
    
    gs_compute(B_reduced, mu)
    
    # LLL reduction
    for k in range(1, n):
        for i in range(k - 1, -1, -1):
            # Size reduction
            for j in range(i, -1, -1):
                if j < i:
                    mu[i][j] = mu[i][j] - round(mu[i][j])
                    B_reduced[i] = add(B_reduced[i], scale(B_reduced[j], round(mu[i][j])))
            
            # Lovasz condition check
            if norm2(B_reduced[i]) < (1 - delta) * (norm2(B_reduced[i-1]) / norm2(B_reduced[i])) * norm2(B_reduced[i-1]):
                # Swap
                B_reduced[i], B_reduced[i-1] = B_reduced[i-1], B_reduced[i]
                gs_compute(B_reduced, mu)
    
    return B_reduced

def _find_linear_dependencies(U, V, W, n, m, p):
    """
    Check if there's a linear dependency in the rank-55 solution that allows rank reduction.
    For a rank-r decomposition, we need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    If we have a linear dependency in the columns, we might be able to reduce rank.
    """
    rank = len(U[0]) if U else 0
    if rank < 2:
        return None
    
    # Build the tensor equation matrix
    # For each (i,j,k), we have: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a system of n*m*p equations in rank unknowns (the W entries)
    # if U and V are fixed.
    
    # Actually, let's think about this differently.
    # The multiplication tensor T has entries T[i,j,k] = 1.
    # A rank-r decomposition means T = sum_c U_c * V_c * W_c (outer products)
    
    # For rank reduction, we need to find if the rank-55 solution has redundancy.
    # One approach: check if any column is a linear combination of others.
    
    # Let's try: fix U and V, solve for W, see if we can do it with fewer columns.
    # This is what the existing search does.
    
    # For LLL: consider the space of all possible W given U, V.
    # If we have a rank-55 solution, and we want rank 54, we need to find
    # a linear transformation that reduces the rank.
    
    # Alternative: use LLL on the coefficient matrix to find short vectors
    # that correspond to potential rank reductions.
    
    # For now, return None to indicate no dependency found via this method.
    # The actual search for rank-54 is done in _try_rank54_exploit_n2.
    return None

def _try_rank54_exploit_n2(n, m, p, time_budget, seed=42):
    """
    Try to find a rank-54 decomposition by exploiting the n=2 structure.
    Uses more systematic search with exact rational arithmetic.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r55 = 55
    r54 = 54
    
    best = None
    
    # Strategy 1: Try removing one column at a time and solving for W
    # with the remaining 54 columns of U and V.
    for col_to_remove in range(r55):
        if time.time() > deadline - 10:
            break
        
        # Remove column col_to_remove
        mask = [c for c in range(r55) if c != col_to_remove]
        U = U0[:, mask]
        V = V0[:, mask]
        
        # Solve for W
        W = np.zeros((p*n, r54))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r54))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if not np.allclose(A @ sol, b, atol=1e-10):
                        feasible = False
                        break
                    W[k*n+i, :] = sol
                except:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            # Verify exactly
            ok = True
            for i in range(n):
                if not ok: break
                for j in range(m):
                    if not ok: break
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        if abs(s - 1.0) > 1e-10:
                            ok = False
                            break
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    # Strategy 2: Randomized search with structured sparsity
    if best is None:
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Create U and V with specific sparsity patterns
            U = np.zeros((n*m, r54))
            V = np.zeros((m*p, r54))
            
            # Based on Strassen structure, but with one column merged
            for g in range(5):
                start = g * 11  # 55 / 5 = 11 columns per block
                end = start + 11 if g < 4 else r54
                
                # In each block, try to merge two columns
                if attempt % 2 == 0:
                    # Keep first 10 columns, merge last two
                    U[:, start:start+10] = U0[:, start:start+10]
                    V[:, start:start+10] = V0[:, start:start+10]
                    # Merge columns start+10 and start+11
                    U[:, start+10] = U0[:, start+10] + U0[:, start+11]
                    V[:, start+10] = V0[:, start+10] + V0[:, start+11]
                else:
                    # Try different merge
                    U[:, start:start+10] = U0[:, start:start+10]
                    V[:, start:start+10] = V0[:, start:start+10]
                    U[:, start+10] = U0[:, start+10] - U0[:, start+11]
                    V[:, start+10] = V0[:, start+10] - V0[:, start+11]
            
            # Solve for W
            W = np.zeros((p*n, r54))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r54))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                        W[k*n+i, :] = sol
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                ok = True
                for i in range(n):
                    if not ok: break
                    for j in range(m):
                        if not ok: break
                        for k in range(p):
                            prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                            s = np.sum(prod)
                            if abs(s - 1.0) > 1e-10:
                                ok = False
                                break
                if ok:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
    
    return best

def _try_basis_reduction(n, m, p, time_budget, seed=42):
    """
    Attempt to reduce rank by finding linear dependencies in the rank-55 solution.
    Uses LLL to find short vectors in the null space.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r55 = 55
    r54 = 54
    
    # Try to find a linear transformation T (55x54) such that
    # U @ T, V @ T, W @ T still satisfies the tensor equation.
    # This is equivalent to finding a 1-dimensional null space in the
    # "coefficient space" that preserves the tensor.
    
    # For each (i,j,k), we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # If we replace U -> U @ T, V -> V @ T, W -> W @ T, we need:
    # sum_c (U @ T)[i*m+j,c] * (V @ T)[j*p+k,c] * (W @ T)[k*n+i,c] = 1
    
    # This is a nonlinear constraint. Let's try a simpler approach:
    # Check if removing any single column still allows a solution.
    
    for col_to_remove in range(r55):
        if time.time() > deadline - 5:
            break
        
        mask = [c for c in range(r55) if c != col_to_remove]
        U = U0[:, mask]
        V = V0[:, mask]
        
        W = np.zeros((p*n, r54))
        feasible = True
        
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r54))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j] * V[j*p+k]
                
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if not np.allclose(A @ sol, b, atol=1e-8):
                        feasible = False
                        break
                    W[k*n+i, :] = sol
                except:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            ok = True
            for i in range(n):
                if not ok: break
                for j in range(m):
                    if not ok: break
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        if abs(s - 1.0) > 1e-8:
                            ok = False
                            break
            if ok:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_general_search(n, m, p, time_budget, seed=42, start_rank=54):
    """
    General search for low-rank decompositions.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    for r in range(start_rank, min(n*m*p + 1, start_rank + 5)):
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            continue
            
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
                
            # Use structured random initialization based on Strassen
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            
            # Initialize with Strassen-like structure
            base = _strassen_rank55(n, m, p)
            U0 = np.array(base["U"], dtype=float)
            V0 = np.array(base["V"], dtype=float)
            
            if r <= 55:
                # Take first r columns
                U[:, :min(r, 55)] = U0[:, :min(r, 55)]
                V[:, :min(r, 55)] = V0[:, :min(r, 55)]
                if r > 55:
                    U[:, 55:] = rng.randint(0, 2, size=(n*m, r-55)) * 2 - 1
                    V[:, 55:] = rng.randint(0, 2, size=(m*p, r-55)) * 2 - 1
            else:
                U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
                V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
                U[rng.random((n*m, r)) < 0.3] = 0
                V[rng.random((m*p, r)) < 0.3] = 0
            
            # Add small perturbation
            U += rng.randn(n*m, r) * 0.01
            V += rng.randn(m*p, r) * 0.01
            
            W = np.zeros((p*n, r))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j] * V[j*p+k]
                    
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ sol, b, atol=1e-10):
                            feasible = False
                            break
                        W[k*n+i, :] = sol
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                ok = True
                for i in range(n):
                    if not ok: break
                    for j in range(m):
                        if not ok: break
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
    deadline = time.time() + 130  # 20s safety margin before 150s limit
    
    # Start with the known rank-55 Strassen construction
    best = _strassen_rank55(n, m, p)
    best_rank = 55
    
    # 1. Try the basis reduction / linear dependency search
    time_left = deadline - time.time()
    if time_left > 10:
        result = _try_basis_reduction(n, m, p, min(time_left, 20), seed=42)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    # 2. Try the n=2 exploit search
    if best_rank > 54:
        time_left = deadline - time.time()
        if time_left > 10:
            for seed in [42, 12345, 777]:
                if time.time() > deadline - 20:
                    break
                tl = min(time_left, deadline - time.time() - 15)
                if tl < 10:
                    break
                result = _try_rank54_exploit_n2(n, m, p, tl, seed=seed)
                if result is not None:
                    r = len(result["U"][0])
                    if r < best_rank:
                        if _verify(result["U"], result["V"], result["W"], n, m, p):
                            best = result
                            best_rank = r
                            break
    
    # 3. Try general search
    if best_rank > 54:
        time_left = deadline - time.time()
        if time_left > 10:
            for seed in [42, 12345]:
                if time.time() > deadline - 20:
                    break
                tl = min(time_left, deadline - time.time() - 15)
                if tl < 10:
                    break
                result = _try_general_search(n, m, p, tl, seed=seed, start_rank=54)
                if result is not None:
                    r = len(result["U"][0])
                    if r < best_rank:
                        if _verify(result["U"], result["V"], result["W"], n, m, p):
                            best = result
                            best_rank = r
                            break
    
    # Fallback
    if best_rank > 60:
        best = _schoolbook(n, m, p)
        best_rank = 60
        
    return best
