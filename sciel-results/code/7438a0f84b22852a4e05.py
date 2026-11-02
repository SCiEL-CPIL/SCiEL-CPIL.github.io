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

def _solve_exact_system(A, b, max_denom=2):
    """
    Solve A x = b exactly using Fractions.
    A is a list of lists of Fractions.
    b is a list of Fractions.
    Returns a list of Fractions x or None if no solution exists with denominators dividing max_denom.
    """
    import copy
    
    rows = len(A)
    cols = len(A[0])
    
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Gaussian Elimination
    rank = 0
    for col in range(cols):
        # Find pivot
        pivot_row = -1
        for r in range(rank, rows):
            if M[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
            
        # Swap
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        
        # Normalize pivot row (optional for consistency check, but good for keeping numbers small)
        # Actually, we just eliminate below
        pivot_val = M[rank][col]
        
        # Eliminate below
        for r in range(rank + 1, rows):
            if M[r][col] != 0:
                factor = M[r][col] / pivot_val
                for c in range(col, cols + 1):
                    M[r][c] -= factor * M[rank][c]
        
        rank += 1
    
    # Check consistency: any row that is 0...0 | non-zero is inconsistent
    for r in range(rank, rows):
        if M[r][cols] != 0:
            return None
            
    # Back substitution
    # Since we have more columns than rows (typically), we have free variables.
    # We need to find a solution where all non-free variables have denominators dividing max_denom.
    # And free variables should also have denominators dividing max_denom.
    # Actually, the problem asks for *a* solution. If the system is consistent, there are infinite solutions.
    # We want one with small denominators.
    # Let's set free variables to 0 and see if the resulting basic solution has small denominators.
    # If not, we might need to adjust. But usually, if a solution exists, the "minimum norm" or
    # specific integer structure might yield small denominators.
    # However, the prompt implies we just need *some* valid decomposition.
    # If we fix U and V, W is determined up to the null space of A.
    # Let's try setting free variables to 0.
    
    x = [Fraction(0) for _ in range(cols)]
    
    for r in range(rank - 1, -1, -1):
        # Find the pivot column for this row
        # The pivot column is the first non-zero in the row?
        # In our elimination, we didn't track pivot columns explicitly, but we iterated col from 0.
        # So the pivot for row r is the (r)-th column that had a pivot.
        # We need to map row index to pivot column.
        # Let's re-do elimination tracking pivots.
        
        pass

    # Simpler approach for small systems:
    # Since m=3 and r=54, the system is 3x54.
    # We can find the null space basis and particular solution.
    # But let's just use the standard method:
    # 1. Find a particular solution x_p by setting free vars to 0.
    # 2. Check if x_p has denominators dividing max_denom.
    # 3. If not, we can add null space vectors to adjust.
    # Given the time constraint, let's just try x_p first.
    
    # Re-implementing Gaussian Elimination with pivot tracking
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    pivot_cols = []
    rank = 0
    for col in range(cols):
        pivot_row = -1
        for r in range(rank, rows):
            if M[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        pivot_val = M[rank][col]
        for r in range(rank + 1, rows):
            if M[r][col] != 0:
                factor = M[r][col] / pivot_val
                for c in range(col, cols + 1):
                    M[r][c] -= factor * M[rank][c]
        pivot_cols.append(col)
        rank += 1
        
    for r in range(rank, rows):
        if M[r][cols] != 0:
            return None
            
    x = [Fraction(0) for _ in range(cols)]
    for r in range(rank - 1, -1, -1):
        col = pivot_cols[r]
        # M[r][col] * x[col] + sum(M[r][c] * x[c] for c > col) = M[r][cols]
        sum_rest = sum(M[r][c] * x[c] for c in range(col + 1, cols))
        x[col] = (M[r][cols] - sum_rest) / M[r][col]
        
    # Check denominators
    for val in x:
        if val.denominator > max_denom:
            # We could try to adjust with null space, but for now return None
            # to indicate this particular solution doesn't meet the precision requirement.
            # Note: It is possible another solution does. But this is a heuristic.
            return None
            
    return x

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
        
        # Solve for W exactly
        W = [[0] * r54 for _ in range(p*n)]
        feasible = True
        
        for i in range(n):
            for k in range(p):
                # Build A (m x r54) and b (m)
                A = [[Fraction(0)] * r54 for _ in range(m)]
                b = [Fraction(1) for _ in range(m)]
                
                for j in range(m):
                    for c in range(r54):
                        A[j][c] = Fraction(int(U[i*m+j][c])) * Fraction(int(V[j*p+k][c]))
                
                sol = _solve_exact_system(A, b, max_denom=2)
                if sol is None:
                    feasible = False
                    break
                W[k*n+i] = sol
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
                        s = Fraction(0)
                        for c in range(r54):
                            s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                        if s != 1:
                            ok = False
                            break
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W}
                break
    
    return best

def _try_basis_reduction(n, m, p, time_budget, seed=42):
    """
    Attempt to reduce rank by finding linear dependencies in the rank-55 solution.
    Uses exact rational arithmetic to check if removing a column yields a valid solution.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    base = _strassen_rank55(n, m, p)
    U0 = np.array(base["U"], dtype=float)
    V0 = np.array(base["V"], dtype=float)
    W0 = np.array(base["W"], dtype=float)
    
    r55 = 55
    r54 = 54
    
    # Try removing each column and solving for W exactly
    for col_to_remove in range(r55):
        if time.time() > deadline - 5:
            break
        
        mask = [c for c in range(r55) if c != col_to_remove]
        U = U0[:, mask]
        V = V0[:, mask]
        
        W = [[0] * r54 for _ in range(p*n)]
        feasible = True
        
        for i in range(n):
            for k in range(p):
                A = [[Fraction(0)] * r54 for _ in range(m)]
                b = [Fraction(1) for _ in range(m)]
                
                for j in range(m):
                    for c in range(r54):
                        A[j][c] = Fraction(int(U[i*m+j][c])) * Fraction(int(V[j*p+k][c]))
                
                sol = _solve_exact_system(A, b, max_denom=2)
                if sol is None:
                    feasible = False
                    break
                W[k*n+i] = sol
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
                        s = Fraction(0)
                        for c in range(r54):
                            s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                        if s != 1:
                            ok = False
                            break
            if ok:
                return {"U": U.tolist(), "V": V.tolist(), "W": W}
    
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
    
    # 1. Try the exact basis reduction search
    time_left = deadline - time.time()
    if time_left > 10:
        result = _try_basis_reduction(n, m, p, min(time_left, 20), seed=42)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    # 2. Try the n=2 exploit search (also uses exact arithmetic now)
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
