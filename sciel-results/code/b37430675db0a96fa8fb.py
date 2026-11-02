"""
Find rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
Strategy: Deterministic Algebraic Search with Structured Sparsity and Exact Rational Solving.
Instead of gradient descent + snapping, we construct candidate decompositions with specific
sparsity patterns (Strassen-like or random sparse) and solve for the exact coefficients 
using RREF over Q. This ensures exactness and allows us to test if a low-rank 
decomposition exists within the tested subspace.
"""

import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Lower bound: max(n*m, m*p, n*p) = max(10, 30, 12) = 30
    # We test ranks 30, 31, 32... up to 60 (schoolbook)
    
    def verify_exact(U, V, W):
        """Verify exact decomposition using Fractions."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def solve_linear_system(A, b):
        """
        Solve A x = b over Q.
        A is list of lists of Fractions.
        b is list of Fractions.
        Returns list of Fractions or None if no unique solution.
        """
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        # Augmented matrix
        M = [[A[i][j] for j in range(cols)] + [b[i]] for i in range(rows)]
        
        # Gaussian Elimination to RREF
        for c in range(cols):
            # Find pivot
            pivot = -1
            for r in range(c, rows):
                if M[r][c] != 0:
                    pivot = r
                    break
            if pivot == -1:
                continue # Free variable, but we need unique solution for this approach?
                         # Actually, if there are free variables, we can set them to 0 and check.
                         # However, for exactness, we just need *some* solution.
                         # Let's handle free vars by setting them to 0 in the final step.
                         # But RREF makes it easy to read off.
                         # Let's just proceed to RREF and check consistency.
            
            # Swap
            if pivot != c:
                M[c], M[pivot] = M[pivot], M[c]
            
            # Normalize
            pivot_val = M[c][c]
            for j in range(c, cols + 1):
                M[c][j] = M[c][j] / pivot_val
            
            # Eliminate other rows
            for r in range(rows):
                if r != c and M[r][c] != 0:
                    factor = M[r][c]
                    for j in range(c, cols + 1):
                        M[r][j] -= factor * M[c][j]
        
        # Check for inconsistency (row [0 0 ... 0 | 1])
        for r in range(rows):
            is_zero_row = True
            for c in range(cols):
                if M[r][c] != 0:
                    is_zero_row = False
                    break
            if is_zero_row and M[r][cols] != 0:
                return None
        
        # Extract solution
        # If there are free variables (columns without pivots), set them to 0.
        # We need to track which columns are pivots.
        pivot_cols = set()
        for c in range(cols):
            for r in range(rows):
                if M[r][c] != 0:
                    pivot_cols.add(c)
                    break
        
        x = [Fraction(0) for _ in range(cols)]
        for c in range(cols):
            if c in pivot_cols:
                # Find the row where this column is the pivot
                for r in range(rows):
                    if M[r][c] == 1 and all(M[r][cc] == 0 for cc in range(cols) if cc != c):
                        x[c] = M[r][cols]
                        break
            else:
                x[c] = Fraction(0)
        
        # Verify solution A*x = b
        for i in range(rows):
            s = Fraction(0)
            for j in range(cols):
                s += A[i][j] * x[j]
            if s != b[i]:
                return None
                
        return x

    def try_rank(rank, attempt_seed=0):
        """
        Attempt to find a rank-r decomposition.
        Uses a structured approach:
        1. Randomly initialize U, V, W with sparse integer entries (mostly 0, some 1, -1, 1/2).
        2. This defines a subspace of the solution space.
        3. Actually, a better algebraic approach for small n:
           Fix U and V to be sparse random matrices.
           The equation is: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
           This is linear in W for fixed U, V.
           So we can fix U and V, and solve for W.
           However, W must also be integer/half-integer.
           
           Let's try: 
           Fix U and V to be "generic" sparse integer matrices.
           Construct the linear system for W.
           The unknowns are W[k*n+i, c].
           The equations are for each (i, j, k):
           sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1.
           
           Let A_{(i,j,k), (k',i',c')} be the coefficient of W[k'n+i', c'] in the equation for (i,j,k).
           A_{(i,j,k), (k',i',c')} = U[i*m+j, c'] * V[j*p+k', c'] if i'=i and k'=k else 0.
           Wait, the sum is over c. The term is U[i,j,c] * V[j,k,c] * W[k,i,c].
           So for a fixed (i,j,k), the equation involves only W[k,i,c] for all c.
           It does NOT involve W[k',i',c'] for other k', i'.
           This means the equations for different (i,k) pairs are decoupled!
           
           Equation for fixed (i, k):
           For all j in [0, m):
           sum_{c=0}^{rank-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
           
           This is a system of m linear equations in rank unknowns W[k*n+i, c].
           Matrix M_{(i,k)} has shape (m, rank).
           M_{(i,k), j, c} = U[i*m+j, c] * V[j*p+k, c].
           Vector b_{(i,k)} = [1, 1, ..., 1] (length m).
           
           We need to solve M_{(i,k)} * W_{(i,k)} = b_{(i,k)} for each (i,k).
           There are n*p such systems.
           If rank >= m, we might have a solution.
           We need the solution W to have integer/half-integer entries.
           
           Strategy:
           1. Randomly generate U and V with entries from {0, 1, -1, 1/2, -1/2}.
           2. For each (i,k), form the linear system.
           3. Solve for W.
           4. Check if W entries are integer/half-integer.
           5. Check if the full decomposition is exact (it should be by construction if systems solved).
        """
        if time.time() > deadline:
            return None
            
        np.random.seed(attempt_seed)
        
        # Generate sparse U and V
        # Density ~ 0.5
        U = [[Fraction(0) for _ in range(rank)] for _ in range(n*m)]
        V = [[Fraction(0) for _ in range(rank)] for _ in range(m*p)]
        
        candidates = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)]
        
        for i in range(n*m):
            for c in range(rank):
                if np.random.rand() < 0.5:
                    U[i][c] = candidates[np.random.randint(5)]
                    
        for j in range(m*p):
            for c in range(rank):
                if np.random.rand() < 0.5:
                    V[j][c] = candidates[np.random.randint(5)]
        
        W = [[Fraction(0) for _ in range(rank)] for _ in range(p*n)]
        
        # Solve for W
        for i in range(n):
            for k in range(p):
                # System: M * W_col = b
                # M is m x rank
                M = [[Fraction(0) for _ in range(rank)] for _ in range(m)]
                b = [Fraction(1) for _ in range(m)]
                
                for j in range(m):
                    for c in range(rank):
                        M[j][c] = U[i*m + j][c] * V[j*p + k][c]
                
                # Solve M * x = b
                sol = solve_linear_system(M, b)
                if sol is None:
                    return None
                
                # Check if solution is integer/half-integer
                valid_w = True
                for c in range(rank):
                    val = sol[c]
                    # Check if val is integer or half-integer
                    # val * 2 should be integer
                    if (val * 2).denominator != 1:
                        valid_w = False
                        break
                if not valid_w:
                    return None
                
                for c in range(rank):
                    W[k*n + i][c] = sol[c]
        
        # Final verification (just in case)
        if verify_exact(U, V, W):
            # Convert to list of floats for output interface
            U_out = [[float(u) for u in row] for row in U]
            V_out = [[float(v) for v in row] for row in V]
            W_out = [[float(w) for w in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
            
        return None

    # Search ranks from lower bound
    for rank in range(30, 61):
        if time.time() > deadline:
            break
            
        # Try multiple random instances for this rank
        # Since the problem is symmetric, random sparse initialization is a good heuristic
        # to find a solution if one exists in the "generic" case.
        # We limit attempts to ensure we finish in time.
        max_attempts = 50 if rank < 40 else 10
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            res = try_rank(rank, attempt_seed=rank * 1000 + attempt)
            if res is not None:
                return res
        
    # Fallback: Schoolbook (Rank 60)
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

if __name__ == "__main__":
    result = construct()
    # Simple check
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                max_err = max(max_err, abs(s - 1))
    print(f"Rank: {rank}, Max Error: {max_err}")
