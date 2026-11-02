"""Matrix multiplication 3x3 * 3x7 tensor rank decomposition."""
import time
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 7]

def _verify_exact(U, V, W, n, m, p):
    """Verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != Fraction(1):
                    return False
    return True

def _try_construct_rank(r, n, m, p, deadline, seed=42):
    """Try to construct a rank-r decomposition using structured random search."""
    rng = random.Random(seed)
    
    # Strategy: initialize with a valid high-rank decomposition, then try to
    # merge columns (reduce rank) while maintaining validity.
    # Or: use a parametric family and solve for parameters.
    
    # For small r, try direct construction with a structured pattern.
    # Key idea: use the "outer product" structure.
    
    # Initialize U, V, W with random values from {-1, 0, 1} or {-1/2, 0, 1/2, 1}
    def rand_val():
        choices = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
        return choices[rng.randint(0, len(choices)-1)]
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        U = [[rand_val() for _ in range(r)] for _ in range(n*m)]
        V = [[rand_val() for _ in range(r)] for _ in range(m*p)]
        W = [[rand_val() for _ in range(r)] for _ in range(p*n)]
        
        if _verify_exact(U, V, W, n, m, p):
            return (U, V, W)
    
    return None

def _try_linear_solve_rank(r, n, m, p, deadline, seed=42):
    """
    For fixed U and V, the equations in W are linear:
    For each (i,j,k): sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1
    
    This gives p*n equations in r unknowns per row of W (actually coupled).
    
    Better: fix U and V, then for each row index (k*n+i) of W, we have m equations:
    For fixed (k,i): for each j in range(m): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
    
    So for each (k,i), we have m linear equations in r unknowns (the row W[k*n+i,:]).
    If r >= m, we can solve this (underdetermined or determined).
    """
    import numpy as np
    
    rng = random.Random(seed)
    
    for attempt in range(20):
        if time.time() > deadline:
            return None
        
        # Random U and V with values in {-1, 0, 1}
        U = [[Fraction(rng.choice([-1, 0, 1])) for _ in range(r)] for _ in range(n*m)]
        V = [[Fraction(rng.choice([-1, 0, 1])) for _ in range(r)] for _ in range(m*p)]
        
        # For each (k, i), solve for W[k*n+i, :]
        W = [[Fraction(0) for _ in range(r)] for _ in range(p*n)]
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build system: for each j, sum_c W[k*n+i,c] * A_c = 1
                # where A_c = U[i*m+j,c] * V[j*p+k,c]
                # m equations, r unknowns
                # Use least squares or find a particular solution
                # Since we need exact integer/half-integer, use rational solving
                
                # Build m x r matrix A where A[j,c] = U[i*m+j,c] * V[j*p+k,c]
                # Solve A * w = ones(m)
                # If r >= m, use numpy to find a solution, then round to nearest half-integer
                
                A = np.array([[float(U[i*m+j][c] * V[j*p+k][c]) for c in range(r)] for j in range(m)])
                b = np.ones(m)
                
                try:
                    # Find minimum-norm solution
                    w = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    valid = False
                    break
                
                # Check if solution exists (residual should be ~0)
                residual = np.linalg.norm(A @ w - b)
                if residual > 1e-10:
                    valid = False
                    break
                
                # Round to nearest half-integer and verify
                w_rounded = [Fraction(round(float(w[c]) * 2) / 2) for c in range(r)]
                
                # Verify this rounded solution works exactly
                for j in range(m):
                    s = Fraction(0)
                    for c in range(r):
                        s += w_rounded[c] * U[i*m+j][c] * V[j*p+k][c]
                    if s != Fraction(1):
                        valid = False
                        break
                if not valid:
                    break
                
                W[k*n+i] = w_rounded
            
            if not valid:
                break
        
        if valid:
            # Full verification
            if _verify_exact(U, V, W, n, m, p):
                return (U, V, W)
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return (U, V, W)

def _try_block_structure(n, m, p, deadline):
    """
    Try to exploit block structure. Split the 3x7 into blocks and use
    a tensor product / Kronecker-like decomposition.
    
    Idea: 3x3 * 3x7 = 3x3 * (3x2 + 3x1) roughly, but that's not quite right.
    
    Alternative: Use the fact that we can share the "3" dimension.
    Write C = A * B where A is 3x3, B is 3x7.
    
    For each column block of B (say 2 columns at a time), we can use
    a 3x3 * 3x2 multiplication which has rank at most 3*3*2 = 18,
    but potentially less.
    
    7 = 2+2+2+1, so 4 blocks: three 3x2 and one 3x1.
    3x2 mult: rank <= 18, 3x1 mult: rank <= 9.
    Total: 3*18 + 9 = 63. Same as schoolbook.
    
    But if we can share computation across blocks, we might do better.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit (150s kill)
    
    # Lower bound on rank
    rank_lb = max(n * m, m * p, n * p)  # max(9, 21, 21) = 21
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Focus on ranks from 21 upward
    for r in range(rank_lb, best_rank):
        if time.time() > deadline:
            break
        
        # Try linear solve approach
        result = _try_linear_solve_rank(r, n, m, p, deadline, seed=r)
        if result is not None:
            best_result = result
            best_rank = r
            break
        
        # Try random search
        if time.time() < deadline - 10:
            result = _try_construct_rank(r, n, m, p, deadline, seed=r + 1000)
            if result is not None:
                best_result = result
                best_rank = r
                break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
