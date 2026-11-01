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

def _solve_exact_linear_system(A, b):
    """
    Solve A x = b exactly using Gaussian elimination with Fractions.
    A is a list of lists of Fractions.
    b is a list of Fractions.
    Returns a list of Fractions x if a solution exists, else None.
    If underdetermined, returns one specific solution (setting free variables to 0).
    """
    n_rows = len(A)
    n_cols = len(A[0])
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    rank = 0
    for col in range(n_cols):
        if rank == n_rows:
            break
        # Find pivot
        pivot_row = -1
        for r in range(rank, n_rows):
            if M[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
            
        # Swap rows
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        
        # Normalize pivot row
        pivot_val = M[rank][col]
        for c in range(col, n_cols + 1):
            M[rank][c] /= pivot_val
            
        # Eliminate
        for r in range(n_rows):
            if r == rank:
                continue
            factor = M[r][col]
            if factor != 0:
                for c in range(col, n_cols + 1):
                    M[r][c] -= factor * M[rank][c]
        rank += 1
        
    # Check for consistency
    for r in range(rank, n_rows):
        if M[r][n_cols] != 0:
            return None
            
    # Construct solution
    x = [Fraction(0)] * n_cols
    # Back substitution for reduced row echelon form
    # Since we did full elimination, M is in RREF
    for r in range(rank):
        # Find the leading column
        lead_col = -1
        for c in range(n_cols):
            if M[r][c] != 0:
                lead_col = c
                break
        if lead_col != -1:
            x[lead_col] = M[r][n_cols]
            
    return x

def _try_structured_decomposition(n, m, p, time_budget, seed=42):
    """
    Try structured decompositions using exact linear algebra.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    best = None
    best_rank = n * m * p + 1
    
    # Try ranks from 30 to n*m*p-1
    # We start low to find minimal rank
    for r in range(30, n*m*p):
        if time.time() > deadline - 2:
            break
        if r >= best_rank:
            continue
            
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
                
            # Initialize U and V with small integers
            U = np.zeros((n*m, r), dtype=int)
            V = np.zeros((m*p, r), dtype=int)
            
            # Use sparse random integers
            for i in range(n*m):
                for c in range(r):
                    if rng.random() < 0.5:
                        U[i, c] = rng.choice([-1, 1])
                        
            for j in range(m*p):
                for c in range(r):
                    if rng.random() < 0.5:
                        V[j, c] = rng.choice([-1, 1])
                        
            # Check consistency for each (i, k)
            feasible = True
            W = np.zeros((p*n, r))
            
            for i in range(n):
                for k in range(p):
                    # Build system A x = 1
                    # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A_float = np.zeros((m, r))
                    for j in range(m):
                        A_float[j, :] = U[i*m+j] * V[j*p+k]
                        
                    # Quick check with numpy
                    # Rank of A must be equal to rank of [A | 1]
                    # Since b is all ones, we can check if 1 is in col(A)
                    # Actually, we just need to check if the system is consistent.
                    # Using lstsq to check residual
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A_float, np.ones(m), rcond=None)
                        if len(residuals) > 0 and residuals[0] > 1e-9:
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
                    
                    if not feasible:
                        break
                    
                    # Solve exactly
                    A_frac = [[Fraction(int(U[i*m+j, c]) * int(V[j*p+k, c])) for c in range(r)] for j in range(m)]
                    b_frac = [Fraction(1)] * m
                    x_frac = _solve_exact_linear_system(A_frac, b_frac)
                    if x_frac is None:
                        feasible = False
                        break
                        
                    # Store solution in W
                    for c in range(r):
                        W[k*n+i, c] = float(x_frac[c])
                        
                if not feasible:
                    break
                    
            if feasible:
                # Convert to lists for verification
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify with Fractions to be sure
                if _verify(U_list, V_list, W_list, n, m, p):
                    # Round to half-integers if possible?
                    # The task allows integer/half-integer.
                    # Our solution is rational. We can keep it as float if it's exact enough, 
                    # but the contract says "integer/half-integer entries".
                    # We should check if the fractions are half-integers.
                    # If not, we might need to scale or reject.
                    # However, the verification `_verify` uses Fraction(U[i*m+j][c])...
                    # If U, V, W are floats, Fraction(float) is inexact.
                    # We should store them as Fractions or ensure they are exact floats.
                    
                    # Let's re-verify with the exact fractions we found
                    # We need to store the exact fractions.
                    # The current W is float. Let's re-compute W as Fractions.
                    
                    W_frac = [[Fraction(0) for _ in range(r)] for _ in range(p*n)]
                    for i in range(n):
                        for k in range(p):
                            A_frac = [[Fraction(int(U[i*m+j, c]) * int(V[j*p+k, c])) for c in range(r)] for j in range(m)]
                            b_frac = [Fraction(1)] * m
                            x_frac = _solve_exact_linear_system(A_frac, b_frac)
                            if x_frac:
                                for c in range(r):
                                    W_frac[k*n+i][c] = x_frac[c]
                                    
                    # Check if entries are half-integers
                    def is_half_int(frac):
                        return frac * 2 == int(frac * 2)
                    
                    valid_half_int = True
                    for row in W_frac:
                        for val in row:
                            if not is_half_int(val):
                                valid_half_int = False
                                break
                        if not valid_half_int:
                            break
                            
                    if valid_half_int:
                        # Convert to floats for the return value? 
                        # The contract says "integer/half-integer entries".
                        # Usually this means the values should be exactly representable.
                        # We can return them as floats if they are exact, or as Fractions?
                        # The seed program returns lists of numbers.
                        # If we return Fractions, does the evaluator handle it?
                        # The seed program uses `Fraction` in `_verify` on the input lists.
                        # If we return Fractions, `Fraction(Fraction)` works.
                        # If we return floats, `Fraction(float)` might be inexact.
                        # Let's return floats that are exactly half-integers.
                        W_list = [[float(val) for val in row] for row in W_frac]
                        
                        # Final check
                        if _verify(U_list, V_list, W_list, n, m, p):
                            best = {"U": U_list, "V": V_list, "W": W_list}
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
    
    # Try structured decomposition approach with multiple seeds
    seeds = [12345, 67890, 11111, 22222, 33333]
    for seed in seeds:
        time_left = deadline - time.time()
        if time_left < 10:
            break
        result = _try_structured_decomposition(n, m, p, time_left, seed=seed)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
                    break
    
    return best
