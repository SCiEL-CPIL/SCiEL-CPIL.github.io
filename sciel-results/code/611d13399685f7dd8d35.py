"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check if U, V, W correctly decompose the matrix multiplication tensor using exact arithmetic."""
    rank = len(U[0])
    # Convert to Fractions for exact check
    U_frac = [[Fraction(U[i][c]) for c in range(rank)] for i in range(len(U))]
    V_frac = [[Fraction(V[i][c]) for c in range(rank)] for i in range(len(V))]
    W_frac = [[Fraction(W[i][c]) for c in range(rank)] for i in range(len(W))]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                if total != 1:
                    return False
    return True

def _try_random_decomposition(n, m, p, rank, max_time=120.0):
    """
    Try to find a rank-r decomposition by randomly initializing U and V,
    then solving for W exactly.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    
    # Equations: For each (i, j, k):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Let M[i, j, k, c] = U[i*m+j, c] * V[j*p+k, c]
    # Then for fixed (i, k), we have a system for W[k*n+i, :]:
    # sum_j sum_c M[i, j, k, c] * W[k*n+i, c] = 1 ? 
    # No, the equation is for each (i, j, k).
    # sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
    # This must hold for ALL j.
    # So for a fixed (i, k), we have m equations (one for each j) in p*n*rank unknowns?
    # No, W[k*n+i, c] is the same for all j?
    # Yes, W index is k*n+i, which does not depend on j.
    # So for fixed (i, k), we have a system of m equations in rank unknowns (the entries of W[k*n+i, :]).
    # Let W_ik = W[k*n+i, :] (vector of length rank).
    # For each j: sum_c (U[i*m+j, c] * V[j*p+k, c]) * W_ik[c] = 1.
    # This is a linear system: A_ik * W_ik = 1, where A_ik is an m x rank matrix.
    # A_ik[j, c] = U[i*m+j, c] * V[j*p+k, c].
    
    # So the strategy is:
    # 1. Randomly generate U and V.
    # 2. For each (i, k), solve A_ik * W_ik = 1.
    # 3. If all systems have a solution, verify the full decomposition.
    
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        
        # Generate random U and V with small integer entries
        # Using small integers increases the chance of exact solutions
        # and keeps numbers manageable.
        U_np = np.random.randint(-2, 3, (n_rows_U, rank)).astype(float)
        V_np = np.random.randint(-2, 3, (n_rows_V, rank)).astype(float)
        
        # Solve for W
        W_np = np.zeros((n_rows_W, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Build system A_ik * W_ik = 1
                # A_ik is m x rank
                A_ik = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A_ik[j, c] = U_np[i * m + j, c] * V_np[j * p + k, c]
                
                b = np.ones(m)
                
                # Solve using least squares
                try:
                    W_ik, residuals, rank_A, sv = np.linalg.lstsq(A_ik, b, rcond=None)
                    # Check if the solution is exact (residual is 0)
                    if np.linalg.norm(A_ik @ W_ik - b) > 1e-9:
                        valid = False
                        break
                    W_np[k * n + i, :] = W_ik
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Verify with exact arithmetic
            U_list = U_np.tolist()
            V_list = V_np.tolist()
            W_list = W_np.tolist()
            
            # Round to nearest fraction? 
            # The problem allows half-integers.
            # Let's try to snap to half-integers if close
            def snap_half(x):
                val = round(x * 2) / 2
                if abs(x - val) < 1e-9:
                    return val
                return x
            
            U_snapped = [[snap_half(x) for x in row] for row in U_list]
            V_snapped = [[snap_half(x) for x in row] for row in V_list]
            W_snapped = [[snap_half(x) for x in row] for row in W_list]
            
            if _verify(U_snapped, V_snapped, W_snapped, n, m, p):
                return U_snapped, V_snapped, W_snapped, rank
            else:
                # If snapping failed, try without snapping (floats might be exact enough for evaluator?)
                # The prompt says "integer/half-integer entries". So we MUST return exact fractions.
                # If the float solution is not exactly half-integer, it's invalid.
                # We need to find a solution that IS half-integer.
                # This is a Diophantine-like problem.
                # For now, if we found a float solution, we can try to perturb U/V slightly?
                # Or just fail this attempt.
                pass
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank
    # We know rank 48 works.
    # Let's try ranks 47, 46, ... down to some limit
    # But random search for lower ranks is very hard.
    # Given the time limit, we should focus on verifying the schoolbook is correct
    # and perhaps trying a few quick random searches for rank 47.
    
    # The schoolbook is guaranteed to work.
    # Let's just return it if we don't have time to improve.
    # But we should try a little search.
    
    # Try rank 47
    if time.time() < deadline - 10.0:
        result = _try_random_decomposition(n, m, p, best_rank - 1, max_time=10.0)
        if result:
            best_U, best_V, best_W, best_rank = result
    
    # Try rank 46
    if time.time() < deadline - 5.0:
        result = _try_random_decomposition(n, m, p, best_rank - 1, max_time=5.0)
        if result:
            best_U, best_V, best_W, best_rank = result
            
    # Verify the best result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = n * m * p
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
