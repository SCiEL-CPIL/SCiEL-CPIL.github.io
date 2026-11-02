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
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _try_random_uv(n, m, p, rank, max_time=120.0):
    """
    Strategy:
    1. Randomly generate U (n*m x rank) and V (m*p x rank) with sparse integer entries.
    2. For each (i, k), form the matrix A_ik (m x rank) where A_ik[j, c] = U[i*m+j, c] * V[j*p+k, c].
    3. Solve A_ik * w = 1 for w (vector of length rank).
    4. If a solution exists for all (i, k), collect w into W.
    5. Check if W entries are close to integers/half-integers. If so, round and verify.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    
    attempts = 0
    
    while time.time() < deadline - 5.0:
        attempts += 1
        
        # Generate random U and V
        # Use sparse matrices with entries {-1, 0, 1}
        # Density can vary
        density = random.uniform(0.1, 0.5)
        
        U_np = np.zeros((n_rows_U, rank))
        V_np = np.zeros((n_rows_V, rank))
        
        for c in range(rank):
            # Random support for U column c
            mask_u = np.random.rand(n_rows_U) < density
            U_np[mask_u, c] = np.random.choice([-1, 1], size=np.sum(mask_u))
            
            # Random support for V column c
            mask_v = np.random.rand(n_rows_V) < density
            V_np[mask_v, c] = np.random.choice([-1, 1], size=np.sum(mask_v))
        
        # Check if the system is solvable for all (i, k)
        W_np = np.zeros((n_rows_W, rank))
        solvable = True
        
        for i in range(n):
            for k in range(p):
                # Construct A_ik: m x rank
                # A_ik[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A_ik = np.zeros((m, rank))
                for j in range(m):
                    A_ik[j, :] = U_np[i*m+j, :] * V_np[j*p+k, :]
                
                # Solve A_ik w = 1
                # We want a solution w.
                # If rank(A_ik) < m, it might be inconsistent.
                # Use lstsq to find minimum norm solution, then check residual.
                try:
                    w, residuals, rank_A, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                    # Check residual
                    if residuals.size > 0:
                        if residuals[0] > 1e-9:
                            solvable = False
                            break
                    else:
                        # Underdetermined, check if 1 is in column space
                        # lstsq handles this, but let's verify
                        if np.linalg.norm(A_ik @ w - np.ones(m)) > 1e-9:
                            solvable = False
                            break
                    W_np[k*n+i, :] = w
                except np.linalg.LinAlgError:
                    solvable = False
                    break
            
            if not solvable:
                break
        
        if not solvable:
            continue
            
        # We have a candidate W. Check if entries are close to integers or half-integers.
        # The problem allows integer/half-integer.
        # Let's check distance to nearest 0.5
        W_half = np.round(W_np * 2) / 2
        error = np.linalg.norm(W_np - W_half)
        
        if error < 1e-6:
            # It's a valid half-integer solution!
            U_exact = [[int(round(U_np[i, c])) for c in range(rank)] for i in range(n_rows_U)]
            V_exact = [[int(round(V_np[i, c])) for c in range(rank)] for i in range(n_rows_V)]
            W_exact = [[float(W_half[i, c]) for c in range(rank)] for i in range(n_rows_W)]
            
            # Final verification
            if _verify(U_exact, V_exact, W_exact, n, m, p):
                return U_exact, V_exact, W_exact, rank
        
        # If not exact, we could try to perturb U, V to make W more integer-like,
        # but for now, just try another random U, V.
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition
    # We know rank 48 works.
    # Let's try ranks from 47 down to, say, 20.
    # But random search might jump to a lower rank directly.
    # Let's try a few specific ranks.
    
    # Known bounds for 3x4x4 are not tight, but 48 is schoolbook.
    # Let's try to find any rank < 48.
    
    # We will try ranks in a random order or descending order.
    # Descending is safer to find a valid one quickly.
    
    # To maximize chances, we can run the random search for the best rank found so far,
    # and if we find one, we can try to go lower.
    
    current_best_rank = best_rank
    
    # Try a few lower ranks
    for r in range(current_best_rank - 1, 19, -1):
        if time.time() > deadline - 20.0:
            break
            
        result = _try_random_uv(n, m, p, r, max_time=deadline - time.time() - 5.0)
        if result:
            U_new, V_new, W_new, rank_new = result
            if rank_new < current_best_rank:
                current_best_rank = rank_new
                best_U, best_V, best_W = U_new, V_new, W_new
                # If we found a lower rank, we can continue trying to go lower
                # or break if we hit a known lower bound.
                # Let's continue to see if we can go lower.
                
    # Verify the best result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = U, V, W
        current_best_rank = n * m * p
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
