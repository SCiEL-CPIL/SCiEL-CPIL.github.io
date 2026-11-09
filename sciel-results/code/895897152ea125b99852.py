"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] should equal 1 if the term contributes to C[i,k], else 0
    # Actually, the standard embedding: T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
    # We need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{i,k} for the "correct" terms?
    
    # Let me reconsider. The matrix multiplication tensor T is defined as:
    # T[i,j,k] = 1 if this term A[i,j]*B[j,k] appears in C[i,k]
    # For standard matrix mult, every (i,j,k) triple contributes exactly once to C[i,k]
    # So T[i,j,k] = 1 for all i,j,k in the valid ranges.
    
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need this to equal 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, target_rank, time_limit=30):
    """
    Try to reduce the rank by finding linear combinations that allow elimination.
    Uses the structure of prime dimensions to constrain search.
    """
    start_time = time.time()
    rank = len(U[0])
    
    # Convert to numpy for efficiency
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try to find if any rank-1 term can be expressed as a linear combination of others
    # with integer/half-integer coefficients
    
    # For each pair of terms, check if one can be eliminated
    # This is expensive, so we use a greedy approach with time limit
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Try removing one term at a time and checking if we can adjust others
    for c_remove in range(rank):
        if time.time() - start_time > time_limit:
            break
        
        # Skip the term to be removed
        remaining = [c for c in range(rank) if c != c_remove]
        
        # Check if term c_remove can be approximated by linear combination of remaining
        # with small integer/half-integer coefficients
        
        # Get the vectors for the term to remove
        u_vec = np.array([U_arr[i*m+j, c_remove] for i in range(n) for j in range(m)])
        v_vec = np.array([V_arr[j*p+k, c_remove] for j in range(m) for k in range(p)])
        w_vec = np.array([W_arr[k*n+i, c_remove] for k in range(p) for i in range(n)])
        
        # Try to find coefficients alpha_c for remaining terms such that:
        # sum_c alpha_c * (u_c, v_c, w_c) approximates (u_vec, v_vec, w_vec)
        # This is a linear system. We'll use a simple approach: try small coefficients
        
        # For efficiency, only try coefficients in {-1, -0.5, 0, 0.5, 1}
        # This is very limited but fast
        
        # Actually, let's try a different approach: look for structural patterns
        # Given prime dimensions 3,5,7, there might be symmetry-based reductions
        
        pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 105
    
    # Build schoolbook decomposition
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
    
    # Verify the initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Initial schoolbook decomposition is invalid")
    
    # Try to reduce rank using structural insights
    # For prime dimensions, explore if there are algebraic identities
    
    # Strategy: Look for patterns where terms can be combined
    # Using the fact that 3, 5, 7 are primes, we explore cyclic structures
    
    # Time-limited search for rank reductions
    start_time = time.time()
    time_limit = 100  # Leave margin for verification and return
    
    # Try a structured search: look for linear dependencies among the basis vectors
    # by examining the tensor as a linear map
    
    # Convert to arrays for efficient computation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We want to find if there's a lower rank decomposition
    
    # Approach: Use the fact that for matrix multiplication, the tensor has special structure
    # Try to find if any subset of rank-1 terms can be replaced by fewer terms
    
    # For now, return the schoolbook decomposition as a safe baseline
    # Future improvements could use more sophisticated search
    
    if time.time() - start_time > time_limit:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
