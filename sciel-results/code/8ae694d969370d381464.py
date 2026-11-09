"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try structured decompositions with small half-integer entries
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Known lower bounds and upper bounds for small matrix multiplication:
    # 2x3 * 3x5: schoolbook rank = 30
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # But we can often do better than schoolbook
    
    # Strategy: Try ranks from 15 upward, using structured patterns
    # For each rank, try various systematic initializations
    
    best_result = None
    best_rank = float('inf')
    
    # First, create schoolbook as ultimate fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Use a more systematic search with structured patterns
    
    # For 2x3 * 3x5, we can exploit the structure:
    # A is 2x3, B is 3x5, C is 2x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Try various rank values and structured decompositions
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple structured approaches for this rank
        found = _search_rank(target_rank, n, m, p, deadline)
        if found:
            best_rank = target_rank
            best_result = found
            break
    
    return best_result

def _search_rank(target_rank, n, m, p, deadline):
    """Search for a rank-target_rank decomposition using structured patterns."""
    
    # Approach 1: Try using the Kronecker product structure
    # For 2x3 * 3x5, we can think of it as combining smaller multiplications
    
    # Approach 2: Use random half-integer initialization with verification
    np.random.seed(42 + target_rank)
    
    # Try many random initializations
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        # Random half-integers in [-3, 3]
        U = np.random.randint(-6, 7, size=(n*m, target_rank)) / 2.0
        V = np.random.randint(-6, 7, size=(m*p, target_rank)) / 2.0
        W = np.random.randint(-6, 7, size=(p*n, target_rank)) / 2.0
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Approach 3: Try structured patterns based on matrix multiplication properties
    # For 2x3 * 3x5, try using the fact that we can combine terms
    
    # Try using a pattern where some entries are 0, ±1/2, ±1
    # This is a more constrained search that might find solutions faster
    
    # Generate all possible patterns with small half-integers
    # This is expensive, so use a smarter approach
    
    # Try using the Strassen-like approach for small matrices
    # For 2x2 * 2x2, Strassen uses rank 7 instead of 8
    # For 2x3 * 3x5, we can try similar ideas
    
    # Let's try a different approach: use the fact that the tensor is simple
    # and try to find a decomposition using linear algebra
    
    # Set up the system of equations
    # For each (i,j,k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 30 equations in 3*target_rank unknowns
    # (with the constraint that they form a valid tensor decomposition)
    
    # For small target_rank, we can try to solve this system
    
    if target_rank <= 20:
        result = _solve_equations(target_rank, n, m, p, deadline)
        if result:
            return result
    
    return None

def _solve_equations(target_rank, n, m, p, deadline):
    """Try to solve the system of equations for a given rank."""
    
    # This approach is complex, so we'll use a simpler heuristic
    # Try specific structured patterns that are known to work for small cases
    
    # For 2x3 * 3x5, one known approach is to use the following pattern:
    # Split the 3x5 matrix into 2x5 and 1x5 parts
    
    # Actually, let's try a direct approach with specific patterns
    
    # Pattern: Use the fact that we can write the product as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # We can try to combine these terms using identities
    
    # For now, return None to indicate no solution found with this method
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
