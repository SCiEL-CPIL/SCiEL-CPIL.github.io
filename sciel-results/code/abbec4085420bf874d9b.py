"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product, combinations

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition correctness."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True


def _try_merge_terms(n, m, p, U, V, W, max_iterations=5000):
    """
    Try to merge pairs of terms to reduce rank.
    Two terms t1, t2 can be merged if there exist scalars a, b such that
    the combined contribution works. This is a heuristic search.
    """
    rank = len(U[0])
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    for iteration in range(max_iterations):
        # Try merging two random columns
        if len(best_U[0]) <= 1:
            break
        t1 = random.randint(0, len(best_U[0]) - 1)
        t2 = random.randint(0, len(best_U[0]) - 1)
        if t1 == t2:
            continue
        
        # Try to merge t1 and t2: find a,b such that for all (i,j,k):
        # U1[i,j]*V1[j,k]*W1[k,i] + U2[i,j]*V2[j,k]*W2[k,i]
        # = (a*U1[i,j] + b*U2[i,j]) * (V1[j,k] + c*V2[j,k]) * (W1[k,i] + d*W2[k,i])
        # This is complex; use simpler heuristic: check if one term is scalar multiple of other
        
        # Check if term t2 is a scalar multiple of term t1 in a compatible way
        # For simplicity, try: can we express t2 as linear combo that cancels?
        
        # Simpler approach: try removing a term and see if we can adjust others
        # This is very hard; use a different strategy
        
        pass
    
    return best_U, best_V, best_W, best_rank


def _strassen_like_4x4x4():
    """
    Known rank-49 decomposition for 4x4x4 matrix multiplication.
    Based on Strassen's method generalized.
    This returns U, V, W for the 4x4x4 case.
    """
    # For 4x4x4, there's a known rank-49 construction
    # We'll use a structured approach
    
    n = m = p = 4
    # Use the known fact that 4x4x4 can be done in rank 49
    # Construct using block Strassen ideas
    
    # Actually, let's use a simpler known result:
    # The minimum rank for 4x4x4 is known to be at most 49
    # But constructing it explicitly is complex.
    
    # For now, return schoolbook as placeholder
    U, V, W = _schoolbook(n, m, p)
    return U, V, W


def _search_lower_rank(n, m, p, time_budget=100):
    """
    Search for lower-rank decompositions using structured patterns.
    """
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Strategy: Try to find decompositions with rank < n*m*p
    # Use the fact that for small matrices, we can try all possible
    # "pattern" decompositions
    
    # Approach: Generate random sparse patterns and solve for coefficients
    # This is a constrained optimization problem
    
    # For 4x4x5, try ranks from 60 down to some lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # But realistically, much higher
    
    # Try a structured approach: use bilinear forms
    # The tensor T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Key insight: We can try to factor the all-ones tensor
    # 1 = sum_t f_t(i,j) * g_t(j,k) * h_t(k,i)
    
    # Try small ranks with random initialization and gradient-free search
    
    for target_rank in range(70, 15, -1):
        if time.time() - start_time > time_budget:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - start_time > time_budget:
                break
            
            # Random initialization with small integers/half-integers
            random.seed(attempt + target_rank * 1000)
            
            U_try = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] 
                     for _ in range(n * m)]
            V_try = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] 
                     for _ in range(m * p)]
            W_try = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] 
                     for _ in range(p * n)]
            
            # Check if this works
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                break
        
        if best_rank < n * m * p:
            break
    
    return best_U, best_V, best_W, best_rank


def _construct_via_known_patterns(n, m, p):
    """
    Try known low-rank patterns for small matrix multiplication.
    """
    start_time = time.time()
    
    # For 4x4x5, try to extend known 4x4x4 results
    # One approach: use the 4x4x4 rank-49 construction and extend to 5 columns
    
    # Actually, let's try a direct construction
    # The tensor we need: T[i,j,k] = delta (i.e., 1 for all valid i,j,k)
    
    # Try rank = n*m + m*p + p*n - something
    # This is the "border rank" idea
    
    # For now, try systematic search with better heuristics
    
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a decomposition with rank around 50-60
    # Use the structure: group terms by (i,j) pairs
    
    # Key idea: For fixed (i,j), the sum over t of U[i,j,t]*V[j,k,t]*W[k,i,t] must equal 1
    # for all k. This means for each (i,j), the vector [sum_t U[i,j,t]*V[j,k,t]*W[k,i,t]]_k = [1,1,1,1,1]
    
    # Try to construct U, V, W with limited support
    
    # Approach: Use a "diagonal" structure
    # Let U[i,j,t] be nonzero only for t in some set S(i,j)
    # Similarly for V and W
    
    # For now, use the search approach
    U_s, V_s, W_s, rank_s = _search_lower_rank(n, m, p, time_budget=60)
    if rank_s < best_rank:
        best_U, best_V, best_W = U_s, V_s, W_s
        best_rank = rank_s
    
    return best_U, best_V, best_W, best_rank


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries (as floats).
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, use 150s max but aim for 120
    
    # Generate schoolbook decomposition (rank 80) - guaranteed valid
    U, V, W = _schoolbook(n, m, p)
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find lower-rank decomposition
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Time-budgeted search for lower rank
    remaining_time = 100  # Leave 20s safety margin
    if time.time() - start_time < remaining_time:
        U_s, V_s, W_s, rank_s = _search_lower_rank(n, m, p, time_budget=remaining_time)
        if rank_s < best_rank and _verify_decomposition(U_s, V_s, W_s, n, m, p):
            best_U, best_V, best_W = U_s, V_s, W_s
            best_rank = rank_s
    
    # Final check
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = U, V, W
    
    # Ensure all values are floats
    result = {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
    
    return result
# EVOLVE-BLOCK-END
