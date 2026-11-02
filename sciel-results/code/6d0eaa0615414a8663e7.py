"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]
import numpy as np
from itertools import product
import time

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the target tensor: T[i,j,k] = 1 for all i,j,k
    # The decomposition should satisfy: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Check each (i, k) pair in the output
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                for c in range(rank):
                    if U[i*m+j][c] != 0 and V[j*p+k][c] != 0 and W[k*n+i][c] != 0:
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            if total != 1:
                return False
    return True

def _find_supports_gf2(n, m, p, target_rank, max_tries=100):
    """
    Find valid sparse support patterns over GF(2) for rank-1 terms.
    Returns a list of (U_support, V_support, W_support) tuples where each support is a set of indices.
    """
    # For each rank-1 term c, we need to choose:
    # - A subset of U indices (rows i*m+j)
    # - A subset of V indices (rows j*p+k)
    # - A subset of W indices (rows k*n+i)
    # such that the tensor product covers all (i,j,k) triples exactly once (mod 2)
    
    # This is a complex constraint. Let's use a simpler approach:
    # Try random sparse patterns and check if they work over GF(2)
    
    supports = []
    np.random.seed(42)
    
    for attempt in range(max_tries):
        # Generate random sparse supports
        candidate_supports = []
        valid = True
        
        for c in range(target_rank):
            # Randomly select which U, V, W entries are nonzero
            U_mask = np.random.randint(0, 2, n*m)
            V_mask = np.random.randint(0, 2, m*p)
            W_mask = np.random.randint(0, 2, p*n)
            
            # Check if this candidate set of supports covers all (i,j,k) exactly once mod 2
            # Build the coverage tensor
            coverage = np.zeros((n, m, p), dtype=int)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Count how many times (i,j,k) is covered
                        count = 0
                        for cc in range(target_rank):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            if cc < len(candidate_supports):
                                U_s, V_s, W_s = candidate_supports[cc]
                                if u_idx in U_s and v_idx in V_s and w_idx in W_s:
                                    count += 1
                        coverage[i, j, k] = count % 2
            
            # Check if coverage is all ones
            if not np.all(coverage == 1):
                valid = False
                break
        
        if valid:
            supports = candidate_supports
            break
    
    return supports

def _solve_for_coefficients(supports, n, m, p):
    """
    Given support patterns, solve for the exact rational coefficients.
    Returns (U, V, W) or None if no solution exists.
    """
    target_rank = len(supports)
    
    # Build the linear system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of n*m*p equations in rank*3 variables (but nonlinear due to products)
    
    # For a rank-1 term, each (U[i*m+j,c], V[j*p+k,c], W[k*n+i,c]) triple must be 0 or 1
    # if we want integer/half-integer solutions with the support fixed.
    
    # Actually, with fixed supports, the coefficients are determined:
    # For each c, if (i,j,k) is in the support of term c, then U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must contribute to the sum.
    
    # Let's use a different approach: assume all nonzero entries in the support are 1.
    # Then check if the decomposition works.
    
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    for c, (U_s, V_s, W_s) in enumerate(supports):
        for idx in U_s:
            U[idx][c] = 1
        for idx in V_s:
            V[idx][c] = 1
        for idx in W_s:
            W[idx][c] = 1
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    return None

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition with rank = n*m*p."""
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
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    
    Tries to find a lower-rank decomposition using CSP over finite fields,
    falling back to the schoolbook algorithm (rank = 60) if no improvement is found.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Start with schoolbook
    best_result = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Try ranks from 20 up to best_rank - 1
    for target_rank in range(max(n*m, m*p, p*n), best_rank):
        if time.time() > deadline - 30:  # Keep 30s safety margin
            break
        
        # Try to find supports for this rank
        supports = _find_supports_gf2(n, m, p, target_rank, max_tries=50)
        
        if supports:
            result = _solve_for_coefficients(supports, n, m, p)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Found a valid decomposition, use it
    
    return best_result
# EVOLVE-BLOCK-END
