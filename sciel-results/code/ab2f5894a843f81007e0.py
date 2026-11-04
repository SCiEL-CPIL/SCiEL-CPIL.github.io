"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
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
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_rank_reduction(n, m, p, max_rank_to_try, time_limit):
    """Try to find a rank-reduced decomposition by merging schoolbook terms."""
    U0, V0, W0 = _schoolbook(n, m, p)
    rank0 = n * m * p
    start_time = time.time()
    
    # For each pair of schoolbook terms, check if they can be merged
    # Two terms (i,j,k) and (i',j',k') can potentially be merged if there's
    # a linear relation that allows representing both with one term plus adjustments
    
    # Actually, a more practical approach: try to find if the tensor has
    # a decomposition with rank < rank0 by solving a system of equations
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k
    # We want to find U, V, W with rank r such that:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns
    # For r < n*m*p, this is underdetermined but we need exact solutions
    
    # Try a greedy approach: start with schoolbook, try to eliminate one term at a time
    # by seeing if it can be expressed as a linear combination of others
    
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    # Convert to numpy for efficiency
    U_np = np.array(U0, dtype=float)
    V_np = np.array(V0, dtype=float)
    W_np = np.array(W0, dtype=float)
    
    # Try to find linear dependencies among the rank-1 tensors
    # Each schoolbook term c corresponds to a rank-1 tensor:
    # T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For schoolbook, T_c is 1 only at position (i_c, j_c, k_c) and 0 elsewhere
    # So the tensors are linearly independent and we can't merge any two directly
    
    # However, we can try a different ansatz: use a structured decomposition
    # that exploits the matrix multiplication structure
    
    # For 3x4x6, try a block-based approach:
    # Split the 3x4 matrix into blocks and the 4x6 into compatible blocks
    
    # Alternative: use known lower bounds and try to find if rank < 72 is possible
    # For 3x4x6, the tensor rank is at least max(3*4, 4*6, 3*6) = 24
    # and at most 3*4*6 = 72
    
    # Try to find a decomposition with rank between 24 and 71
    
    # Use a randomized approach: generate random U, V, W and try to solve
    # for the remaining degrees of freedom
    
    for target_rank in range(min(71, max_rank_to_try), 23, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a rank-target_rank decomposition
        # Use a simple iterative approach: start with random initialization
        # and try to satisfy the constraints
        
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            U_rand = np.random.randn(n * m, target_rank) * 0.1
            V_rand = np.random.randn(m * p, target_rank) * 0.1
            W_rand = np.random.randn(p * n, target_rank) * 0.1
            
            # Check if this random initialization happens to work (unlikely)
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c] 
                                  for c in range(target_rank))
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                break
        
        if best_rank < rank0:
            break
    
    return best_rank, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find a rank-reduced decomposition
    if time.time() < deadline - 30:  # Leave 30s for verification and return
        max_rank_to_try = min(71, best_rank - 1)
        time_left = deadline - time.time() - 5
        if time_left > 10:
            rank_found, U_found, V_found, W_found = _try_rank_reduction(
                n, m, p, max_rank_to_try, time_left
            )
            if rank_found < best_rank:
                # Verify the found decomposition
                if _verify(U_found, V_found, W_found, n, m, p, tol=1e-8):
                    best_rank = rank_found
                    best_result = {"U": U_found, "V": V_found, "W": W_found}
    
    return best_result

# EVOLVE-BLOCK-END
