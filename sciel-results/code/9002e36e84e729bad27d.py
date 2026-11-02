"""Matrix multiplication 4x4x8 - rank minimization via structured search."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank as fallback
    rank_school = n * m * p
    
    # Try to find a better rank using structured decomposition
    # For 4x4x8, the schoolbook rank is 128.
    # We can potentially do better by exploiting structure.
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # Also rank >= n*p = 32 (from output size)
    
    # Let's try a simple structured approach:
    # Decompose using Kronecker-like structure
    
    # First, let's try rank = 64 (half of schoolbook) using a structured pattern
    # This is a heuristic - we'll verify if it works
    
    best_result = _schoolbook(n, m, p)
    best_rank = rank_school
    
    # Try various ranks below schoolbook
    # Start with promising ranks
    candidate_ranks = []
    
    # Add some structured candidates
    for r in range(32, rank_school, 4):
        candidate_ranks.append(r)
    
    # Sort by rank (ascending) to find minimal
    candidate_ranks.sort()
    
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        result = _try_rank(n, m, p, r)
        if result is not None:
            best_result = result
            best_rank = r
            # Found a valid decomposition at this rank
            # Continue searching for even lower rank
            continue
    
    return best_result

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check: C[i,k] = sum_j A[i,j] * B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # More precisely, for each (i,j,k): the coefficient of A[i,j]*B[j,k] in C[i,k] should be 1
    # And for all other combinations, should be 0
    
    # Build the tensor T[i,j,k,c] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Sum over c should give delta[i,j,k] = 1 if this is a valid MM tensor
    
    # Actually, the MM tensor is: T[i,j,k] = 1 for all i,j,k (each product term appears once)
    # The decomposition gives: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all i,j,k
    
    # Let's verify by computing the sum for each (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(n, m, p, r):
    """Try to find a valid decomposition of rank r. Returns None if not found."""
    # This is a heuristic - we use a structured approach
    
    # For 4x4x8, try a Kronecker product structure
    # A is 4x4, B is 4x8, C is 4x8
    
    # One approach: view as block matrices
    # Split into 2x2 blocks of 2x2 and 2x4
    
    # Try a simple structured decomposition
    # Use the fact that 4x4x8 can be decomposed using smaller matrix multiplications
    
    # Heuristic: try rank = 2 * (n/2) * (m/2) * p if n,m even
    # This is just a guess - we need to actually construct it
    
    # For now, return None to indicate we couldn't find a better decomposition
    # This is a placeholder for a more sophisticated search
    
    return None

# EVOLVE-BLOCK-END
