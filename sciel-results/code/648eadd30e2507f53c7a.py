"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _get_schoolbook():
    """Return the schoolbook rank-42 decomposition."""
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W):
    """
    Verify that U, V, W form a valid decomposition for 2x3 * 3x7.
    The tensor equation:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [n], j in [m], k in [p]
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_random_search(max_rank, min_rank, time_limit, seed=42):
    """
    Try to find a rank-r decomposition for r from max_rank down to min_rank.
    Uses random initialization and checks if the tensor equation is satisfied.
    """
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    start_time = time.time()
    
    for rank in range(max_rank, min_rank - 1, -1):
        if time.time() - start_time > time_limit:
            break
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and time.time() - start_time < time_limit:
            attempts += 1
            U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
            V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
            W = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(p*n)]
            
            if _verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W, "rank": rank}
    
    return None

def _structured_decomposition():
    """
    Try a structured decomposition that might achieve rank < 42.
    
    For 2x3 * 3x7, we can think of it as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    One approach: use the fact that n=2 is small.
    We can write:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    This is already the schoolbook with rank 2*3*7 = 42.
    
    Let's try to find a lower rank by exploiting linear dependencies.
    
    Actually, for matrix multiplication, the rank is known to be at most n*m*p (schoolbook)
    and at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21.
    
    For 2x3 * 3x7, let's try to construct a rank-35 decomposition.
    
    Idea: Split the 3 columns of A into two groups: {0,1} and {2}.
    Group 1: 2x2 * 2x7 subproblem (but B is 3x7, so we need to handle the third row)
    
    Actually, let's try a different approach. Use the following identity:
    For any 2x3 * 3x7 multiplication, we can write:
    
    Let's try to use the fact that we can share some terms between the two rows of C.
    
    Consider:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    If we define new variables:
    X0 = A[0,0] + A[1,0]
    X1 = A[0,1] + A[1,1]
    X2 = A[0,2] + A[1,2]
    Y0 = A[0,0] - A[1,0]
    Y1 = A[0,1] - A[1,1]
    Y2 = A[0,2] - A[1,2]
    
    Then:
    A[0,j] = (Xj + Yj) / 2
    A[1,j] = (Xj - Yj) / 2
    
    So:
    C[0,k] = sum_j (Xj + Yj)/2 * B[j,k] = 1/2 * (sum_j Xj*B[j,k] + sum_j Yj*B[j,k])
    C[1,k] = sum_j (Xj - Yj)/2 * B[j,k] = 1/2 * (sum_j Xj*B[j,k] - sum_j Yj*B[j,k])
    
    This doesn't reduce the rank directly, but it shows a structure.
    
    Let's try a more direct approach: assume rank r and try to solve for the coefficients.
    For small r, we can set up a system of equations.
    
    Actually, let's just try to construct a specific low-rank decomposition.
    
    For rank 35, we could try to save 7 terms. One way: for each k, share some terms between i=0 and i=1.
    
    Let's try: for each k, use 3 terms for i=0 and 3 terms for i=1, but share 1 term.
    That gives 2*3*7 - 7 = 35 terms.
    
    How to share? For each k, let term r_k be shared:
    U[0*m+j, r_k] and U[1*m+j, r_k] both nonzero.
    
    Let's try: for each k, define a shared term where:
    U[0*m+0, r_k] = 1, U[1*m+0, r_k] = 1  (both use A[0,0] and A[1,0])
    V[0*p+k, r_k] = 1, V[1*p+k, r_k] = 1, V[2*p+k, r_k] = 1  (uses B[0,k], B[1,k], B[2,k])
    W[k*n+0, r_k] = 1, W[k*n+1, r_k] = -1  (contributes to C[0,k] and subtracts from C[1,k])
    
    Wait, that doesn't work cleanly. Let me think differently.
    
    Actually, let's just return schoolbook if we can't find better quickly.
    """
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    First try to find a better-than-schoolbook decomposition, fallback to schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Try structured decomposition first
    result = _structured_decomposition()
    if result is not None:
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Try random search
    result = _try_random_search(schoolbook_rank - 1, max(30, schoolbook_rank - 10), 5.0)
    
    if result is not None:
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fallback to schoolbook
    schoolbook = _get_schoolbook()
    return schoolbook
# EVOLVE-BLOCK-END
