"""Matrix multiplication tensor decomposition search for 4x5 * 5x5."""
import time
import random
import itertools
from fractions import Fraction

# Use Fraction for exact arithmetic
# Dimensions
n, m, p = 4, 5, 5

# The matrix multiplication tensor T has entries T[i, j, k] = 1 if the output
# element (i,k) depends on input A[i,j] and B[j,k].
# In our factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k output}
# Actually, the standard definition: (AB)[i,k] = sum_j A[i,j] B[j,k]
# So the tensor is T[i, j, k] = 1 for all valid i,j,k where the product contributes.
# More precisely, the matrix multiplication tensor is a 3-tensor with indices (i, j, k)
# where i in [0,n), j in [0,m), k in [0,p). The value is 1 for all entries because
# every triple (i,j,k) contributes to the product.

# We want to find U, V, W of rank r such that:
# sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k

# This is equivalent to finding a rank-r decomposition of the matrix multiplication tensor.

# Known lower bounds: For n x m * m x p, the rank is at least max(n*m, m*p, n*p)
# For 4x5 * 5x5: lower bound is max(20, 25, 20) = 25
# Upper bound from schoolbook: 4*5*5 = 100

# Let's try to find a decomposition with rank close to the lower bound.
# We'll use a randomized approach with exact arithmetic.

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Try different ranks starting from the lower bound
    min_rank = max(n*m, m*p, n*p)  # 25
    
    # The schoolbook gives rank 100
    # Let's try to find something better
    
    # For small matrices, we can use known results or try to construct
    # a decomposition using the fact that this is a tensor rank problem.
    
    # One approach: use the "winograd" style or other fast matrix multiplication
    # But for 4x5*5x5, let's try a systematic search.
    
    # Actually, let's use a simpler approach: try rank values and use 
    # randomized initialization with exact arithmetic refinement.
    
    # For now, let's return the schoolbook decomposition which is known to work.
    # We can try to improve it.
    
    rank = n * m * p  # 100
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
    
    # Try to find a better decomposition
    # Use a randomized approach with exact arithmetic
    
    # Try ranks from lower bound up to current best
    best_rank = rank
    best_U = U
    best_V = V
    best_W = W
    
    # For each candidate rank, try to find a valid decomposition
    for candidate_rank in range(min_rank, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            # Initialize with small random rational numbers
            U_try = [[random.randint(-2, 2) / 2 for _ in range(candidate_rank)] for _ in range(n * m)]
            V_try = [[random.randint(-2, 2) / 2 for _ in range(candidate_rank)] for _ in range(m * p)]
            W_try = [[random.randint(-2, 2) / 2 for _ in range(candidate_rank)] for _ in range(p * n)]
            
            # Check if this decomposition works
            if _check_decomposition(U_try, V_try, W_try, candidate_rank):
                best_rank = candidate_rank
                best_U = U_try
                best_V = V_try
                best_W = W_try
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}


def _check_decomposition(U, V, W, rank):
    """Check if the given U, V, W exactly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True


# Let's also try a more structured approach using known fast matrix multiplication
# ideas or algebraic constructions.

# For 4x5 * 5x5, we can potentially use a block decomposition.
# Let's try a different strategy: use the fact that we can embed smaller
# fast matrix multiplication schemes.

# Actually, let's try a greedy/iterative approach where we build up the
# decomposition column by column.

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with schoolbook
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
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to reduce rank by finding linear dependencies
    # Use a systematic approach: try to eliminate columns
    
    # For each pair of columns, check if one can be expressed as a combination
    # of others, and if so, remove it.
    
    # This is complex, so let's use a simpler heuristic:
    # Try random subsets of columns and check if they still span the tensor.
    
    # Actually, let's use a different approach: try to find a decomposition
    # with rank equal to known bounds using algebraic methods.
    
    # For now, return the schoolbook decomposition.
    return {"U": best_U, "V": best_V, "W": best_W}
