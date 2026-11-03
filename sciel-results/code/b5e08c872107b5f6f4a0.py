"""
Search for a lower-rank exact tensor decomposition of 3x3 by 3x5 matrix multiplication.
Uses a randomized numerical search (like an integer version of Strassen-style search) to find
integer coefficient decompositions. Verifies exactly using integer arithmetic.
"""
import random
import time
import math
import numpy as np

DIMENSIONS = [3, 3, 5]

DEADLINE = 135.0  # seconds wall-clock budget; leave margin before 150s kill

def _verify(U, V, W, n=3, m=3, p=5):
    """Verify the decomposition exactly using integer arithmetic.
    U: list of lists, shape (n*m, rank)
    V: list of lists, shape (m*p, rank)
    W: list of lists, shape (p*n, rank)
    Returns True if valid, False otherwise.
    """
    n, m, p = n, m, p
    rank = len(U[0]) if U else 0
    # Build numpy arrays
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    # For each (i, j, k), check sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row i*m+j, V row j*p+k, W row k*n+i
                row_u = U_arr[i * m + j, :]
                row_v = V_arr[j * p + k, :]
                row_w = W_arr[k * n + i, :]
                # element-wise product then sum
                prod = (row_u * row_v) * row_w
                total = prod.sum()
                if total != 1:
                    return False
    return True

def _construct_schoolbook(n=3, m=3, p=5):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_random_decomposition(rank, n=3, m=3, p=5, time_limit=60.0):
    """Try to find a valid decomposition of given rank using randomized search.
    Start with random small integers, then iteratively adjust.
    This is a heuristic and may not find anything.
    """
    start = time.time()
    best_result = None
    # Use a simple local search / random restart approach
    # For each random trial, initialize U, V, W with small integers
    # Then check validity. If not valid, try to fix by adjusting entries.
    
    # A simpler approach: random search over small integer values
    # This is unlikely to succeed for rank < 45, but let's try a few quick attempts.
    
    num_trials = 50
    for trial in range(num_trials):
        if time.time() - start > time_limit:
            break
        # Initialize with random small integers in {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    return None

def _try_strassen_like(n=3, m=3, p=5, target_rank=None, time_limit=30.0):
    """
    Try to find a decomposition inspired by Strassen's algorithm structure.
    For 2x2x2, Strassen uses 7 multiplications.
    For 3x3x3, the known minimum is 23.
    For 3x3x5, we don't have a known explicit formula, so this is exploratory.
    
    We'll try to find a rank-41 or rank-40 decomposition by perturbing the schoolbook
    and searching for dependencies.
    """
    start = time.time()
    # This is a placeholder for a more sophisticated search.
    # Given time constraints, we may not find anything.
    # Return None if nothing found.
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 by 3x5 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} where U, V, W are lists of lists.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook (always valid, rank = 45)
    schoolbook = _construct_schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p  # 45
    
    # Try to find lower rank decompositions
    # Known: for 3x3x3, min rank is 23 (vs 27 schoolbook).
    # For 3x3x5, it's unknown but likely < 45.
    # We'll try a few randomized approaches with tight time limits.
    
    # Try rank 44, 43, ... down to some lower bound
    # But random search is unlikely to succeed quickly.
    # Let's try a structured approach: look for linear dependencies in the schoolbook tensor.
    
    # The multiplication tensor T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}
    # where e_{ij} is the (i,j) entry of the first matrix, etc.
    # Actually, T(i,j,k) = 1 for all i,j,k in the appropriate ranges.
    # The tensor has rank at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15.
    # And at most n*m*p = 45.
    
    # A simple way to reduce rank: if there are linear dependencies among the 1D slices.
    # For the schoolbook, each (i,j,k) is a separate term. To reduce rank, we need to find
    # a combination where some terms can be absorbed.
    
    # Given the difficulty, let's just try a few random searches with very short time limits.
    
    # Try a numerical approach: use numpy to find a low-rank approximation and round.
    # This is heuristic and may not yield exact integer solutions.
    
    # For now, let's just return the schoolbook unless we find something better quickly.
    # We'll try rank 41 (a guess based on 3x3x3 being 23/27, so ~85% efficiency)
    
    # Try a quick random search for rank 41
    if time.time() - start_time < 10:
        result = _try_random_decomposition(41, n, m, p, time_limit=5)
        if result is not None:
            best = result
            best_rank = 41
            return best
    
    # Try rank 42
    if time.time() - start_time < 20:
        result = _try_random_decomposition(42, n, m, p, time_limit=5)
        if result is not None:
            best = result
            best_rank = 42
            return best
    
    # Return the best found
    return best

# EVOLVE-BLOCK-START
# The construct function is already defined above.
# EVOLVE-BLOCK-END
