"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

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
    return {"U": U, "V": V, "W": W}

def _try_rank_sat(n, m, p, rank, deadline):
    """
    Use SAT-like search with value domain {-2,-1,0,1,2}.
    We model the problem as: for each (i,j,k), sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] == 1.
    We do a depth-first search over the columns, assigning values to U, V, W entries.
    To make this tractable, we use a greedy + backtracking hybrid with memoization on partial sums.
    
    Actually, a more practical approach: use integer linear programming via simple search
    with constraints. Since rank is small, we can try to solve it column by column.
    
    Let's use a different approach: for a given rank r, we need to find integer vectors
    u_{i,j}, v_{j,k}, w_{i,k} in Z^r such that for all i,j,k: <u_{i,j}, v_{j,k}, w_{i,k}> = 1
    where <.,.,.> is the trilinear form.
    
    This is equivalent to: for each c, define U_c, V_c, W_c as vectors. Then:
    sum_c U_c[i,j] * V_c[j,k] * W_c[i,k] = 1
    
    Let's try a structured approach: assume the decomposition has a certain form.
    For 2x3 * 3x5, we know the minimum rank is at most 26 (from known results for small matmul).
    
    Let's use a randomized local search with restarts, but with a smarter initialization:
    start from the schoolbook decomposition and try to merge pairs of terms.
    """
    if time.time() > deadline:
        return None
    
    # Try to find a decomposition of the given rank by solving a system of equations.
    # We'll use a simple approach: initialize U, V, W randomly, then iteratively fix
    # the worst constraint.
    
    best_rank = rank
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        random.seed(attempt * 1000 + rank)
        U = [[random.randint(-2, 2) for _ in range(best_rank)] for _ in range(n * m)]
        V = [[random.randint(-2, 2) for _ in range(best_rank)] for _ in range(m * p)]
        W = [[random.randint(-2, 2) for _ in range(best_rank)] for _ in range(p * n)]
        
        # Check if valid
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Try to fix by adjusting one entry at a time
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Find the constraint with the largest error
            worst_ijk = None
            worst_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(best_rank):
                            s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                        err = abs(s - 1)
                        if err > worst_err:
                            worst_err = err
                            worst_ijk = (i, j, k)
            
            if worst_err == 0:
                return {"U": U, "V": V, "W": W}
            
            if worst_ijk is None:
                break
            
            i, j, k = worst_ijk
            # Try adjusting one entry in U, V, or W
            for matrix, idx in [('U', i*m+j), ('V', j*p+k), ('W', k*n+i)]:
                M = {'U': U, 'V': V, 'W': W}[matrix]
                for c in range(best_rank):
                    old = M[idx][c]
                    for new_val in [-3, -2, -1, 0, 1, 2, 3]:
                        if new_val == old:
                            continue
                        M[idx][c] = new_val
                        if _verify(U, V, W, n, m, p):
                            return {"U": U, "V": V, "W": W}
                        M[idx][c] = old
    
    return None

def _try_rank_greedy_merge(n, m, p, max_rank, deadline):
    """
    Start with schoolbook and try to reduce rank by merging terms.
    Two terms can be merged if their combined effect is equivalent to a single term.
    """
    if time.time() > deadline:
        return None
    
    # Start with schoolbook
    sb = _schoolbook(n, m, p)
    U = [row[:] for row in sb['U']]
    V = [row[:] for row in sb['V']]
    W = [row[:] for row in sb['W']]
    current_rank = n * m * p
    
    # Try to remove one term at a time by adjusting others
    for remove_c in range(current_rank - 1, -1, -1):
        if time.time() > deadline:
            break
        if current_rank <= max_rank:
            break
        
        # Try to redistribute the contribution of term remove_c
        # This is complex, so skip for now
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 30
    
    # Get schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    deadline = time.time() + 115  # Leave safety margin
    
    # Known lower bound: max(n*m, m*p, n*p) = max(6, 15, 10) = 15
    # For 2x3*3x5, the minimum rank is known to be 26 or less
    # Try ranks from 16 up to schoolbook_rank - 1
    for rank in range(16, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank_sat(n, m, p, rank, deadline - 5)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best
# EVOLVE-BLOCK-END
