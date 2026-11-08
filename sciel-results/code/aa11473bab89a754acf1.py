"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Leave 10s safety margin

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _check_decomposition(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) in the MM tensor
    # Actually, the MM tensor is T[i,j,k] = delta where output[i,k] = sum_j A[i,j] B[j,k]
    # The tensor entry is 1 if we're computing output[i,k] from A[i,j]*B[j,k]
    # So for each (i,j,k), T[i,j,k] = 1
    # The decomposition requires: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _check_decomposition_float(n, m, p, U, V, W, tol=1e-9):
    """Fast float check for approximate verification."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(n, m, p, U, V, W, max_iter=500):
    """Try to reduce rank by finding linear dependencies and eliminating columns."""
    rank = len(U[0])
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    for iteration in range(max_iter):
        if _time_left() < 5:
            break
        
        # Try to find two columns that are linearly dependent or can be merged
        # Strategy: for each pair of columns (a, b), check if we can find a linear combination
        # that preserves the tensor while allowing us to eliminate one column
        
        found = False
        for a in range(rank):
            if _time_left() < 5:
                    break
            for b in range(a + 1, rank):
                # Check if column b is a scalar multiple of column a in all three matrices
                # Or more generally, if there exists alpha, beta such that
                # for all (i,j,k): U[i*m+j,a]*V[j*p+k,a]*W[k*n+i,a] * alpha +
                #                  U[i*m+j,b]*V[j*p+k,b]*W[k*n+i,b] * beta
                # can replace one of them
                
                # Simple check: are the columns identical?
                same = True
                for r in range(len(U)):
                    if U[r][a] != U[r][b]:
                        same = False
                        break
                if same:
                    for r in range(len(V)):
                        if V[r][a] != V[r][b]:
                            same = False
                            break
                if same:
                    for r in range(len(W)):
                        if W[r][a] != W[r][b]:
                            same = False
                            break
                if same:
                    # Remove column b
                    for row in U:
                        del row[b]
                    for row in V:
                        del row[b]
                    for row in W:
                        del row[b]
                    rank -= 1
                    found = True
                    break
            if found:
                break
        
        if not found:
            # Try random linear combinations to see if we can create a zero column
            for trial in range(20):
                if _time_left() < 5:
                    break
                # Pick two random columns and a random ratio
                a, b = random.sample(range(rank), 2)
                # Try to find alpha such that column a - alpha * column b creates simplification
                # For simplicity, try alpha = 1, -1, 2, 1/2
                for alpha_val in [1, -1, 2, Fraction(1,2), Fraction(-1,2), 3]:
                    # Create a new column: U_new = U[a] - alpha * U[b]
                    # Check if this leads to a zero row in some matrix
                    zero_count = 0
                    for r in range(len(U)):
                        if U[r][a] - alpha_val * U[r][b] == 0:
                            zero_count += 1
                    # If many zeros, this might be useful
                    # This is a heuristic; exact verification would be expensive
                    pass
            break  # No simple reduction found
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Schoolbook initialization
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
    
    # Verify schoolbook
    assert _check_decomposition(n, m, p, U, V, W), "Schoolbook decomposition failed"
    
    # Try to reduce rank
    U, V, W = _try_reduce_rank(n, m, p, U, V, W, max_iter=200)
    
    current_rank = len(U[0])
    
    # Try structured ansatz: exploit symmetry in the 3x3x3 sub-problem
    # The 3x3x3 MM tensor has known rank 20. We can try to embed that structure.
    # For 3x3x4, we can think of it as two 3x3x2 problems or similar decompositions.
    
    # Alternative approach: try random rank-r decompositions for r < current_rank
    # and verify exactly
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    # Try to find a lower rank decomposition using random search
    for target_rank in range(best_rank - 1, max(15, best_rank - 15), -1):
        if _time_left() < 10:
            break
        
        # Try several random initializations
        for attempt in range(50):
            if _time_left() < 10:
                break
            
            # Random initialization with small integers
            U_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
            W_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Quick float check
            if _check_decomposition_float(n, m, p, U_rand, V_rand, W_rand, tol=1e-6):
                # Exact check
                if _check_decomposition(n, m, p, U_rand, V_rand, W_rand):
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    best_rank = target_rank
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
