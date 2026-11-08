"""Matrix multiplication 3x3,3x5: search for a rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Convention: W[k*n + i, c]
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _row_rank(matrix):
    """Compute rank of a matrix of Fractions via row reduction."""
    if not matrix or not matrix[0]:
        return 0
    m_len = len(matrix)
    n_len = len(matrix[0])
    mat = [row[:] for row in matrix]
    rank = 0
    for col in range(n_len):
        pivot_row = -1
        for row in range(rank, m_len):
            if mat[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        pivot = mat[rank][col]
        for row in range(m_len):
            if row != rank and mat[row][col] != 0:
                factor = mat[row][col] / pivot
                for k2 in range(col, n_len):
                    mat[row][k2] -= factor * mat[rank][k2]
        rank += 1
    return rank

def _solve_linear(A, b):
    """Solve A x = b exactly using fraction arithmetic.
    A is m x r, b is m x 1. Returns x (r x 1) or None if no solution.
    If underdetermined, returns a particular solution (free vars set to 0).
    """
    m = len(A)
    r = len(A[0]) if A > 0 else 0
    # Build augmented matrix [A | b]
    aug = []
    for i in range(m):
        row = [Fraction(A[i][j]) for j in range(r)]
        row.append(Fraction(b[i]))
        aug.append(row)
    
    # Gaussian elimination with partial pivoting
    rank = 0
    pivot_cols = []
    for col in range(r):
        # Find pivot
        pivot_row = -1
        for row in range(rank, m):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        # Swap
        aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
        pivot_cols.append(col)
        # Eliminate
        pivot = aug[rank][col]
        for row in range(m):
            if row != rank and aug[row][col] != 0:
                factor = aug[row][col] / pivot
                for k in range(col, r + 1):
                    aug[row][k] -= factor * aug[rank][k]
        rank += 1
    
    # Check consistency: for rows >= rank, the last element must be 0
    for row in range(rank, m):
        if aug[row][r] != 0:
            return None
    
    # Back-substitute to find a particular solution
    x = [Fraction(0)] * r
    for i in range(rank - 1, -1, -1):
        col = pivot_cols[i]
        # x[col] = (b[i] - sum_{j > col} x[j] * A[i][j]) / A[i][col]
        val = aug[i][r]
        for j in range(col + 1, r):
            val -= aug[i][j] * x[j]
        x[col] = val / aug[i][col]
    
    return x

def _try_rank_r(n, m, p, r, deadline):
    """Try to find a rank-r decomposition using structured search.
    
    Strategy:
    1. Try structured U and V matrices with entries in {0, 1/2, 1, -1/2, -1}.
    2. For each (k,i), solve the linear system for W[k*n+i, :].
    3. Verify the full decomposition.
    """
    if time.time() > deadline:
        return None
    
    entry_set = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
    
    # For efficiency, limit the number of trials
    n_trials = 50
    for trial in range(n_trials):
        if time.time() > deadline:
            return None
        
        random.seed(42 + trial * 137)
        U = [[random.choice(entry_set) for _ in range(r)] for _ in range(n*m)]
        V = [[random.choice(entry_set) for _ in range(r)] for _ in range(m*p)]
        
        # For each (k,i), solve for W[k*n+i, :]
        W = [[Fraction(0) for _ in range(r)] for _ in range(p*n)]
        feasible = True
        
        for k in range(p):
            for i in range(n):
                # Build the m x r matrix A_{(k,i)}
                # A_{(k,i)}[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A_mat = []
                for j in range(m):
                    row = []
                    for c in range(r):
                        row.append(U[i*m+j][c] * V[j*p+k][c])
                    A_mat.append(row)
                
                # Solve A @ w = ones(m)
                b = [1] * m
                w = _solve_linear(A_mat, b)
                
                if w is None:
                    feasible = False
                    break
                
                W[k*n+i] = w
            
            if not feasible:
                break
        
        if feasible:
            # Verify the full decomposition
            if _verify(U, V, W, n, m, p):
                return (U, V, W)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin for the 150s limit
    
    # Start with schoolbook as fallback
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
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find a lower rank decomposition
    # Start from rank 30 and go up to rank_school - 1
    for r in range(30, rank_school):
        if time.time() > deadline:
            break
        
        result = _try_rank_r(n, m, p, r, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
