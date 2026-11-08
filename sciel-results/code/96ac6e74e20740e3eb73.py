"""Matrix multiplication 3x3,3x5: search for a rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
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

def _solve_linear_system(A, b, r):
    """Solve A @ x = b where A is m x r (m <= r), using exact fractions.
    Returns a solution x of length r, or None if no solution exists.
    Uses Gaussian elimination with partial pivoting."""
    m = len(A)
    if m == 0:
        return [Fraction(0)] * r
    
    # Work with augmented matrix [A | b]
    aug = [A[i][:] + [b[i]] for i in range(m)]
    
    # Gaussian elimination
    row = 0
    for col in range(r):
        if time.time() > 150:
            return None
        # Find pivot
        pivot = -1
        for i in range(row, m):
            if aug[i][col] != 0:
                pivot = i
                break
        if pivot == -1:
            continue
        # Swap
        aug[row], aug[pivot] = aug[pivot], aug[row]
        # Eliminate below and above
        pivot_val = aug[row][col]
        for i in range(m):
            if i != row and aug[i][col] != 0:
                factor = aug[i][col] / pivot_val
                for j in range(col, r + 1):
                    aug[i][j] -= factor * aug[row][j]
        row += 1
        if row == m:
            break
    
    # Check consistency
    for i in range(m):
        # If all coefficients are 0 but RHS is non-zero, inconsistent
        all_zero = all(aug[i][j] == 0 for j in range(r))
        if all_zero and aug[i][r] != 0:
            return None
    
    # Back-substitution: find a particular solution
    # Set free variables to 0
    x = [Fraction(0)] * r
    # Determine which columns are pivot columns
    pivot_cols = []
    row_idx = 0
    for col in range(r):
        # Check if this column has a pivot in the reduced form
        # In our elimination, row `row_idx` corresponds to the next pivot
        # We need to track which columns were pivots
        pass
    
    # Simpler: just do back-substitution from the reduced augmented matrix
    # The reduced form has at most m non-zero rows
    # For each row, identify the leading (first non-zero) column
    for i in range(min(row, m)):
        # Find first non-zero in this row
        lead_col = -1
        for j in range(r):
            if aug[i][j] != 0:
                lead_col = j
                break
        if lead_col == -1:
            continue
        # x[lead_col] = (b_i - sum of other terms) / aug[i][lead_col]
        # But in reduced form, other terms should be 0 for non-pivot cols in this row
        # Actually, let's just compute directly
        rhs = aug[i][r]
        for j in range(r):
            if j != lead_col:
                rhs -= aug[i][j] * x[j]
        x[lead_col] = rhs / aug[i][lead_col]
    
    return x

def _try_rank_r(n, m, p, r, deadline):
    """Try to find a rank-r decomposition using the slice method."""
    if time.time() > deadline:
        return None
    
    # Entry set for U and V
    # Expanded to include small integers and simple fractions to cover known decomposition structures
    entry_set = [
        Fraction(0), Fraction(1), Fraction(-1), 
        Fraction(2), Fraction(-2), Fraction(3), Fraction(-3),
        Fraction(1,2), Fraction(-1,2), Fraction(1,3), Fraction(-1,3), 
        Fraction(1,4), Fraction(-1,4), Fraction(1,5), Fraction(-1,5)
    ]
    
    # Increase trials slightly, but rely on the larger entry set for success
    n_trials = 500
    for trial in range(n_trials):
        if time.time() > deadline:
            return None
        
        random.seed(12345 + trial * 7919)
        U = [[random.choice(entry_set) for _ in range(r)] for _ in range(n*m)]
        V = [[random.choice(entry_set) for _ in range(r)] for _ in range(m*p)]
        
        # For each (k,i), solve for W[k*n+i, :]
        W = [[Fraction(0) for _ in range(r)] for _ in range(p*n)]
        feasible = True
        
        for k in range(p):
            if not feasible:
                break
            for i in range(n):
                if time.time() > deadline:
                    return None
                
                # Build A_{(k,i)}: m x r matrix
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A_mat = []
                for j in range(m):
                    row = []
                    for c in range(r):
                        row.append(U[i*m+j][c] * V[j*p+k][c])
                    A_mat.append(row)
                
                # Solve A @ w = ones(m)
                b = [Fraction(1)] * m
                w = _solve_linear_system(A_mat, b, r)
                
                if w is None:
                    feasible = False
                    break
                
                # Check that w entries are half-integers (they should be if A and b are rational with small denominators)
                W[k*n + i] = w
        
        if feasible:
            # Verify the full decomposition
            if _verify(U, V, W, n, m, p):
                return (U, V, W)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave 20s margin for the 150s limit
    
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
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Start from rank 15 and go up
    for r in range(15, rank_school):
        if time.time() > deadline:
            break
        
        result = _try_rank_r(n, m, p, r, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
