"""Matrix multiplication tensor decomposition for 2x3 * 3x7."""
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly satisfies the tensor equation."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if s != 1:
                    return False
    return True

def _gauss_solve(A, b):
    """Solve Ax = b over Q using Gaussian elimination. Returns None if no unique solution."""
    n = len(A)
    if n == 0:
        return []
    m = len(A[0])
    # Augmented matrix
    aug = [[Fraction(A[i][j]) for j in range(m)] + [Fraction(b[i])] for i in range(n)]
    
    # Forward elimination
    for col in range(min(n, m)):
        # Find pivot
        pivot_row = None
        for row in range(col, n):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        # Swap
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        # Normalize
        pv = aug[col][col]
        for j in range(col, m + 1):
            aug[col][j] /= pv
        # Eliminate
        for row in range(n):
            if row != col and aug[row][col] != 0:
                factor = aug[row][col]
                for j in range(col, m + 1):
                    aug[row][j] -= factor * aug[col][j]
    
    # Check for inconsistency
    for row in range(n):
        all_zero = all(aug[row][j] == 0 for j in range(m))
        if all_zero and aug[row][m] != 0:
            return None
    
    # Extract solution (assuming full rank for simplicity)
    x = [Fraction(0)] * m
    for row in range(min(n, m)):
        if all(aug[row][j] == 0 for j in range(m)):
            continue
        # Find the column with 1
        for col in range(m):
            if aug[row][col] == 1:
                x[col] = aug[row][m]
                break
        else:
            # Non-pivot column - free variable, set to 0
            pass
    
    # Verify
    for i in range(n):
        s = sum(Fraction(A[i][j]) * x[j] for j in range(m))
        if s != Fraction(b[i]):
            return None
    return x

def _try_rank_with_fixed_W(n, m, p, rank, W_fixed, deadline):
    """Try to find U, V given W_fixed such that the tensor equation holds."""
    # For each j, we need: for all i,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let's fix U and solve for V, or vice versa.
    # 
    # Strategy: Fix U (random rational), then for each j, solve for V[:, j*p+k] 
    # from the linear system. But V entries are shared across different (i,k) pairs.
    #
    # Alternative: For each j, define matrix A_j of size (n*p) x rank where
    # A_j[i*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
    # Then we need: A_j * v_j = 1 (vector of ones), where v_j is V[:, j*p+k] for k=0..p-1
    # But v_j has p*rank entries (one per k), and we have n*p equations.
    # Actually V[j*p+k, r] for fixed j gives p rows, each of length rank.
    # So v_j is a vector of length p*rank.
    # The equation is: for each i,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is n*p equations in p*rank unknowns (the V entries for this j).
    
    # Let's try: fix U randomly, then for each j solve for V[j*p: (j+1)*p, :]
    # But V entries for different j are independent, so we can solve each j separately.
    
    # However, U is shared across all j. So we need to choose U such that for ALL j,
    # the system is solvable. This is a global constraint.
    
    # Simpler approach: Fix both U and W, then V is determined (if the systems are consistent).
    # Let's fix U and W, then check if V exists.
    
    # For each j:
    #   Build system: for all i,k: sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = 1
    #   This is n*p equations in p*rank unknowns.
    #   If n*p > p*rank, we need n > rank for overdetermined. For n=2, rank >= 2.
    #   If consistent, V is determined (up to free variables if underdetermined).
    
    # Let's fix U and W, then solve for V.
    
    U = [[Fraction(random.randint(-3, 3)) for _ in range(rank)] for _ in range(n * m)]
    
    V = [[Fraction(0) for _ in range(rank)] for _ in range(m * p)]
    
    for j in range(m):
        # Build system for V[j*p: (j+1)*p, :]
        # Unknowns: V[j*p+k, r] for k in 0..p-1, r in 0..rank-1 => p*rank unknowns
        # Equations: for i in 0..n-1, k in 0..p-1: sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = 1
        
        # This is a linear system. Let's build it.
        num_eqs = n * p
        num_unk = p * rank
        
        # Matrix A: num_eqs x num_unk
        A_sys = [[Fraction(0) for _ in range(num_unk)] for _ in range(num_eqs)]
        b_sys = [Fraction(1) for _ in range(num_eqs)]
        
        for i in range(n):
            for k in range(p):
                eq_idx = i * p + k
                for r in range(rank):
                    unk_idx = k * rank + r
                    A_sys[eq_idx][unk_idx] = Fraction(U[i * m + j][r]) * Fraction(W[k * n + i][r])
        
        # Solve
        sol = _gauss_solve(A_sys, b_sys)
        if sol is None:
            return None
        
        # Extract V entries
        for k in range(p):
            for r in range(rank):
                V[j * p + k][r] = sol[k * rank + r]
    
    # Verify
    if _verify_decomposition(U, V, W_fixed, n, m, p):
        return U, V, W_fixed
    
    return None

def _try_rank_search(n, m, p, rank, deadline):
    """Try to find a rank-R decomposition using random W and solving for U, V."""
    max_attempts = 50
    attempt = 0
    while attempt < max_attempts and time.time() < deadline:
        attempt += 1
        # Random W
        W = [[Fraction(random.randint(-3, 3)) for _ in range(rank)] for _ in range(p * n)]
        
        result = _try_rank_with_fixed_W(n, m, p, rank, W, deadline)
        if result is not None:
            return result
    
    return None

def _try_rank_symmetric(n, m, p, rank, deadline):
    """Try a more structured approach: use the fact that for each j, the problem is independent."""
    # For rank R, we need R triples of vectors.
    # Let's try: for each rank component r, choose U_r, V_r, W_r.
    # The constraint is: for all i,j,k: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1
    
    # This is a system of n*m*p equations in rank*(n*m + m*p + p*n) unknowns.
    # For n=2, m=3, p=7: 42 equations in rank*41 unknowns.
    # For rank=1: 41 unknowns, 42 equations. Likely no solution.
    # For rank=2: 82 unknowns, 42 equations. Should be solvable if consistent.
    
    # Let's try rank=2 first with a structured approach.
    # Fix W, then the system becomes linear in U and V.
    # Actually, for fixed W, the equations are:
    # sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1
    # This is bilinear in U and V, so not linear.
    
    # Alternative: fix U, then it's linear in V (for each j independently).
    # Or fix V, then it's linear in U.
    
    # Let's fix V randomly, then solve for U.
    # For fixed V, the equations are:
    # For all i,j,k: sum_r U_r[i*m+j] * (V_r[j*p+k] * W_r[k*n+i]) = 1
    # Let C_r[i*m+j, k*n+i] = V_r[j*p+k] * W_r[k*n+i] ... no, that's not right.
    
    # Actually, for fixed V and W, the equation for U is:
    # For all i,j,k: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1
    # This is linear in U. For each (i,j), we have p equations (one per k) in rank unknowns (U_r[i*m+j]).
    # So for each (i,j), we solve a p x rank system.
    # If p > rank, overdetermined. For p=7, rank=2: 7 equations in 2 unknowns.
    # This will only be solvable if the equations are consistent.
    
    # Let's try: fix W, fix U randomly, solve for V.
    # For fixed U and W, for each j, solve for V[:, j*p:k] as before.
    
    # This is what _try_rank_with_fixed_W does. Let's just use that.
    return _try_rank_search(n, m, p, rank, deadline)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Theoretical lower bounds: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Also, rank >= max(n, m, p) = 7
    # Let's try ranks from a reasonable lower bound upward
    
    # Start from rank 21 (the matricization lower bound) down to smaller values
    # Actually, let's try a range of ranks
    
    # For efficiency, try ranks in a strategic order
    candidate_ranks = list(range(21, rank_school)) + list(range(1, 21))
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:  # 10s safety margin
            break
        if rank >= best_rank:
            continue
        
        result = _try_rank_search(n, m, p, rank, deadline - 5)
        if result is not None:
            U_found, V_found, W_found = result
            best_result = {"U": U_found, "V": V_found, "W": W_found}
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
