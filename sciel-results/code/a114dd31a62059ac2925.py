"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition with tolerance."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x):
    return abs(2 * x - round(2 * x)) < 1e-9

def _verify_entries(U, V, W):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _try_constructed_rank(n, m, p, rank, deadline):
    """
    Try to find a rank-r decomposition using alternating least squares.
    For fixed V and W, the problem is linear in U, and vice versa.
    """
    if time.time() > deadline:
        return None
    
    import random
    rng = random.Random(rank * 1000 + 42)
    
    # Initialize with small random values
    def _rand_halfint():
        choices = [-1, -0.5, 0, 0.5, 1]
        return rng.choice(choices)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    # Try a few random initializations
    for trial in range(3):
        if time.time() > deadline - 2:
            break
        
        U = [[_rand_halfint() for _ in range(rank)] for _ in range(n * m)]
        V = [[_rand_halfint() for _ in range(rank)] for _ in range(m * p)]
        W = [[_rand_halfint() for _ in range(rank)] for _ in range(p * n)]
        
        # Alternating minimization
        for it in range(50):
            if time.time() > deadline - 2:
                break
            
            # Update U: for each (i,j), solve sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is a linear system in U[i*m+j,:]
            for i in range(n):
                for j in range(m):
                    # Build linear system: for each k, sum_r U[j+r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                    # Let M[k,r] = V[j*p+k,r] * W[k*n+i,r], then sum_r U[r] * M[k,r] = 1
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            A_mat[k, r] = V[j * p + k][r] * W[k * n + i][r]
                    b_vec = np.ones(p)
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for r in range(rank):
                            U[i * m + j][r] = sol[r]
                    except:
                        pass
            
            # Update V: for each (j,k), solve sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            A_mat[i, r] = U[i * m + j][r] * W[k * n + i][r]
                    b_vec = np.ones(n)
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for r in range(rank):
                            V[j * p + k][r] = sol[r]
                    except:
                        pass
            
            # Update W: for each (k,i), solve sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A_mat[j, r] = U[i * m + j][r] * V[j * p + k][r]
                    b_vec = np.ones(m)
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for r in range(rank):
                            W[k * n + i][r] = sol[r]
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        err += (s - 1.0) ** 2
            err = np.sqrt(err / (n * m * p))
            
            if err < best_err:
                best_err = err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
            
            if err < 1e-6:
                break
        
        if best_err < 1e-6:
            break
    
    if best_err < 1e-6 and best_U is not None:
        # Round to nearest half-integer
        def _round_halfint(x):
            return round(2 * x) / 2.0
        
        U = [[_round_halfint(x) for x in row] for row in best_U]
        V = [[_round_halfint(x) for x in row] for row in best_V]
        W = [[_round_halfint(x) for x in row] for row in best_W]
        
        # Verify after rounding
        if _verify_decomposition(U, V, W, n, m, p) and _verify_entries(U, V, W):
            return U, V, W
    
    return None

def _try_linear_solve(n, m, p, rank, deadline):
    """
    Try a structured approach: assume W has a Kronecker-like structure.
    For n=2, try W[k*n+i, r] = alpha_r[i] * beta_r[k]
    """
    if time.time() > deadline:
        return None
    
    import random
    rng = random.Random(rank * 777 + 13)
    
    best_result = None
    best_err = float('inf')
    
    for trial in range(2):
        if time.time() > deadline - 2:
            break
        
        # Initialize alpha and beta
        alpha = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n)]
        beta = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p)]
        
        W = [[0] * rank for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W[k * n + i][r] = alpha[i][r] * beta[k][r]
        
        # Now alternate: fix W, solve for U and V
        U = [[0.5 for _ in range(rank)] for _ in range(n * m)]
        V = [[0.5 for _ in range(rank)] for _ in range(m * p)]
        
        for it in range(30):
            if time.time() > deadline - 2:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            A_mat[k, r] = V[j * p + k][r] * W[k * n + i][r]
                    b_vec = np.ones(p)
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for r in range(rank):
                            U[i * m + j][r] = sol[r]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            A_mat[i, r] = U[i * m + j][r] * W[k * n + i][r]
                    b_vec = np.ones(n)
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for r in range(rank):
                            V[j * p + k][r] = sol[r]
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        err += (s - 1.0) ** 2
            err = np.sqrt(err / (n * m * p))
            
            if err < best_err:
                best_err = err
                if err < 1e-6:
                    def _round_halfint(x):
                        return round(2 * x) / 2.0
                    U_r = [[_round_halfint(x) for x in row] for row in U]
                    V_r = [[_round_halfint(x) for x in row] for row in V]
                    W_r = [[_round_halfint(x) for x in row] for row in W]
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p) and _verify_entries(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    best_result = (U_r, V_r, W_r)
                    break
            if err < 1e-6:
                break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not (_verify_decomposition(U, V, W, n, m, p) and _verify_entries(U, V, W)):
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    # For [2,5,6], lower bound is max(10, 30, 12) = 30
    # Try ranks from 59 down to 30, but prioritize higher ranks first (easier to solve)
    
    # First try some specific ranks that are more likely to have solutions
    candidate_ranks = [55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]
    
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try constructed approach first
        result = _try_constructed_rank(n, m, p, r, deadline)
        if result is not None:
            U, V, W = result
            best_rank = r
            break
        
        # Try linear solve approach
        result = _try_linear_solve(n, m, p, r, deadline)
        if result is not None:
            U, V, W = result
            best_rank = r
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
