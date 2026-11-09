"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W form an exact decomposition.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_r_linear_solve(n, m, p, r, max_attempts=200, seed=42):
    """
    Try to find a rank-r decomposition by fixing U and V randomly 
    and solving for W using linear systems.
    """
    np.random.seed(seed)
    
    # We want integer solutions. 
    # If we pick U, V from {-1, 0, 1}, the coefficients in the linear system 
    # for W will be integers. 
    # If the system has a solution, and we solve it exactly (or check for integer LSQ),
    # we might find an integer W.
    
    for attempt in range(max_attempts):
        # Generate random sparse U and V
        # Using small integers helps with exactness
        U = np.random.randint(-1, 2, size=(n * m, r))
        V = np.random.randint(-1, 2, size=(m * p, r))
        
        # Ensure some non-zeros to avoid trivial 0=1 failures
        # If a row in the linear system is all zeros, it's 0=1 -> No solution.
        # We check this during solve.
        
        W = np.zeros((p * n, r))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Construct the linear system for W[k, i, :]
                # Variables: W[k, i, 0], ..., W[k, i, r-1]
                # Equations: for j in 0..m-1:
                #   sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] = 1
                
                A_mat = np.zeros((m, r))
                for j in range(m):
                    # U row index: i*m + j
                    # V row index: j*p + k
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Solve A_mat * w_vec = ones(m)
                # If A_mat has rank < m, or no solution, this will fail or give non-integer.
                # We use least squares and check residual.
                try:
                    w_vec, residuals, rank, s = lstsq(A_mat, np.ones(m), cond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if the solution is close to integers
                # Since coefficients are integers, if a rational solution exists,
                # it might be integer.
                # We check if the residual is small AND the solution is close to integers.
                if len(residuals) > 0 and residuals[0] > 1e-9:
                    valid = False
                    break
                
                # Check integer-ness
                w_int = np.round(w_vec)
                if np.linalg.norm(A_mat @ w_int - np.ones(m)) > 1e-9:
                    valid = False
                    break
                
                W[k * n + i, :] = w_int
        
        if not valid:
            continue
            
        # Final verification
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Trivial solution fallback
    rank_trivial = n * m * p
    U_triv = np.zeros((n * m, rank_trivial))
    V_triv = np.zeros((m * p, rank_trivial))
    W_triv = np.zeros((p * n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, col] = 1.0
                V_triv[j * p + k, col] = 1.0
                W_triv[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_trivial
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Lower bound for rank is max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound upwards
    # We expect the rank to be close to the lower bound for small matrices
    # but Strassen shows it can be lower than nm*p.
    # For 3x5x7, lb=35.
    
    for r in range(lb, rank_trivial):
        if time.time() > deadline:
            break
            
        # Try multiple seeds for this rank
        # More attempts for lower ranks as they are harder
        attempts = 50 if r < lb + 10 else 10
        
        found = False
        for seed in range(attempts):
            if time.time() > deadline:
                break
                
            res = _try_rank_r_linear_solve(n, m, p, r, max_attempts=1, seed=seed + r * 1000)
            if res is not None:
                U, V, W = res
                best_rank = r
                best_U, best_V, best_W = U, V, W
                found = True
                break
                
        if found:
            # If we found a solution, we can try to go lower? 
            # No, we are iterating r upwards. 
            # If we found one, it's the minimum we found so far.
            # We can break if we are confident, but let's just return the best.
            # Actually, since we start from lb, the first r that works is the min.
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
