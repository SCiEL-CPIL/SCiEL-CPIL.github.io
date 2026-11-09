"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin
    
    # The tensor T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # We need U (n*m x r), V (m*p x r), W (p*n x r) with:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*p, m*min(n,p)) = max(15, 12) = 15
    lower_bound = max(n * p, m * min(n, p))
    
    best_result = None
    
    for r in range(lower_bound, n * m * p + 1):
        if time.time() > deadline:
            break
        if best_result is not None:
            break  # We found a valid one, stop
            
        result = _try_rank(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
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


def _try_rank(n, m, p, r, deadline):
    """Try to find an exact half-integer decomposition of rank r."""
    
    # Try multiple random initializations
    for seed in range(20):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(seed * 1000 + 42)
        
        # Sparse random initialization: most entries zero, some non-zero
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Fill with random values, but keep sparsity
        for _ in range(r):
            # Pick random rows to set
            u_row = np.random.randint(0, n * m)
            v_row = np.random.randint(0, m * p)
            w_row = np.random.randint(0, p * n)
            U[u_row, _] = np.random.choice([-1, -0.5, 0.5, 1])
            V[v_row, _] = np.random.choice([-1, -0.5, 0.5, 1])
            W[w_row, _] = np.random.choice([-1, -0.5, 0.5, 1])
        
        # Ensure at least some structure: set first row of each to 1
        if n * m > 0:
            U[0, 0] = 1
        if m * p > 0:
            V[0, 0] = 1
        if p * n > 0:
            W[0, 0] = 1
        
        # Alternating least squares with polish
        U, V, W = _als_polish(n, m, p, r, U, V, W, deadline)
        
        if U is None:
            continue
        
        # Round to half-integers and verify
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if _verify(n, m, p, U_round, V_round, W_round):
            return _to_lists(n, m, p, U_round, V_round, W_round)
        
        # Also try integer rounding
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if _verify(n, m, p, U_int, V_int, W_int):
            return _to_lists(n, m, p, U_int, V_int, W_int)
    
    return None


def _als_polish(n, m, p, r, U, V, W, deadline):
    """Alternating least squares with final polish pass."""
    
    # Phase 1: Standard ALS iterations
    for iteration in range(200):
        if time.time() > deadline - 20:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check if we're close to half-integers
        if iteration % 20 == 0 and iteration > 0:
            U_check = np.round(U * 2) / 2
            V_check = np.round(V * 2) / 2
            W_check = np.round(W * 2) / 2
            if _verify(n, m, p, U_check, V_check, W_check):
                return U_check, V_check, W_check
    
    # Phase 2: Polish pass - try to nudge values closer to half-integers
    # Use a smooth objective: minimize sum of squared residuals + penalty for non-half-integer values
    for polish_iter in range(50):
        if time.time() > deadline - 10:
            break
        
        # Small perturbations toward half-integers
        U_target = np.round(U * 2) / 2
        V_target = np.round(V * 2) / 2
        W_target = np.round(W * 2) / 2
        
        # Blend current with target
        alpha = 0.3
        U = (1 - alpha) * U + alpha * U_target
        V = (1 - alpha) * V + alpha * V_target
        W = (1 - alpha) * W + alpha * W_target
        
        # One ALS step to correct
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except:
                    pass
    
    return U, V, W


def _verify(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _to_lists(n, m, p, U, V, W):
    """Convert numpy arrays to lists of lists with exact half-integer values."""
    r = U.shape[1]
    U_list = [[float(U[i, c]) for c in range(r)] for i in range(n * m)]
    V_list = [[float(V[i, c]) for c in range(r)] for i in range(m * p)]
    W_list = [[float(W[i, c]) for c in range(r)] for i in range(p * n)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
