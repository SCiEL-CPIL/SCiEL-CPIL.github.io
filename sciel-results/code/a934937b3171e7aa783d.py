"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i, j, k] must be 1 for all i, j, k
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _round_to_half_integer(x):
    """Round a float to the nearest half-integer."""
    return round(x * 2) / 2.0

def _find_decomposition(n, m, p, target_rank, time_limit):
    """
    Attempts to find a rank-target_rank decomposition for n x m x p matrix multiplication.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # Number of equations is n*m*p
    n_eq = n * m * p
    
    # We want to minimize sum (dot(U_i, V_j, W_k) - 1)^2
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    
    # Precompute index mappings
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            return None
        
        np.random.seed(attempt * 12345 + target_rank)
        
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Gradient descent
        lr = 0.05
        for iteration in range(500):
            if time.time() - start_time > time_limit:
                return None
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # Compute gradients
            # err[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1
            # dU[i*m+j, r] = sum_k err[i,j,k] V[j*p+k, r] W[k*n+i, r]
            # dV[j*p+k, r] = sum_i err[i,j,k] U[i*m+j, r] W[k*n+i, r]
            # dW[k*n+i, r] = sum_j err[i,j,k] U[i*m+j, r] V[j*p+k, r]
            
            # To speed up, we can vectorize partially or just loop since n,m,p are small
            # n=2, m=3, p=3 -> 18 terms. R is up to 17.
            # 18 * 17 * 3 = ~900 ops per gradient step. Very fast.
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        # val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                        val = 0.0
                        for r in range(target_rank):
                            val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                        
                        err = val - 1.0
                        
                        for r in range(target_rank):
                            term_vw = V[v_idx, r] * W[w_idx, r]
                            term_uw = U[u_idx, r] * W[w_idx, r]
                            term_uv = U[u_idx, r] * V[v_idx, r]
                            
                            dU[u_idx, r] += err * term_vw
                            dV[v_idx, r] += err * term_uw
                            dW[w_idx, r] += err * term_uv
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            # Calculate max error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = 0.0
                        for r in range(target_rank):
                            val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-6:
                # Try to round to half-integers
                U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
                V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
                W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
                
                # Verify exactness
                if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                
                # If rounding failed, maybe try integer rounding
                U_int = np.round(U).astype(int)
                V_int = np.round(V).astype(int)
                W_int = np.round(W).astype(int)
                if _verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
            # Adaptive learning rate
            if iteration % 50 == 0:
                lr *= 0.95
        
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the schoolbook decomposition of rank n*m*p
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
    
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try to find a lower rank decomposition
    # Lower bound is max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    # We try ranks from 9 to 17
    
    for target_rank in range(9, rank_school):
        if time.time() > deadline - 10:  # Keep 10s safety margin
            break
        
        # Allocate time for this rank based on remaining time
        # Simple heuristic: more time for lower ranks
        time_for_rank = min(20, (deadline - time.time()) / (rank_school - target_rank))
        
        result = _find_decomposition(n, m, p, target_rank, time_for_rank)
        
        if result is not None:
            U_found, V_found, W_found = result
            best_result = {"U": U_found, "V": V_found, "W": W_found}
            best_rank = target_rank
            break  # Found the lowest rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
