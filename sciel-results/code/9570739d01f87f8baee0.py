import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _to_half_integer_matrix(U, tol=1e-9):
    """Convert a float matrix to half-integers if possible, else return None."""
    result = []
    for row in U:
        new_row = []
        for val in row:
            if abs(val) < tol:
                new_row.append(0)
            else:
                # Try to find nearest half-integer
                doubled = val * 2
                nearest = round(doubled)
                if abs(doubled - nearest) > tol:
                    return None
                if nearest % 2 == 0:
                    new_row.append(nearest // 2)
                else:
                    new_row.append(Fraction(nearest, 2))
        result.append(new_row)
    return result

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W form a valid matrix multiplication tensor decomposition."""
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    rank = U.shape[1]
    # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually the decomposition is: T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U has indices (i,j), V has indices (j,k), W has indices (k,i)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(expected - actual) > tol:
                    return False
    return True

def _try_rank_search(n, m, p, max_rank, min_rank, time_limit, start_time):
    """Try to find a decomposition with rank in [min_rank, max_rank]."""
    best = None
    for rank in range(min_rank, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        for trial in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization with small values
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Iterative refinement using gradient descent
            lr = 0.01
            for iteration in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute gradient for each factor
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                            # dT/dU[i*m+j,r] = V[j*p+k,r] * W[k*n+i,r]
                            # dT/dV[j*p+k,r] = U[i*m+j,r] * W[k*n+i,r]
                            # dT/dW[k*n+i,r] = U[i*m+j,r] * V[j*p+k,r]
                            # Loss = sum_{i,j,k} (T - 1)^2, so gradient of loss w.r.t. factor
                            # is 2*(T-1) * partial_T/partial_factor
                            T_val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = T_val - 1.0
                            for r in range(rank):
                                grad_U[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                                grad_V[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                                grad_W[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check if we've converged
                if iteration % 50 == 0:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                max_err = max(max_err, abs(T_val - 1.0))
                    if max_err < 1e-6:
                        break
            
            # Check if we found a valid solution
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(T_val - 1.0))
            
            if max_err < 1e-6:
                # Try to convert to half-integers
                U_int = _to_half_integer_matrix(U.tolist())
                V_int = _to_half_integer_matrix(V.tolist())
                W_int = _to_half_integer_matrix(W.tolist())
                
                if U_int is not None and V_int is not None and W_int is not None:
                    # Verify with exact arithmetic
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = Fraction(0)
                                for r in range(rank):
                                    s += U_int[i*m+j][r] * V_int[j*p+k][r] * W_int[k*n+i][r]
                                if s != 1:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        best = (rank, U_int, V_int, W_int)
                        return best
                
                # Even if not half-integer, check if it's a valid real decomposition
                # and try to find a nearby half-integer solution
                # For now, just record it as a candidate
    
    return best

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    start_time = time.time()
    time_limit = 100  # seconds, leave 50s safety margin
    
    # Try to find better decomposition with rank search
    # Start from lower ranks and go up
    min_rank = 15
    max_rank = rank_schoolbook - 1
    
    result = _try_rank_search(n, m, p, max_rank, min_rank, time_limit, start_time)
    
    if result is not None:
        rank, U, V, W = result
        if rank < best_rank:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
    
    return {"U": best_U, "V": best_V, "W": best_W}
