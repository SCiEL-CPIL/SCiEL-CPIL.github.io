import time
import numpy as np
from itertools import product
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [3, 3, 4]
DEADLINE = 115.0  # seconds, with safety margin before 120s

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Schoolbook rank is 36
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # We'll try ranks from 25 to 35 (36 is schoolbook)
    # The minimum possible rank for 3x3x4 is at least 27 (lower bound)
    # Let's try a few candidate ranks
    
    for rank in range(27, 36):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        # Try to find a decomposition of this rank
        result = _try_find_decomposition(n, m, p, rank, time.time() - start_time)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break  # Found a valid decomposition, stop searching lower ranks
    
    if best_U is None:
        # Return schoolbook
        return _schoolbook_decomposition(n, m, p)
    else:
        return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook_decomposition(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == delta[i,k] for all i,j,k
    
    Wait, the matrix multiplication tensor is:
    T[i, j, k] = delta[j, k] ... no.
    
    Matrix multiplication: (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    The tensor is T[i, k, j] = delta_{i,k} * something? 
    
    Actually, the standard matrix multiplication tensor M_{i,j,k} where 
    M_{i,j,k} = 1 if the term A_{i,j} * B_{j,k} appears, which it always does.
    
    So M_{i,j,k} = 1 for all i, j, k.
    
    The decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 for all i,j,k.
    
    Wait, that's not right either. Let me re-read the task.
    
    "Matrix multiplication 3,3,4" - multiply 3x3 by 3x4.
    Output is 3x4.
    
    The tensor for matrix multiplication: T_{i,k,j} = 1 (all ones), because every 
    product A_{i,j} * B_{j,k} contributes to C_{i,k}.
    
    Actually, the tensor is indexed as (input1, input2, output).
    Input1: A is 3x3, indexed by (i, j) where i is row, j is col.
    Input2: B is 3x4, indexed by (j, k) where j is row, k is col.
    Output: C is 3x4, indexed by (i, k).
    
    The tensor T_{(i,j), (j,k), (i,k)} = 1 for valid j.
    
    So T[i, j, k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3}.
    
    The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_find_decomposition(n, m, p, rank, elapsed):
    """Try to find a rank decomposition using optimization.
    
    We use the fact that we can fix some structure. 
    For 3x3x4, we can try to use the known rank-20 decomposition for 3x3x3 
    and extend it, or use a direct optimization approach.
    
    Strategy: Use random initialization and gradient descent on the 
    objective sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
    
    We'll try a few random restarts.
    """
    # Initialize with random values
    np.random.seed(42)
    
    U0 = np.random.randn(n * m, rank) * 0.5
    V0 = np.random.randn(m * p, rank) * 0.5
    W0 = np.random.randn(p * n, rank) * 0.5
    
    # Flatten
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def objective(x):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        # Compute the tensor
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    total += (val - 1.0) ** 2
        return total
    
    # Use BFGS or L-BFGS-B
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12})
        
        if result.fun < 1e-8:
            U = result.x[:n*m*rank].reshape(n*m, rank)
            V = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
            
            # Try to round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                return (U_r.tolist(), V_r.tolist(), W_r.tolist())
            
            # Try with integers
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if _verify_decomposition(U_i.tolist(), V_i.tolist(), W_i.tolist(), n, m, p):
                return (U_i.tolist(), V_i.tolist(), W_i.tolist())
            
            # Use the unrounded values if they're close enough
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
                return (U.tolist(), V.tolist(), W.tolist())
    except Exception:
        pass
    
    return None
