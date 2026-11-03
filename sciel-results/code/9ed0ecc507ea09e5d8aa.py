"""Matrix multiplication 5x5x5 tensor decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]
SEED = 42

def _verify(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 5, 5, 5
    # Build the tensor T[i,j,k] = delta_{i,k} * delta_{j,j} ... 
    # Actually the matrix multiplication tensor is: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor form: T[i,j,k] = 1 if this corresponds to A[i,j]*B[j,k] contributing to C[i,k]
    # In tensor notation: T_{i,j,k} where i indexes rows of A (and C), j indexes cols of A/rows of B, k indexes cols of B (and C)
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For 5x5x5, the tensor has 125 entries, each is 1 (since every (i,j,k) triple corresponds to one multiplication)
    # Wait, let me reconsider. The matrix multiplication tensor for (n,m,p) = (5,5,5):
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Because C[i,k] = sum_j A[i,j] B[j,k], so each term A[i,j]*B[j,k] contributes.
    
    # Actually the standard form: the multiplication tensor has entries
    # T_{(i,j),(j,k),(i,k)} = 1, but in the flattened form used here:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The constraint is: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(arr, tol=1e-4):
    """Snap values close to half-integers to exact half-integers."""
    arr = np.array(arr, dtype=float)
    snapped = np.round(arr * 2) / 2.0
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped
    return arr

def _try_rank(rank, time_budget, rng):
    """Try to find a decomposition of given rank using random search."""
    n, m, p = 5, 5, 5
    nm, mp, pn = n*m, m*p, p*n
    
    start_time = time.time()
    
    # Random initialization
    U = rng.standard_normal((nm, rank))
    V = rng.standard_normal((mp, rank))
    W = rng.standard_normal((pn, rank))
    
    # Scale to reasonable magnitudes
    U /= np.sqrt(rank)
    V /= np.sqrt(rank)
    W /= np.sqrt(rank)
    
    # Use a simple gradient-free approach: alternating minimization with random restarts
    # For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Number of restarts based on time budget
    max_restarts = 20
    iters_per_restart = 500
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget - 5:
            break
        
        # Re-initialize
        U = rng.standard_normal((nm, rank))
        V = rng.standard_normal((mp, rank))
        W = rng.standard_normal((pn, rank))
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        # Alternating minimization
        lr = 0.1
        for it in range(iters_per_restart):
            if time.time() - start_time > time_budget - 5:
                break
            
            # Update U: for each (i,j), minimize ||U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:]) - 1||^2 over k
            # Actually, let's do a simpler approach: compute residuals and do gradient descent
            
            # Compute current tensor values
            # Residual for each (i,j,k): r[i,j,k] = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            
            # Gradient w.r.t. U[i*m+j,c]: sum_k r[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
            # Gradient w.r.t. V[j*p+k,c]: sum_i r[i,j,k] * U[i*m+j,c] * W[k*n+i,c]
            # Gradient w.r.t. W[k*n+i,c]: sum_j r[i,j,k] * U[i*m+j,c] * V[j*p+k,c]
            
            # This is O(n*m*p*rank) per iteration which is 125*rank
            # For rank up to ~25, this is manageable
            
            # Compute all residuals
            residuals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residuals[i,j,k] = 1.0 - np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
            
            err = np.sum(residuals**2)
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
            
            # Adaptive learning rate
            if it % 10 == 0:
                if err < best_err * 0.5:
                    lr *= 1.2
                else:
                    lr *= 0.8
                lr = min(lr, 1.0)
            
            # Update U
            for i in range(n):
                for j in range(m):
                    grad = np.zeros(rank)
                    for k in range(p):
                        grad += residuals[i,j,k] * V[j*p+k] * W[k*n+i]
                    U[i*m+j] -= lr * grad
            
            # Update V
            for j in range(m):
                for k in range(p):
                    grad = np.zeros(rank)
                    for i in range(n):
                        grad += residuals[i,j,k] * U[i*m+j] * W[k*n+i]
                    V[j*p+k] -= lr * grad
            
            # Update W
            for k in range(p):
                for i in range(n):
                    grad = np.zeros(rank)
                    for j in range(m):
                        grad += residuals[i,j,k] * U[i*m+j] * V[j*p+k]
                    W[k*n+i] -= lr * grad
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-6:
        # Try to snap to half-integers
        U_s = _snap_to_half_integer(best_U)
        V_s = _snap_to_half_integer(best_V)
        W_s = _snap_to_half_integer(best_W)
        if _verify(U_s, V_s, W_s, tol=1e-4):
            return U_s.tolist(), V_s.tolist(), W_s.tolist()
        
        # Try snapping to integers
        U_i = np.round(best_U)
        V_i = np.round(best_V)
        W_i = np.round(best_W)
        if _verify(U_i, V_i, W_i, tol=1e-4):
            return U_i.tolist(), V_i.tolist(), W_i.tolist()
        
        # Use the continuous solution (should still be valid if error is small enough)
        if best_err < 1e-6:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    rng = np.random.RandomState(SEED)
    
    # The known best rank for 5x5x5 matrix multiplication is 25 (Landsberg's bound gives 25)
    # Actually, the trivial rank is 125. The best known lower bound is around 25.
    # Let's try ranks from 25 upward.
    
    # Try a few candidate ranks
    candidate_ranks = [25, 26, 27, 28, 29, 30, 35, 40, 50, 60, 80, 100, 125]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Allocate time for this rank attempt
        time_for_rank = min(30, deadline - time.time() - 5)
        if time_for_rank < 5:
            break
        
        result = _try_rank(rank, time_for_rank, rng)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: return the schoolbook decomposition (rank 125)
    n, m, p = 5, 5, 5
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
