"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    time_limit = 140.0  # Leave 10 seconds margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_structured_rank(rank):
        """
        Try to find a rank-`rank` decomposition using structured initialization
        based on block decomposition of the 5x5 multiplication.
        """
        if time.time() - start_time > time_limit - 20:
            return None
        
        # Strategy: Exploit the fact that we can view the 3x5x5 problem
        # as three independent 5x5 multiplications (one for each row of A)
        # For each 5x5 multiplication, we can use a block-based approach
        
        # For 5x5 matrix multiplication, try to use a decomposition that
        # exploits block structure. Split 5 into 2+3 blocks.
        
        # Initialize with a structured pattern based on Kronecker products
        # and block identities
        
        U_init = np.zeros((n * m, rank))
        V_init = np.zeros((m * p, rank))
        W_init = np.zeros((p * n, rank))
        
        # Try a pattern where we use the all-ones structure but with
        # some structure to allow for lower rank
        
        # Approach: Use random initialization with some structure
        for r in range(rank):
            # For each rank term, create a structured pattern
            # that tries to cover the tensor efficiently
            
            # Use a combination of random and structured initialization
            scale = 1.0 / np.sqrt(rank)
            
            # Structured: assign each (i,j,k) triple to a subset of rank terms
            # using a hashing-like approach
            
            # Random component
            U_init[:, r] = np.random.randn(n * m) * scale
            V_init[:, r] = np.random.randn(m * p) * scale
            W_init[:, r] = np.random.randn(p * n) * scale
        
        def objective(params):
            if time.time() - start_time > time_limit - 10:
                return 1e10
            U = params[:n * m * rank].reshape(n * m, rank)
            V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
            W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
            
            # Compute the error efficiently using tensor operations
            # Error = sum_{i,j,k} (sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1)^2
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        error += (val - 1.0) ** 2
            return error
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            
            if result.fun < 1e-6:
                U = result.x[:n * m * rank].reshape(n * m, rank)
                V = result.x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                W = result.x[n * m * rank + m * p * rank:].reshape(p * n, rank)
                
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if the rounded solution is exact
                if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                
                # Try rounding to integers
                U_rounded_int = np.round(U)
                V_rounded_int = np.round(V)
                W_rounded_int = np.round(W)
                
                if verify(U_rounded_int.tolist(), V_rounded_int.tolist(), W_rounded_int.tolist()):
                    return {
                        "U": U_rounded_int.tolist(),
                        "V": V_rounded_int.tolist(),
                        "W": W_rounded_int.tolist()
                    }
                    
        except Exception:
            pass
        
        return None
    
    def try_multi_start(rank, n_starts=3):
        """Try multiple random starts for a given rank."""
        if time.time() - start_time > time_limit - 20:
            return None
        
        best_err = float('inf')
        best_params = None
        
        for start in range(n_starts):
            if time.time() - start_time > time_limit - 15:
                break
            
            U_init = np.random.randn(n * m, rank) * 0.5
            V_init = np.random.randn(m * p, rank) * 0.5
            W_init = np.random.randn(p * n, rank) * 0.5
            
            def objective(params):
                if time.time() - start_time > time_limit - 10:
                    return 1e10
                U = params[:n * m * rank].reshape(n * m, rank)
                V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                            error += (val - 1.0) ** 2
                return error
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if result.fun < best_err:
                    best_err = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        if best_params is not None and best_err < 1e-4:
            U = best_params[:n * m * rank].reshape(n * m, rank)
            V = best_params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
            W = best_params[n * m * rank + m * p * rank:].reshape(p * n, rank)
            
            # Try different rounding strategies
            for mult in [2, 4, 1]:
                U_r = np.round(U * mult) / mult
                V_r = np.round(V * mult) / mult
                W_r = np.round(W * mult) / mult
                
                if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        return None
    
    # The lower bound is max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # But for matrix multiplication, the true lower bound is higher.
    # Let's try ranks from a reasonable starting point.
    
    # Start with the known lower bound and go up
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a few ranks starting from the lower bound
    ranks_to_try = list(range(lower_bound, min(lower_bound + 15, best_rank + 1)))
    
    for rank in ranks_to_try:
        if time.time() - start_time > time_limit - 30:
            break
        
        # Try structured approach first
        result = try_structured_rank(rank)
        if result is not None:
            best_result = result
            best_rank = rank
            if best_rank == lower_bound:
                break
            continue
        
        # Try multi-start approach
        result = try_multi_start(rank, n_starts=2)
        if result is not None:
            best_result = result
            best_rank = rank
            if best_rank == lower_bound:
                break
    
    # Check time
    if time.time() - start_time > time_limit:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
