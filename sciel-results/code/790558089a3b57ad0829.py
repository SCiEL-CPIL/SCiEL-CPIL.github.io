import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matmul tensor with half-integer entries."""
    # Check entries are half-integers (within floating point tolerance)
    for M in (U, V, W):
        M2 = np.array(M, dtype=np.float64) * 2.0
        if not np.allclose(M2, np.round(M2), atol=1e-7):
            return False
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(M):
    """Snap matrix entries to nearest 0.5."""
    return np.round(np.array(M, dtype=np.float64) * 2.0) / 2.0

def _construct_schoolbook(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank_with_heuristics(R, n, m, p, time_limit, seed_base=0):
    """
    Try to find a rank-R decomposition using structured initialization and refinement.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    max_attempts = 200
    
    # Known lower bounds for matrix multiplication rank
    # For (3,4,8): min rank is at least max(n*m, m*p, n*p) = max(12, 32, 24) = 32
    # But there are stronger bounds. Let's just try from the theoretical minimum.
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        seed = seed_base + attempt * 137
        np.random.seed(seed)
        
        # Strategy 1: Random initialization with sparsity
        # Strategy 2: Structured initialization based on known decompositions
        
        # Initialize with small random values
        scale = 1.0
        U = np.random.randn(n*m, R) * scale
        V = np.random.randn(m*p, R) * scale
        W = np.random.randn(p*n, R) * scale
        
        # Add some structure: try to make columns sparse
        # Zero out a fraction of entries
        sparsity = 0.5
        U[np.random.rand(n*m, R) < sparsity] = 0
        V[np.random.rand(m*p, R) < sparsity] = 0
        W[np.random.rand(p*n, R) < sparsity] = 0
        
        # Ensure no column is entirely zero
        for r in range(R):
            if np.all(U[:, r] == 0):
                U[np.random.randint(n*m), r] = 1.0
            if np.all(V[:, r] == 0):
                V[np.random.randint(m*p), r] = 1.0
            if np.all(W[:, r] == 0):
                W[np.random.randint(p*n), r] = 1.0
        
        # Iterative refinement: alternating minimization
        # For each (i,j,k), we want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a 3-way tensor completion problem.
        
        # Simple gradient descent on the residual
        lr = 0.01
        for step in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Compute residuals
            residuals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residuals[i,j,k] = 1.0 - np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
            
            if np.max(np.abs(residuals)) < 1e-10:
                break
            
            # Gradient for U
            grad_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_U[i*m+j] += -residuals[i,j,k] * V[j*p+k] * W[k*n+i]
            
            # Gradient for V
            grad_V = np.zeros_like(V)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_V[j*p+k] += -residuals[i,j,k] * U[i*m+j] * W[k*n+i]
            
            # Gradient for W
            grad_W = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_W[k*n+i] += -residuals[i,j,k] * U[i*m+j] * V[j*p+k]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Snap to half-integers
        U_snap = _snap_to_half_integer(U)
        V_snap = _snap_to_half_integer(V)
        W_snap = _snap_to_half_integer(W)
        
        if _verify(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
        
        # Also check if the non-snapped version is close enough that snapping might work with perturbation
        # Try small perturbations around the snapped values
        for pert_attempt in range(5):
            if time.time() - start_time > time_limit:
                break
            U_pert = U_snap + np.random.randn(n*m, R) * 0.1
            V_pert = V_snap + np.random.randn(m*p, R) * 0.1
            W_pert = W_snap + np.random.randn(p*n, R) * 0.1
            U_pert_snap = _snap_to_half_integer(U_pert)
            V_pert_snap = _snap_to_half_integer(V_pert)
            W_pert_snap = _snap_to_half_integer(W_pert)
            if _verify(U_pert_snap, V_pert_snap, W_pert_snap, n, m, p):
                return U_pert_snap, V_pert_snap, W_pert_snap
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 10s safety margin before 150s limit
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _construct_schoolbook(n, m, p)
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = n * m * p
    
    # Search for lower rank
    # Theoretical lower bound: max(n*m, m*p, n*p) = max(12, 32, 24) = 32
    # But we can try even lower
    min_rank = max(n*m, m*p, n*p)
    
    for R in range(min_rank, best_rank + 1):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        if time_left < 1:
            break
            
        # Give more time to lower ranks (harder problems)
        time_per_rank = min(5.0, time_left / max(1, best_rank - R + 1))
        
        result = _try_rank_with_heuristics(R, n, m, p, time_limit=time_per_rank, seed_base=R)
        
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_result = {
                "U": U_sol.tolist(),
                "V": V_sol.tolist(),
                "W": W_sol.tolist()
            }
            best_rank = R
            # Continue searching for even lower rank? No, we're going up from min_rank.
            # So the first one we find is the minimal.
            return best_result
    
    return best_result
