"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try ranks from schoolbook-1 down to 18
    for rank in range(schoolbook_rank - 1, 17, -1):
        if time.time() > deadline:
            break
        
        try:
            result = _try_rank(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                continue
        except Exception:
            continue
    
    return best_result

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition with the given rank using structured search."""
    np.random.seed(42)
    
    # Strategy: Try multiple random initializations with alternating least squares
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization with different scales
        scale_u = np.random.choice([0.5, 1.0, 2.0])
        scale_v = np.random.choice([0.5, 1.0, 2.0])
        scale_w = np.random.choice([0.5, 1.0, 2.0])
        
        U = np.random.randn(n * m, rank) * scale_u
        V = np.random.randn(m * p, rank) * scale_v
        W = np.random.randn(p * n, rank) * scale_w
        
        # Alternating minimization
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve linear system
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # This is: U[i*m+j,:] @ diag(V[j*p+k,:]*W[k*n+i,:]) = 1 for each k
                    # Build matrix A where A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve A.T @ U[i*m+j,:] = ones(p)
                    try:
                        U_new[i * m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V_new[j * p + k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W_new[k * n + i, :] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            W = W_new
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-8:
            # Try to snap to half-integers
            for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                U_snap = np.round(U * denom) / denom
                V_snap = np.round(V * denom) / denom
                W_snap = np.round(W * denom) / denom
                
                # Verify exactness
                max_err_snap = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_snap[i * m + j, :], V_snap[j * p + k, :] * W_snap[k * n + i, :])
                            max_err_snap = max(max_err_snap, abs(val - 1.0))
                
                if max_err_snap < 1e-9:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
    
    return None

# EVOLVE-BLOCK-END
