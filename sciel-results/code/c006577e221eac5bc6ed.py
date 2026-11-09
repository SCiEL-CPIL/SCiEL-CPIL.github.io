"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # T should be all 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _als_decompose(n, m, p, target_rank, n_trials=10, seed=42):
    """Try to find a rank-target_rank decomposition using ALS."""
    rng = np.random.RandomState(seed)
    
    for trial in range(n_trials):
        U = rng.randn(n*m, target_rank)
        V = rng.randn(m*p, target_rank)
        W = rng.randn(p*n, target_rank)
        
        # Normalize initial values
        U /= np.linalg.norm(U) + 1e-10
        V /= np.linalg.norm(V) + 1e-10
        W /= np.linalg.norm(W) + 1e-10
        
        # ALS iterations
        for it in range(50):
            # Update U: for each row i*m+j, solve least squares
            for idx_u in range(n*m):
                i = idx_u // m
                j = idx_u % m
                # T[i,j,k] = sum_c U[idx_u,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is nonlinear. Let's use a different approach.
                # Fix V and W, solve for U[idx_u, :]
                # For each k: sum_c U[idx_u,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                # Let a_c = V[j*p+k,c] * W[k*n+i,c]
                # Then sum_c U[idx_u,c] * a_c = 1 for all k
                # This is an underdetermined system (target_rank equations for target_rank unknowns)
                # Actually, we have p equations (one for each k) and target_rank unknowns
                # If target_rank > p, it's underdetermined; if target_rank <= p, we can solve
                
                # Build the system: A[idx_u] * U[idx_u, :] = b where A is p x target_rank
                A = np.zeros((p, target_rank))
                for k in range(p):
                    for c in range(target_rank):
                        A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                b = np.ones(p)
                
                # Solve A * U[idx_u, :] = b using least squares
                U[idx_u, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Similarly update V and W
            for idx_v in range(m*p):
                j = idx_v // p
                k = idx_v % p
                A = np.zeros((n, target_rank))
                for i in range(n):
                    for c in range(target_rank):
                        A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                b = np.ones(n)
                V[idx_v, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            
            for idx_w in range(p*n):
                k = idx_w // n
                i = idx_w % n
                A = np.zeros((m, target_rank))
                for j in range(m):
                    for c in range(target_rank):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                b = np.ones(m)
                W[idx_w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        result = _als_decompose(n, m, p, target_rank, n_trials=5, seed=target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
