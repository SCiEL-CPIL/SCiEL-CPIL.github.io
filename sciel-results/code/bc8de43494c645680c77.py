"""Matrix multiplication 3x3x3 rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0

# EVOLVE-BLOCK-START

def _schoolbook():
    """Rank-27 schoolbook decomposition."""
    n, m, p = 3, 3, 3
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


def _verify(U, V, W, n=3, m=3, p=3):
    """Verify that U, V, W form a valid exact tensor decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _get_target_tensor(n=3, m=3, p=3):
    """Get the target matrix multiplication tensor as a 9x9x9 array.
    
    T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k), else 0.
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T


def _compute_error(U, V, W, T):
    """Compute the squared error between the decomposition and target tensor."""
    # U: (9, r), V: (9, r), W: (9, r)
    # T_reconstructed[a,b,c] = sum_c U[a,c]*V[b,c]*W[c_index,c]
    error = 0.0
    r = U.shape[1]
    for a in range(U.shape[0]):
        for b in range(V.shape[0]):
            for c_idx in range(W.shape[0]):
                s = 0.0
                for col in range(r):
                    s += U[a, col] * V[b, col] * W[c_idx, col]
                diff = s - T[a, b, c_idx]
                error += diff * diff
    return error


def _search_rank_r(r, T, deadline, max_iters=500):
    """Search for a rank-r decomposition using iterative optimization.
    
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 3, 3, 3
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    # Try multiple random initializations
    for init in range(20):
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + init)
        
        # Initialize with random values
        U = np.random.randn(dim_a, r) * 0.5
        V = np.random.randn(dim_b, r) * 0.5
        W = np.random.randn(dim_c, r) * 0.5
        
        # Iterative optimization: fix two, solve for the third
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U (fix V, W)
            # For each column c: U[:,c] should minimize sum_{a,b,c_idx} (T[a,b,c_idx] - V[b,c]*W[c_idx,c]*U[a,c])^2
            # This is a linear system in U[:,c]
            for c in range(r):
                # Build the linear system for U[:,c]
                # Error term involving U[:,c]: sum_{a,b,c_idx} (T[a,b,c_idx] - V[b,c]*W[c_idx,c]*U[a,c])^2
                # = sum_{a} (sum_{b,c_idx} V[b,c]*W[c_idx,c] * (T[a,b,c_idx] - V[b,c]*W[c_idx,c]*U[a,c]))^2
                # Actually, let's use gradient descent instead for simplicity
                
                # Compute gradient of error w.r.t. U[:,c]
                grad_U_c = np.zeros(dim_a)
                for a in range(dim_a):
                    for b in range(dim_b):
                        for c_idx in range(dim_c):
                            # d/dU[a,c] of (T[a,b,c_idx] - sum_col U[a,col]*V[b,col]*W[c_idx,col])^2
                            # = -2 * (T[a,b,c_idx] - sum_col U[a,col]*V[b,col]*W[c_idx,col]) * V[b,c] * W[c_idx,c]
                            s = 0.0
                            for col in range(r):
                                s += U[a, col] * V[b, col] * W[c_idx, col]
                            diff = T[a, b, c_idx] - s
                            grad_U_c[a] += 2 * diff * V[b, c] * W[c_idx, c]
                
                # Simple gradient step
                lr = 0.01
                U[:, c] -= lr * grad_U_c
            
            # Update V (fix U, W)
            for c in range(r):
                grad_V_c = np.zeros(dim_b)
                for b in range(dim_b):
                    for a in range(dim_a):
                        for c_idx in range(dim_c):
                            s = 0.0
                            for col in range(r):
                                s += U[a, col] * V[b, col] * W[c_idx, col]
                            diff = T[a, b, c_idx] - s
                            grad_V_c[b] += 2 * diff * U[a, c] * W[c_idx, c]
                V[:, c] -= lr * grad_V_c
            
            # Update W (fix U, V)
            for c in range(r):
                grad_W_c = np.zeros(dim_c)
                for c_idx in range(dim_c):
                    for a in range(dim_a):
                        for b in range(dim_b):
                            s = 0.0
                            for col in range(r):
                                s += U[a, col] * V[b, col] * W[c_idx, col]
                            diff = T[a, b, c_idx] - s
                            grad_W_c[c_idx] += 2 * diff * U[a, c] * V[b, c]
                W[:, c] -= lr * grad_W_c
            
            # Check error
            err = _compute_error(U, V, W, T)
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                # Found a solution, try to round to half-integers
                return _try_round(best_U, best_V, best_W, T)
        
        if best_error < 1e-10:
            return _try_round(best_U, best_V, best_W, T)
    
    return None


def _try_round(U, V, W, T, tol=1e-8):
    """Try to round U, V, W to half-integers and verify."""
    # Try rounding each matrix independently
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Convert to lists
    U_list = U_rounded.tolist()
    V_list = V_rounded.tolist()
    W_list = W_rounded.tolist()
    
    if _verify(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try different rounding strategies
    for scale in [1, 2, 4]:
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        U_l = U_r.tolist()
        V_l = V_r.tolist()
        W_l = W_r.tolist()
        if _verify(U_l, V_l, W_l):
            return {"U": U_l, "V": V_l, "W": W_l}
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    # Start with the guaranteed valid rank-27 schoolbook
    best = _schoolbook()
    best_rank = 27
    
    # Get the target tensor
    T = _get_target_tensor()
    
    # Try to find lower-rank decompositions
    for r in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        
        result = _search_rank_r(r, T, deadline)
        if result is not None:
            best = result
            best_rank = r
            break  # Found the lowest rank we're searching for
    
    return best

# EVOLVE-BLOCK-END
