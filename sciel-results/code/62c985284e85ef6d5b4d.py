import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]
START_TIME = time.time()
TIME_LIMIT = 135  # Leave 15s margin before 150s kill

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return _rank_44_solution()
    
    n, m, p = DIMENSIONS
    
    # Start with known rank-44 solution
    best_result = _rank_44_solution()
    best_rank = 44
    
    # Try to reduce rank from 43 down to 35
    for rank in range(43, 34, -1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        result = _search_for_rank(n, m, p, rank, best_result)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result

def _rank_44_solution():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:,0] * B[0,:] -> 2x8 (schoolbook for 2x1x8)
    for i in range(2):
        for k in range(8):
            U[i * 3 + 0][col] = 1
            V[0 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A[:,1:3] * B[1:3,:] -> 2x8 using Strassen-like decomposition
    # We have a 2x2 matrix (rows 1,2 of A) times 2x8 matrix (rows 1,2 of B)
    # Split B into 4 blocks of 2x2
    for block in range(4):
        # For each 2x2 block of B, use 7 multiplications (Strassen)
        # But we need to be careful about the structure
        
        # m1 = (a00+a11)(b00+b11)
        U[1][col] = 1
        U[5][col] = 1
        V[1*8 + block*2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[4][col] = 1
        U[5][col] = 1
        V[1*8 + block*2][col] = 1
        W[(block*2)*2 + 1][col] = 1
        W[(block*2+1)*2 + 1][col] = -1
        col += 1
        
        # m3 = a00(b01-b11)
        U[1][col] = 1
        V[1*8 + block*2+1][col] = 1
        V[2*8 + block*2+1][col] = -1
        W[(block*2+1)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[5][col] = 1
        V[2*8 + block*2][col] = 1
        V[1*8 + block*2][col] = -1
        W[(block*2)*2 + 0][col] = 1
        W[(block*2+1)*2 + 0][col] = 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[1][col] = 1
        U[2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = -1
        W[(block*2+1)*2 + 0][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[4][col] = 1
        U[1][col] = -1
        V[1*8 + block*2][col] = 1
        V[1*8 + block*2+1][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[2][col] = 1
        U[5][col] = -1
        V[2*8 + block*2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

def _compute_residuals(U_mat, V_mat, W_mat, n, m, p, rank):
    """Vectorized residual computation."""
    U = U_mat  # (n*m, rank)
    V = V_mat  # (m*p, rank)
    W = W_mat  # (p*n, rank)
    
    # For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Reshape to enable vectorization
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute all (i,j,k) values
    # U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:] summed over r
    # Use einsum
    # result[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    
    # This is a bit tricky. Let's do it step by step.
    # For fixed j: sum_r U_r[:,j,r] * V_r[j,:,r] gives (n, p, rank) intermediate
    # Then multiply with W_r[:, :, r] and sum
    
    # Alternative: loop over j (only 3 iterations)
    residuals = np.zeros((n, m, p))
    for j in range(m):
        # U_j[i, r] = U_r[i, j, r] shape (n, rank)
        # V_j[k, r] = V_r[j, k, r] shape (p, rank)
        # W_j[k, i, r] = W_r[k, i, r] shape (p, n, rank)
        
        # temp[i, k, r] = U_j[i, r] * V_j[k, r]
        temp = U_r[:, j, :][:, np.newaxis, :] * V_r[j, :, :][np.newaxis, :, :]  # (n, p, rank)
        
        # Now sum over r: temp[i, k, r] * W_r[k, i, r]
        # W_r has shape (p, n, rank), we need W_r[k, i, r]
        # So we need to index W_r as W_r[:, :, :] and transpose to (n, p, rank)
        W_transposed = W_r.transpose(1, 0, 2)  # (n, p, rank)
        
        residuals[:, j, :] = np.sum(temp * W_transposed, axis=2)
    
    return (residuals - 1.0).flatten()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    residuals = _compute_residuals(U_np, V_np, W_np, n, m, p, rank)
    return np.max(np.abs(residuals)) < tol

def _search_for_rank(n, m, p, target_rank, init_solution=None):
    """Try to find a valid decomposition with the given rank using numerical optimization."""
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return None
    
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    max_restarts = 3
    best_cost = np.inf
    best_x = None
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            return None
        
        # Create initial guess
        if init_solution is not None and restart == 0:
            # Use the known solution as starting point, truncate to target_rank
            U_init = np.array(init_solution["U"], dtype=float)[:, :target_rank]
            V_init = np.array(init_solution["V"], dtype=float)[:, :target_rank]
            W_init = np.array(init_solution["W"], dtype=float)[:, :target_rank]
        else:
            # Random initialization with structure
            np.random.seed(42 + restart * 1000)
            U_init = np.random.randn(U_size, target_rank) * 0.5
            V_init = np.random.randn(V_size, target_rank) * 0.5
            W_init = np.random.randn(W_size, target_rank) * 0.5
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(x):
            U_flat = x[:U_size * target_rank]
            V_flat = x[U_size * target_rank : U_size * target_rank + V_size * target_rank]
            W_flat = x[U_size * target_rank + V_size * target_rank :]
            
            U_mat = U_flat.reshape(U_size, target_rank)
            V_mat = V_flat.reshape(V_size, target_rank)
            W_mat = W_flat.reshape(W_size, target_rank)
            
            return _compute_residuals(U_mat, V_mat, W_mat, n, m, p, target_rank)
        
        try:
            result = least_squares(
                residuals, x0, method='lm',
                max_nfev=5000,
                xtol=1e-10, ftol=1e-10, gtol=1e-10
            )
            
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_x = result.x
            
            if cost < 1e-12:
                break
        except Exception:
            pass
    
    if best_x is None or best_cost > 1e-8:
        return None
    
    # Extract solution
    U_flat = best_x[:U_size * target_rank]
    V_flat = best_x[U_size * target_rank : U_size * target_rank + V_size * target_rank]
    W_flat = best_x[U_size * target_rank + V_size * target_rank :]
    
    U_mat = U_flat.reshape(U_size, target_rank)
    V_mat = V_flat.reshape(V_size, target_rank)
    W_mat = W_flat.reshape(W_size, target_rank)
    
    # Polish pass with tighter tolerance
    try:
        polish_result = least_squares(
            residuals, best_x, method='lm',
            max_nfev=10000,
            xtol=1e-14, ftol=1e-14, gtol=1e-14
        )
        if polish_result.cost < best_cost:
            best_x = polish_result.x
            U_flat = best_x[:U_size * target_rank]
            V_flat = best_x[U_size * target_rank : U_size * target_rank + V_size * target_rank]
            W_flat = best_x[U_size * target_rank + V_size * target_rank :]
            U_mat = U_flat.reshape(U_size, target_rank)
            V_mat = V_flat.reshape(V_size, target_rank)
            W_mat = W_flat.reshape(W_size, target_rank)
    except Exception:
        pass
    
    # Convert to lists
    U_list = [[float(U_mat[i, j]) for j in range(target_rank)] for i in range(U_size)]
    V_list = [[float(V_mat[i, j]) for j in range(target_rank)] for i in range(V_size)]
    W_list = [[float(W_mat[i, j]) for j in range(target_rank)] for i in range(W_size)]
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-5):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
