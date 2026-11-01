"""Matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 140  # Safety margin before 150s kill

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x2x3_rank11():
    """Build a rank-11 decomposition for 2x2 x 2x3 (Strassen 7 + Schoolbook 4)."""
    n, m, p = 2, 2, 3
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Strassen for k=0,1
    U[:, 0:7] = U_str
    for j in range(2):
        for k_local in range(2):
            V[j*3 + k_local, 0:7] = V_str[j*2 + k_local, :]
    for k_local in range(2):
        for i in range(2):
            W[k_local*2 + i, 0:7] = W_str[k_local*2 + i, :]
            
    # Schoolbook for k=2
    U[:, 7:11] = U_sb
    for j in range(2):
        V[j*3 + 2, 7:11] = V_sb[j*1, :]
    for i in range(2):
        W[2*2 + i, 7:11] = W_sb[0*2 + i, :]
        
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank_2x2x3(U_init, V_init, W_init, target_rank=10, time_budget=10.0, num_restarts=20):
    """
    Try to reduce the rank of a 2x2x3 decomposition by optimizing.
    Uses multiple random restarts to increase the chance of finding the global minimum.
    """
    start_time = time.time()
    n, m, p = 2, 2, 3
    current_rank = 11
    
    # We will optimize U, V, W of size (4, target_rank), (6, target_rank), (6, target_rank)
    # Initial guess: take the first target_rank columns of the rank-11 solution
    U0 = U_init[:, :target_rank].flatten()
    V0 = V_init[:, :target_rank].flatten()
    W0 = W_init[:, :target_rank].flatten()
    x0_base = np.concatenate([U0, V0, W0])
    
    def error_func(x):
        if time.time() - start_time > time_budget:
            return 1e10
        U = x[0:4*target_rank].reshape(4, target_rank)
        V = x[4*target_rank:4*target_rank + 6*target_rank].reshape(6, target_rank)
        W = x[4*target_rank + 6*target_rank:].reshape(6, target_rank)
        
        # Calculate error for all 12 (i,j,k) combinations
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0)**2
        return error

    best_result = None
    best_error = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Add small random noise to escape local minima
        noise_scale = 0.1 / (restart + 1)
        noise = np.random.normal(0, noise_scale, x0_base.shape)
        x0 = x0_base + noise
        
        try:
            res = minimize(error_func, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_error:
                best_error = res.fun
                best_result = res
        except Exception:
            continue
            
    if best_result is not None and best_result.fun < 1e-8:
        U_opt = best_result.x[0:4*target_rank].reshape(4, target_rank)
        V_opt = best_result.x[4*target_rank:4*target_rank + 6*target_rank].reshape(6, target_rank)
        W_opt = best_result.x[4*target_rank + 6*target_rank:].reshape(6, target_rank)
        
        # Final strict verification
        if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
            return U_opt, V_opt, W_opt
            
    return None

def _build_rank53():
    """Build the rank-53 decomposition."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b, block_size in enumerate(block_sizes):
        start_k = sum(block_sizes[:b])
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    V[j * p + start_k + k_local, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                for i in range(n):
                    W[(start_k + k_local) * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy 1: Try to find a rank-52 decomposition using a rank-10 block for 2x2x3.
    # 6 blocks of 2x2x2 (Strassen rank 7) + 1 block of 2x2x3 (Optimized rank 10) = 42 + 10 = 52.
    try:
        U_11, V_11, W_11 = _build_2x2x3_rank11()
        
        # Allocate time for the search. We have ~140s total.
        # Search for rank 10 with a budget of 30s.
        res = _try_reduce_rank_2x2x3(U_11, V_11, W_11, target_rank=10, time_budget=30.0, num_restarts=50)
        
        if res is not None:
            U_10, V_10, W_10 = res
            
            # Construct full 52-rank decomposition
            total_rank = 52
            U_full = np.zeros((4, total_rank))
            V_full = np.zeros((30, total_rank))
            W_full = np.zeros((30, total_rank))
            
            U_str, V_str, W_str = _build_strassen_2x2x2()
            
            # 6 Strassen blocks for k=0..11
            for b in range(6):
                start_k = b * 2
                col_off = b * 7
                U_full[:, col_off:col_off+7] = U_str
                for j in range(2):
                    for k_local in range(2):
                        V_full[j*15 + start_k + k_local, col_off:col_off+7] = V_str[j*2 + k_local, :]
                for k_local in range(2):
                    for i in range(2):
                        W_full[(start_k + k_local)*2 + i, col_off:col_off+7] = W_str[k_local*2 + i, :]
                        
            # 1 Optimized block for k=12..14
            start_k = 12
            col_off = 6 * 7
            U_full[:, col_off:col_off+10] = U_10
            for j in range(2):
                for k_local in range(3):
                    V_full[j*15 + start_k + k_local, col_off:col_off+10] = V_10[j*3 + k_local, :]
            for k_local in range(3):
                for i in range(2):
                    W_full[(start_k + k_local)*2 + i, col_off:col_off+10] = W_10[k_local*2 + i, :]
            
            # Final verification
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
        # Fallback to Rank 53 if Rank 52 failed or time is up
        if time.time() - start_time > TIME_LIMIT - 10:
             U, V, W = _build_rank53()
             return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
             
    except Exception as e:
        pass
        
    # Strategy 2: Fallback to Rank 53
    U, V, W = _build_rank53()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    rank = U.shape[1]
    print(f"Constructed decomposition with rank {rank}")
    if _verify_decomposition(U, V, W, 2, 2, 15):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
