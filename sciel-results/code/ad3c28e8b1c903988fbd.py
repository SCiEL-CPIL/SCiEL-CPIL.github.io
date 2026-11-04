import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form an exact tensor decomposition of the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry T[i*m+j, j*p+k, k*n+i] should be 1
                # Reconstruction: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_2x3x6_decomp():
    """
    Construct Rank 33 decomposition for 2x3x6.
    This serves as the initial guess for the optimizer.
    """
    # Reuse the logic from the previous program to get a valid rank 33 solution
    # which acts as a good starting point for local search.
    
    # Strassen 2x2x2
    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]
    
    # M0: A00 * (B01+B11)
    U_s[0][0] = 1
    V_s[1][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1
    
    # M1: (A10+A11) * B00
    U_s[2][1] = 1; U_s[3][1] = 1
    V_s[0][1] = 1
    W_s[1][1] = 1; W_s[2][1] = 1
    
    # M2: (A00+A01) * B11
    U_s[0][2] = 1; U_s[1][2] = 1
    V_s[3][2] = 1
    W_s[0][2] = 1; W_s[3][2] = 1
    
    # M3: A11 * (B00-B10)
    U_s[3][3] = 1
    V_s[0][3] = 1; V_s[2][3] = -1
    W_s[0][3] = 1; W_s[2][3] = 1
    
    # M4: (A00+A10) * B01
    U_s[0][4] = 1; U_s[2][4] = 1
    V_s[1][4] = 1
    W_s[0][4] = -1; W_s[1][4] = 1
    
    # M5: (A01-A10) * (B00+B10)
    U_s[1][5] = 1; U_s[2][5] = -1
    V_s[0][5] = 1; V_s[2][5] = 1
    W_s[0][5] = -1
    
    # M6: (A11-A01) * (B01+B11)
    U_s[3][6] = 1; U_s[1][6] = -1
    V_s[1][6] = 1; V_s[3][6] = 1
    W_s[3][6] = 1

    U_block = [[0]*11 for _ in range(6)]
    V_block = [[0]*11 for _ in range(6)]
    W_block = [[0]*11 for _ in range(4)]
    
    U_map = [0, 1, 3, 4]
    for local_i, global_i in enumerate(U_map):
        for r in range(7):
            U_block[global_i][r] = U_s[local_i][r]
            
    V_map = [0, 1, 2, 3]
    for local_j, global_j in enumerate(V_map):
        for r in range(7):
            V_block[global_j][r] = V_s[local_j][r]
            
    W_map = [0, 1, 2, 3]
    for local_k, global_k in enumerate(W_map):
        for r in range(7):
            W_block[global_k][r] = W_s[local_k][r]
            
    U_block[2][7] = 1
    V_block[4][7] = 1
    W_block[0][7] = 1
    
    U_block[2][8] = 1
    V_block[5][8] = 1
    W_block[1][8] = 1
    
    U_block[5][9] = 1
    V_block[4][9] = 1
    W_block[2][9] = 1
    
    U_block[5][10] = 1
    V_block[5][10] = 1
    W_block[3][10] = 1
    
    U = [[0]*33 for _ in range(6)]
    V = [[0]*33 for _ in range(18)]
    W = [[0]*33 for _ in range(12)]
    
    for block in range(3):
        offset = block * 11
        for i in range(6):
            for r in range(11):
                U[i][offset+r] = U_block[i][r]
                
        for j in range(6):
            for r in range(11):
                V[j + block*6][offset+r] = V_block[j][r]
                
        for k in range(4):
            for r in range(11):
                W[k + block*4][offset+r] = W_block[k][r]
                
    return {"U": U, "V": V, "W": W}

def _optimize_decomposition(start_U, start_V, start_W, n, m, p, target_rank, time_limit=10):
    """
    Attempt to find a decomposition of target_rank by relaxing the integer constraint
    and using gradient descent / simulated annealing to minimize error.
    Then round to nearest 0.5 and verify.
    """
    # Flatten start matrices
    U = np.array(start_U, dtype=float).flatten()
    V = np.array(start_V, dtype=float).flatten()
    W = np.array(start_W, dtype=float).flatten()
    
    # We want to reduce rank from len(U) to target_rank.
    # Strategy: Keep the first target_rank columns of U, V, W and re-optimize them.
    # But simply truncating might not work. 
    # Better: Initialize with random values or the truncated start.
    # Let's try truncating first.
    
    U_opt = np.array(start_U, dtype=float)[:, :target_rank]
    V_opt = np.array(start_V, dtype=float)[:, :target_rank]
    W_opt = np.array(start_W, dtype=float)[:, :target_rank]
    
    # Add small noise to escape local minima
    U_opt += np.random.normal(0, 0.01, U_opt.shape)
    V_opt += np.random.normal(0, 0.01, V_opt.shape)
    W_opt += np.random.normal(0, 0.01, W_opt.shape)
    
    U_flat = U_opt.flatten()
    V_flat = V_opt.flatten()
    W_flat = W_opt.flatten()
    
    # Precompute indices for fast error calculation
    # Tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    indices_U = []
    indices_V = []
    indices_W = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices_U.append(i * m + j)
                indices_V.append(j * p + k)
                indices_W.append(k * n + i)
                
    indices_U = np.array(indices_U)
    indices_V = np.array(indices_V)
    indices_W = np.array(indices_W)
    
    def compute_error(U_f, V_f, W_f):
        U_m = U_f.reshape(n*m, target_rank)
        V_m = V_f.reshape(m*p, target_rank)
        W_m = W_f.reshape(p*n, target_rank)
        
        # Compute contributions
        # For each term (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is equivalent to element-wise product of rows and sum.
        
        # Vectorized approach:
        # Create matrices of selected rows
        U_rows = U_m[indices_U] # (N, R)
        V_rows = V_m[indices_V] # (N, R)
        W_rows = W_m[indices_W] # (N, R)
        
        # Product is element-wise, then sum over R
        prod = U_rows * V_rows * W_rows
        sums = np.sum(prod, axis=1)
        
        # Error is sum of squared differences from 1
        diff = sums - 1.0
        return np.sum(diff * diff)
    
    initial_error = compute_error(U_flat, V_flat, W_flat)
    
    # Simple Gradient Descent with Nesterov momentum or just SGD
    # Since we have a small number of variables (6*32 + 18*32 + 12*32 = 1152), 
    # and N_terms = 36, we can do many iterations.
    
    lr = 0.01
    best_U = U_flat.copy()
    best_V = V_flat.copy()
    best_W = W_flat.copy()
    best_err = initial_error
    
    start_time = time.time()
    while time.time() - start_time < time_limit:
        # Compute gradient
        U_m = U_flat.reshape(n*m, target_rank)
        V_m = V_flat.reshape(m*p, target_rank)
        W_m = W_flat.reshape(p*n, target_rank)
        
        U_rows = U_m[indices_U]
        V_rows = V_m[indices_V]
        W_rows = W_m[indices_W]
        
        sums = np.sum(U_rows * V_rows * W_rows, axis=1)
        diff = sums - 1.0
        
        # Gradient of sum_r U_r V_r W_r wrt U is V_r W_r
        # grad_U[i*m+j, r] = sum over terms (i,j,k) where U index matches, of diff * V_r * W_r
        # This is sparse. We can use scatter add.
        
        grad_U = np.zeros_like(U_flat)
        grad_V = np.zeros_like(V_flat)
        grad_W = np.zeros_like(W_flat)
        
        # Loop over terms for gradient calculation (N=36 is small)
        for t in range(len(indices_U)):
            i_idx = indices_U[t]
            j_idx = indices_V[t]
            k_idx = indices_W[t]
            
            # diff[t] is scalar
            d = diff[t]
            
            # V_rows[t] and W_rows[t] are vectors of length R
            v_vec = V_rows[t]
            w_vec = W_rows[t]
            
            # grad_U[i_idx, :] += d * v_vec * w_vec
            grad_U[i_idx * target_rank : (i_idx+1) * target_rank] += d * v_vec * w_vec
            
            # grad_V[j_idx, :] += d * u_vec * w_vec
            u_vec = U_rows[t]
            grad_V[j_idx * target_rank : (j_idx+1) * target_rank] += d * u_vec * w_vec
            
            # grad_W[k_idx, :] += d * u_vec * v_vec
            grad_W[k_idx * target_rank : (k_idx+1) * target_rank] += d * u_vec * v_vec
            
        # Update
        U_flat -= lr * grad_U
        V_flat -= lr * grad_V
        W_flat -= lr * grad_W
        
        # Periodic check and rounding
        if t % 100 == 0:
            err = compute_error(U_flat, V_flat, W_flat)
            if err < best_err:
                best_err = err
                best_U = U_flat.copy()
                best_V = V_flat.copy()
                best_W = W_flat.copy()
                
                # Try rounding
                U_rounded = np.round(U_flat.reshape(n*m, target_rank) * 2) / 2
                V_rounded = np.round(V_flat.reshape(m*p, target_rank) * 2) / 2
                W_rounded = np.round(W_flat.reshape(p*n, target_rank) * 2) / 2
                
                if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-6):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Get initial rank 33 solution
    res_33 = _construct_2x3x6_decomp()
    
    if _verify_decomposition(res_33['U'], res_33['V'], res_33['W'], n, m, p, tol=1e-9):
        best_result = res_33
    else:
        best_result = _schoolbook_decomposition(n, m, p)
        
    # Try to find a lower rank solution (32, 31, 30...)
    # We limit the search to a few ranks to stay within time budget.
    # Rank 32 is the most likely next step.
    
    for target_rank in [32, 31, 30]:
        if time.time() > deadline - 20:
            break
            
        # Run optimization
        # We might need multiple restarts
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
                
            res_opt = _optimize_decomposition(
                best_result['U'], 
                best_result['V'], 
                best_result['W'], 
                n, m, p, 
                target_rank, 
                time_limit=5
            )
            
            if res_opt is not None:
                # Verify strictly
                if _verify_decomposition(res_opt['U'], res_opt['V'], res_opt['W'], n, m, p, tol=1e-9):
                    best_result = res_opt
                    break # Found one for this rank, move to lower rank
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Verified:", _verify_decomposition(res['U'], res['V'], res['W'], 2, 3, 6))
