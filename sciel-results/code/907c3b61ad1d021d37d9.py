import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strategy: The 2x4 x 4x8 multiplication can be decomposed into
    # 4 independent 2x4 x 4x2 sub-problems (one per column-pair of C).
    # Each sub-problem has schoolbook rank 16, block-Strassen rank 14.
    # We attempt to find a rank-13 decomposition for each sub-problem
    # via numerical optimization, giving total rank 4*13 = 52.
    # If that fails, fall back to rank 56 (block-Strassen).
    
    deadline = start_time + 140  # safety margin
    
    def get_strassen_242():
        """Get the rank-14 block-Strassen decomposition for 2x4 x 4x2."""
        # A is 2x4: indices (i,j) -> i*4+j, i in [0,1], j in [0,3]
        # B is 4x2: indices (j,k) -> j*2+k, j in [0,3], k in [0,1]
        # C is 2x2: W convention W[k*n+i, r], n=2, so index = k*2+i
        
        U = np.zeros((8, 14))
        V = np.zeros((8, 14))
        W = np.zeros((4, 14))
        
        # Strassen 2x2 x 2x2 local matrices
        U_local = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=float)
        
        V_local = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ], dtype=float)
        
        W_local = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=float)
        
        # Block 1: A1 = A[:, 0:2], B1 = B[0:2, :], C contribution
        # A1 local (i,j) -> global i*4 + j
        a1_map = [0, 1, 4, 5]
        b1_map = [0, 1, 2, 3]
        c1_map = [0, 1, 2, 3]  # k*2+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        for l in range(7):
            r = l
            for la in range(4):
                val = U_local[la, l]
                if val != 0:
                    U[a1_map[la], r] += val
            for lb in range(4):
                val = V_local[lb, l]
                if val != 0:
                    V[b1_map[lb], r] += val
            for lc in range(4):
                val = W_local[lc, l]
                if val != 0:
                    W[c1_map[lc], r] += val
        
        # Block 2: A2 = A[:, 2:4], B2 = B[2:4, :], C contribution
        a2_map = [2, 3, 6, 7]
        b2_map = [4, 5, 6, 7]
        c2_map = [0, 1, 2, 3]
        
        for l in range(7):
            r = 7 + l
            for la in range(4):
                val = U_local[la, l]
                if val != 0:
                    U[a2_map[la], r] += val
            for lb in range(4):
                val = V_local[lb, l]
                if val != 0:
                    V[b2_map[lb], r] += val
            for lc in range(4):
                val = W_local[lc, l]
                if val != 0:
                    W[c2_map[lc], r] += val
        
        return U, V, W
    
    def verify_242(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact for 2x4 x 4x2."""
        nn, mm, pp = 2, 4, 2
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    actual = np.sum(U[i*mm+j, :] * V[j*pp+k, :] * W[k*nn+i, :])
                    if abs(actual - 1.0) > tol:
                        return False
        return True
    
    def optimize_242(rank, max_iter=5000, lr=0.01, seed=42):
        """Try to find a rank-r decomposition of the 2x4 x 4x2 tensor."""
        nn, mm, pp = 2, 4, 2
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(nn*mm, rank) * 0.5
        V = rng.randn(mm*pp, rank) * 0.5
        W = rng.randn(pp*nn, rank) * 0.5
        
        # Target tensor: all ones
        # T[i,j,k] = 1 for all i,j,k
        # We want sum_r U[i*mm+j, r] * V[j*pp+k, r] * W[k*nn+i, r] = 1
        
        best_err = float('inf')
        best_params = (U.copy(), V.copy(), W.copy())
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute error: for each (i,j,k), err = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1
            # Gradient w.r.t. U[i,j,r]: V[j,k,r]*W[k,i,r] summed over k... 
            # Actually: d/dU[i,j,r] sum_k (sum_r' U[i,j,r']*V[j,k,r']*W[k,i,r'] - 1)^2
            # = 2 * sum_k (sum_r' U[i,j,r']*V[j,k,r']*W[k,i,r'] - 1) * V[j,k,r] * W[k,i,r]
            
            # Build residual matrix R[i,j,k] = sum_r U[i*mm+j,r]*V[j*pp+k,r]*W[k*nn+i,r] - 1
            R = np.zeros((nn, mm, pp))
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        R[i,j,k] = np.dot(U[i*mm+j, :], V[j*pp+k, :] * W[k*nn+i, :]) - 1.0
            
            err = np.sum(R**2)
            if err < best_err:
                best_err = err
                best_params = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-12:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        # dU[i*mm+j, r] += R[i,j,k] * V[j*pp+k, r] * W[k*nn+i, r]
                        # dV[j*pp+k, r] += R[i,j,k] * U[i*mm+j, r] * W[k*nn+i, r]
                        # dW[k*nn+i, r] += R[i,j,k] * U[i*mm+j, r] * V[j*pp+k, r]
                        r_idx = np.arange(rank)
                        dU[i*mm+j, :] += R[i,j,k] * V[j*pp+k, :] * W[k*nn+i, :]
                        dV[j*pp+k, :] += R[i,j,k] * U[i*mm+j, :] * W[k*nn+i, :]
                        dW[k*nn+i, :] += R[i,j,k] * U[i*mm+j, :] * V[j*pp+k, :]
            
            # Gradient descent with adaptive learning rate
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Reduce learning rate periodically
            if it % 1000 == 999 and it > 0:
                lr *= 0.5
        
        return best_params, best_err
    
    # First, get the rank-14 Strassen solution as a fallback
    U14, V14, W14 = get_strassen_242()
    
    # Try to find a rank-13 solution
    U13, V13, W13 = None, None, None
    found_13 = False
    
    if time.time() < deadline - 5:
        # Try multiple random initializations for rank 13
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            U_try, V_try, W_try = optimize_242(13, max_iter=3000, lr=0.05, seed=100+attempt)
            if verify_242(U_try, V_try, W_try, tol=1e-9):
                U13, V13, W13 = U_try, V_try, W_try
                found_13 = True
                break
        
        # If rank 13 didn't work, try rank 14 with optimization to ensure exactness
        if not found_13 and time.time() < deadline - 5:
            U14_opt, V14_opt, W14_opt = optimize_242(14, max_iter=2000, lr=0.02, seed=200)
            if verify_242(U14_opt, V14_opt, W14_opt, tol=1e-9):
                U14, V14, W14 = U14_opt, V14_opt, W14_opt
    
    # Build the full 2x4 x 4x8 decomposition
    if found_13:
        total_rank = 4 * 13  # 52
    else:
        total_rank = 4 * 14  # 56
    
    U_full = np.zeros((n * m, total_rank), dtype=float)  # 8 x total_rank
    V_full = np.zeros((m * p, total_rank), dtype=float)  # 32 x total_rank
    W_full = np.zeros((p * n, total_rank), dtype=float)  # 16 x total_rank
    
    # For each column-pair c in 0..3:
    # A is unchanged (same 2x4 matrix)
    # B columns 2c, 2c+1: B[0:4, 2c:2c+2]
    # C columns 2c, 2c+1: C[0:2, 2c:2c+2]
    
    for c_pair in range(4):
        base_r = c_pair * (13 if found_13 else 14)
        sub_rank = 13 if found_13 else 14
        U_sub = U13 if found_13 else U14
        V_sub = V13 if found_13 else V14
        W_sub = W13 if found_13 else W14
        
        # U: same for all column-pairs (A is the same 2x4 matrix)
        # U_full[i*4+j, base_r + r] = U_sub[i*4+j, r]
        for i in range(2):
            for j in range(4):
                for r in range(sub_rank):
                    U_full[i*4 + j, base_r + r] = U_sub[i*4 + j, r]
        
        # V: B[j, 2c_pair + k] -> V_full[j*8 + 2*c_pair + k, base_r + r] = V_sub[j*2 + k, r]
        for j in range(4):
            for k in range(2):
                for r in range(sub_rank):
                    V_full[j*8 + 2*c_pair + k, base_r + r] = V_sub[j*2 + k, r]
        
        # W: C[k, 2c_pair + i]... wait, W convention is W[k*n+i, r]
        # For the sub-problem, W_sub[k*2+i, r] corresponds to C[i, k] in the sub-problem
        # In the full problem, C[i, 2*c_pair + k] -> W_full[(2*c_pair + k)*2 + i, r]
        for k in range(2):
            for i in range(2):
                for r in range(sub_rank):
                    W_full[(2*c_pair + k)*2 + i, base_r + r] = W_sub[k*2 + i, r]
    
    # Verify the full decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.sum(U_full[i*m + j, :] * V_full[j*p + k, :] * W_full[k*n + i, :])
                if abs(actual - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank 64
        U_full = np.zeros((n * m, n * m * p), dtype=float)
        V_full = np.zeros((m * p, n * m * p), dtype=float)
        W_full = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_full[ii*m + jj, idx] = 1.0
                    V_full[jj*p + kk, idx] = 1.0
                    W_full[kk*n + ii, idx] = 1.0
                    idx += 1
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
