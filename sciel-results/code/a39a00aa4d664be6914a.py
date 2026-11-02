import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = 2, 3, 9
    
    # The 2x3x9 tensor decomposes into 3 independent 2x3x3 tensors
    # (columns 0-2, 3-5, 6-8 of the 9-column output)
    # Each 2x3x3 tensor has rank 6
    # Total rank = 3 * 6 = 18
    
    # First, find a rank-6 decomposition for 2x3x3
    # U: (6, R), V: (9, R), W: (6, R)
    # Condition: sum_r U[3i+j, r] * V[3j+k, r] * W[2k+i, r] = 1 for all i,j,k
    
    def find_2x3x3_decomp(R, time_limit):
        """Find rank-R decomposition for 2x3x3 tensor."""
        # U: (6, R), V: (9, R), W: (6, R)
        # Indices: U[i*3+j], V[j*3+k], W[k*2+i]
        
        best = None
        best_err = float('inf')
        
        for attempt in range(200):
            if time.time() > time_limit:
                break
            
            np.random.seed(attempt * 12345 + 67890)
            U = np.random.randn(6, R)
            V = np.random.randn(9, R)
            W = np.random.randn(6, R)
            
            # Gradient descent on the sum of squared residuals
            for step in range(500):
                if time.time() > time_limit:
                    break
                
                # Compute residuals and gradients
                # residual[i,j,k] = sum_r U[3i+j,r]*V[3j+k,r]*W[2k+i,r] - 1
                # d/dU[a,r] = sum over (j,k) where 3i+j=a: V[3j+k,r]*W[2k+i,r]*(residual)
                # This is complex, let's use a simpler approach
                
                # Actually, let's just compute the residual and use a simple update
                res = np.zeros((2,3,3))
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            s = 0.0
                            for r in range(R):
                                s += U[3*i+j, r] * V[3*j+k, r] * W[2*k+i, r]
                            res[i,j,k] = s - 1.0
                
                err = np.sum(res**2)
                if err < best_err:
                    best_err = err
                    best = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-12:
                    break
                
                # Simple gradient step
                lr = 0.01
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            for r in range(R):
                                # d/dU[3i+j,r]
                                grad_U[3*i+j, r] += V[3*j+k, r] * W[2*k+i, r] * res[i,j,k]
                                # d/dV[3j+k,r]
                                grad_V[3*j+k, r] += U[3*i+j, r] * W[2*k+i, r] * res[i,j,k]
                                # d/dW[2k+i,r]
                                grad_W[2*k+i, r] += U[3*i+j, r] * V[3*j+k, r] * res[i,j,k]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Normalize to prevent divergence
                U /= (np.linalg.norm(U) + 1e-10)
                V /= (np.linalg.norm(V) + 1e-10)
                W /= (np.linalg.norm(W) + 1e-10)
        
        if best is not None and best_err < 1e-8:
            return best
        return None
    
    # Try to find rank-6 decomposition
    decomp = find_2x3x3_decomp(6, start_time + 120)
    
    if decomp is not None:
        U_sub, V_sub, W_sub = decomp
        # Build the full 2x3x9 decomposition
        # The 9 output columns split into 3 groups of 3
        # For group g (g=0,1,2), output columns are 3g, 3g+1, 3g+2
        # U is the same for all groups (input A is 2x3)
        # V: for group g, B columns are 3g, 3g+1, 3g+2
        #   V index for group g, local column k: (3g+k)*3 + j... wait
        
        # Let me re-index carefully.
        # Full problem: A is 2x3, B is 3x9, C is 2x9
        # U: (2*3, R) = (6, R)
        # V: (3*9, R) = (27, R)
        # W: (9*2, R) = (18, R)
        # Condition: sum_r U[i*3+j, r] * V[j*9+k, r] * W[k*2+i, r] = 1 for all i,j,k
        
        # For the block structure:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Split k into k = 3g + k' where g=0,1,2 and k'=0,1,2
        # Then C[i, 3g+k'] = sum_j A[i,j] B[j, 3g+k']
        # This is a 2x3x3 multiplication for each g
        
        # For block g:
        # U_block[i*3+j] = U_sub[i*3+j] (same for all g)
        # V_block[j*9 + (3g+k')] = V_sub[j*3+k'] (for the local 3x3 block)
        # W_block[(3g+k')*2 + i] = W_sub[k'*2+i]
        
        total_rank = 18
        U_full = np.zeros((6, total_rank))
        V_full = np.zeros((27, total_rank))
        W_full = np.zeros((18, total_rank))
        
        for g in range(3):
            r_offset = g * 6
            # U is the same
            U_full[:, r_offset:r_offset+6] = U_sub
            # V: for local column k', global column is 3g+k'
            # V_full[j*9 + (3g+k'), r_offset + r] = V_sub[j*3+k', r]
            for j in range(3):
                for k_local in range(3):
                    k_global = 3*g + k_local
                    V_full[j*9 + k_global, r_offset:r_offset+6] = V_sub[j*3 + k_local]
            # W: for local column k', global column is 3g+k'
            # W_full[(3g+k')*2 + i, r_offset + r] = W_sub[k'*2 + i, r]
            for k_local in range(3):
                k_global = 3*g + k_local
                for i in range(2):
                    W_full[k_global*2 + i, r_offset+r_offset: r_offset+r_offset+6] = W_sub[k_local*2 + i]
        
        # Verify
        max_err = 0
        for i in range(2):
            for j in range(3):
                for k in range(9):
                    s = 0.0
                    for r in range(total_rank):
                        s += U_full[i*3+j, r] * V_full[j*9+k, r] * W_full[k*2+i, r]
                    max_err = max(max_err, abs(s - 1.0))
        
        if max_err < 1e-8:
            # Convert to lists
            U_list = [[float(U_full[i,j]) for j in range(total_rank)] for i in range(6)]
            V_list = [[float(V_full[i,j]) for j in range(total_rank)] for i in range(27)]
            W_list = [[float(W_full[i,j]) for j in range(total_rank)] for i in range(18)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 54
    rank = 54
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(27)]
    W = [[0.0] * rank for _ in range(18)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(9):
                U[i*3+j][col] = 1.0
                V[j*9+k][col] = 1.0
                W[k*2+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
