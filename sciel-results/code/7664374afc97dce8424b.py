"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time as _time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = _time.time()
    deadline = start_time + 110  # safety margin
    
    def elapsed():
        return _time.time() - start_time
    
    def verify(U_arr, V_arr, W_arr):
        """Verify that the decomposition is exact."""
        # T[i*m+j, j*p+k, k*n+i] should equal 1 for all i,j,k
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U_arr.shape[1]):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        # Also check that non-matching entries are 0
        # T[a,b,c] = 0 unless a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # This is automatically satisfied if the above holds for all i,j,k
        # and the tensor is sparse
        return True
    
    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    # First: build schoolbook decomposition (rank 80)
    schoolbook_rank = n * m * p  # 80
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Try to find a lower-rank decomposition using gradient descent
    best_result = None
    best_rank = schoolbook_rank
    
    # Try various target ranks
    target_ranks = [72, 70, 68, 66, 64, 60, 56, 52, 48]
    
    for target_r in target_ranks:
        if elapsed() > deadline - 20:
            break
        
        # Initialize with structured values based on schoolbook
        # Project the schoolbook terms onto target_r dimensions
        U_init = np.random.randn(n * m, target_r) * 0.1
        V_init = np.random.randn(m * p, target_r) * 0.1
        W_init = np.random.randn(p * n, target_r) * 0.1
        
        # Add some structure: use the schoolbook pattern
        for idx in range(min(schoolbook_rank, target_r)):
            U_init[idx % (n*m), idx] += 1.0
            V_init[idx % (m*p), idx] += 1.0
            W_init[idx % (p*n), idx] += 1.0
        
        # Gradient descent
        U_gd = U_init.copy()
        V_gd = V_init.copy()
        W_gd = W_init.copy()
        
        lr = 0.01
        for iteration in range(500):
            if elapsed() > deadline - 15:
                break
            
            # Compute residual: R[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c]
            # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
            # Otherwise T[a,b,c] = 0
            
            # Compute gradient w.r.t. U:
            # d/dU[a,r] sum_{a',b',c'} R[a',b',c']^2
            # = 2 * sum_{b',c'} R[a,b',c'] * V[b',r] * W[c',r]
            
            # This is expensive. Let's use a simplified approach.
            # Compute the residual tensor and gradients efficiently.
            
            # For each r, the contribution to the residual is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # Residual = sum_r U[:,r]⊗V[:,r]⊗W[:,r] - T
            
            # Gradient w.r.t. U[:,r] = sum_{b,c} R[:,b,c] * V[b,r] * W[c,r]
            # This can be computed as: for each r, grad_U[:,r] = (R_matmuls V[:,r] * W[:,r])
            
            # Let's compute R as a 3D array
            # R[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c]
            
            # This is (n*m) x (m*p) x (p*n) = 16 x 20 x 20 = 6400 entries
            # For each entry, sum over r terms: 6400 * target_r operations per gradient step
            
            # Build T
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] = 1.0
            
            # Compute residual
            R = np.zeros((n*m, m*p, p*n))
            for r in range(target_r):
                R += np.einsum('a,b,c->abc', U_gd[:,r], V_gd[:,r], W_gd[:,r])
            R -= T
            
            # Compute gradients
            # grad_U[a,r] = sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
            # grad_V[b,r] = sum_{a,c} R[a,b,c] * U[a,r] * W[c,r]
            # grad_W[c,r] = sum_{a,b} R[a,b,c] * U[a,r] * V[b,r]
            
            grad_U = np.zeros_like(U_gd)
            grad_V = np.zeros_like(V_gd)
            grad_W = np.zeros_like(W_gd)
            
            for r in range(target_r):
                # grad_U[:,r] = sum_{b,c} R[:,:,b,c]... 
                # Actually: grad_U[a,r] = sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
                # = (R @ (V[:,r] * W[:,r]))[a] where @ is matrix product over (b,c)
                
                # R[a,b,c] * V[b,r] * W[c,r] summed over b,c
                # = sum_b sum_c R[a,b,c] * V[b,r] * W[c,r]
                # = sum_b (sum_c R[a,b,c] * W[c,r]) * V[b,r]
                
                # Let's compute this as: for each a, grad_U[a,r] = sum_{b,c} R[a,b,c]*V[b,r]*W[c,r]
                # Use einsum
                grad_U[:,r] = np.einsum('abc,b,c->a', R, V_gd[:,r], W_gd[:,r])
                grad_V[:,r] = np.einsum('abc,a,c->b', R, U_gd[:,r], W_gd[:,r])
                grad_W[:,r] = np.einsum('abc,a,b->c', R, U_gd[:,r], V_gd[:,r])
            
            # Update
            U_gd -= lr * grad_U
            V_gd -= lr * grad_V
            W_gd -= lr * grad_W
            
            # Check convergence
            if iteration % 50 == 0:
                res_norm = np.linalg.norm(R)
                if res_norm < 1e-8:
                    break
        
        # Snap to half-integers
        U_snap = snap_to_half_int(U_gd)
        V_snap = snap_to_half_int(V_gd)
        W_snap = snap_to_half_int(W_gd)
        
        # Verify
        if verify(U_snap, V_snap, W_snap):
            if target_r < best_rank:
                best_rank = target_r
                best_result = (U_snap, V_snap, W_snap)
        
        # Also try without snapping (in case the values are already close to half-integers)
        if verify(U_gd, V_gd, W_gd):
            # Check if entries are already half-integers
            U_close = np.allclose(U_gd, snap_to_half_int(U_gd), atol=1e-6)
            V_close = np.allclose(V_gd, snap_to_half_int(V_gd), atol=1e-6)
            W_close = np.allclose(W_gd, snap_to_half_int(W_gd), atol=1e-6)
            if U_close and V_close and W_close:
                if target_r < best_rank:
                    best_rank = target_r
                    best_result = (U_snap, V_snap, W_snap)
    
    # If no lower-rank decomposition found, return schoolbook
    if best_result is None:
        U_final = U_sb
        V_final = V_sb
        W_final = W_sb
    else:
        U_final, V_final, W_final = best_result
    
    # Convert to lists of integers (multiply by 2 to get integers, then divide back)
    # Actually, the problem says integer/half-integer entries
    # Let's return as floats that are multiples of 0.5
    rank_final = U_final.shape[1]
    
    U_list = [[float(U_final[i, j]) for j in range(rank_final)] for i in range(n * m)]
    V_list = [[float(V_final[i, j]) for j in range(rank_final)] for i in range(m * p)]
    W_list = [[float(W_final[i, j]) for j in range(rank_final)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
