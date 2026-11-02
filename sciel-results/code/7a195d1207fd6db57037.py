"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor: T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k) for the multiplication
    
    # The multiplication tensor has T[i,j,k] = 1 if the indices match the multiplication pattern
    # Actually, the tensor is: for each (i,j,k), the coefficient is 1 (it's just the identity in the bilinear form)
    # Wait, let me reconsider. The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the "multiplication" maps (A,B) -> C
    # In the tensor decomposition, we need:
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for all valid (i,j,k) that participate
    # Actually no - the tensor for matrix multiplication C = A*B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the "all ones" in the appropriate sense)
    # No wait - the tensor is defined such that <T, A ⊗ B> = C
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # And the decomposition is: T = sum_r U[:, :, r] ⊗ V[:, :, r] ⊗ W[:, :, r]
    # Where U is indexed by (i,j), V by (j,k), W by (i,k) with the trace convention W[k*n+i, r]
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's try to find a low-rank decomposition
    # Start with known good ranks and try to go lower
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try ranks from low to high
    # Known: for 3x3x3, min rank is 23. For 3x3x4, let's try various ranks.
    # Let's try ranks: 24, 25, 26, ... up to 36
    
    for target_rank in range(20, min(37, n*m*p + 1)):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of this rank
        # Use random initialization and optimization
        success = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            try:
                # Random initialization
                U = np.random.randn(n*m, target_rank)
                V = np.random.randn(m*p, target_rank)
                W = np.random.randn(p*n, target_rank)
                
                # Scale to help convergence
                scale = 1.0 / (target_rank ** 0.5)
                U *= scale
                V *= scale
                W *= scale
                
                # Use gradient descent to minimize the error
                # Error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                
                lr = 0.01
                for step in range(2000):
                    if time.time() > deadline:
                        break
                    
                    # Compute current values
                    # C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # But we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for each (i,j,k)
                    
                    # Compute gradient
                    # Let S[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # Error = sum_{i,j,k} (S[i,j,k] - 1)^2
                    # dError/dU[i*m+j,r] = 2*(S[i,j,k]-1) * V[j*p+k,r] * W[k*n+i,r]
                    
                    # This is expensive to compute naively. Let's use a smarter approach.
                    
                    # Actually, let's just check if we're close enough
                    # Compute S for all (i,j,k)
                    S = np.zeros((n, m, p))
                    for r in range(target_rank):
                        u_col = U[:, r]  # (n*m,)
                        v_col = V[:, r]  # (m*p,)
                        w_col = W[:, r]  # (p*n,)
                        
                        # Reshape
                        u_reshaped = u_col.reshape(n, m)
                        v_reshaped = v_col.reshape(m, p)
                        w_reshaped = w_col.reshape(p, n)
                        
                        # S[i,j,k] += u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    S[i,j,k] += u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
                    
                    error = np.sum((S - 1.0)**2)
                    
                    if error < 1e-10:
                        success = True
                        break
                    
                    if step % 50 == 0 and error < 1e-6:
                        # Check for near-exact
                        if error < 1e-8:
                            success = True
                            break
                    
                    # Compute gradients (approximate)
                    # This is getting complex. Let's use a simpler approach:
                    # Just do random restarts with different initializations
                    
                    # Update using simple gradient step
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                err = S[i,j,k] - 1.0
                                for r in range(target_rank):
                                    ui = U[i*m+j, r]
                                    vj = V[j*p+k, r]
                                    wk = W[k*n+i, r]
                                    grad_U[i*m+j, r] += 2*err * vj * wk
                                    grad_V[j*p+k, r] += 2*err * ui * wk
                                    grad_W[k*n+i, r] += 2*err * ui * vj
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    
                    # Normalize to prevent blowup
                    for r in range(target_rank):
                        norm_u = np.linalg.norm(U[:, r])
                        norm_v = np.linalg.norm(V[:, r])
                        norm_w = np.linalg.norm(W[:, r])
                        if norm_u > 0 and norm_v > 0 and norm_w > 0:
                            # Keep product roughly constant
                            target_norm = 1.0
                            scale_u = target_norm / norm_u if norm_u > target_norm else 1.0
                            scale_v = target_norm / norm_v if norm_v > target_norm else 1.0
                            scale_w = target_norm / norm_w if norm_w > target_norm else 1.0
                            U[:, r] *= scale_u
                            V[:, r] *= scale_v
                            W[:, r] *= scale_w
                    
                    if error < 1e-8:
                        success = True
                        break
                    
                    if step % 100 == 0 and error > 1e2:
                        break  # Diverging
                    
                if success:
                    break
                    
            except Exception:
                continue
        
        if success:
            # Verify the solution
            S = np.zeros((n, m, p))
            for r in range(target_rank):
                u_col = U[:, r]
                v_col = V[:, r]
                w_col = W[:, r]
                u_reshaped = u_col.reshape(n, m)
                v_reshaped = v_col.reshape(m, p)
                w_reshaped = w_col.reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            S[i,j,k] += u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
            
            if np.allclose(S, 1.0, atol=1e-8):
                # Convert to list format
                U_list = [[float(U[i, r]) for r in range(target_rank)] for i in range(n*m)]
                V_list = [[float(V[i, r]) for r in range(target_rank)] for i in range(m*p)]
                W_list = [[float(W[i, r]) for r in range(target_rank)] for i in range(p*n)]
                
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better solution found, return schoolbook
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
# EVOLVE-BLOCK-END
