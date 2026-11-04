import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition with rank p*m*n."""
    rank = p * m * n
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
    return U, V, W, rank

def _residual(U, V, W, n, m, p):
    """Compute sum of squared residuals of tensor equations."""
    # T[i,j,k] should equal sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    # But the actual tensor is T[i,j,k] = 1 for the "multiplication" pattern
    # Actually, the tensor is defined by: C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition means: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] should
    # reconstruct the multiplication. Let me think about what the target is.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if we consider the
    # trilinear form sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C'[k,i] where C' is
    # the dual of C. Actually, the standard is:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] * (dual)
    # 
    # For the decomposition to work, we need:
    # For all A, B: sum_{i,k} C[i,k] D[i,k] = sum_t (sum_{i,j} U[i*m+j,t] A[i,j]) 
    #   * (sum_{j,k} V[j*p+k,t] B[j,k]) * (sum_{i,k} W[k*n+i,t] D[i,k])
    # where C = A*B.
    # 
    # This means: sum_{i,j,k} A[i,j] B[j,k] D[i,k] = 
    #   sum_t (sum_{i',j'} U[i'*m+j',t] A[i',j']) (sum_{j',k'} V[j'*p+k',t] B[j',k']) (sum_{i'',k''} W[k''*n+i'',t] D[i'',k''])
    # 
    # Expanding the RHS: sum_t sum_{i,j,j',k,k',i'',k''} U[i*m+j,t] V[j'*p+k',t] W[k''*n+i'',t] A[i,j] B[j',k'] D[i'',k'']
    # 
    # For this to equal LHS = sum_{i,j,k} A[i,j] B[j,k] D[i,k], we need:
    # sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t] = 1 for all i,j,k
    # Wait, that's not right either. Let me reconsider.
    
    # Actually, the correct condition is that the bilinear form matches. The
    # multiplication tensor is the 3-tensor T with T[i,j,k] such that
    # <T, A⊗B⊗C^T> = trace(C^T A B) = sum_{i,k} C[i,k] (AB)[i,k]
    # = sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    # 
    # So T[i,j,k] in the sense that sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]... 
    # I'm getting confused with index conventions.
    
    # Let me use a different approach. The decomposition is valid iff for all
    # random A, B: C = A*B can be computed as sum_t W_t * (U_t · A) * (V_t · B)
    # where (U_t · A) = sum_{i,j} U[i*m+j,t] A[i,j], etc.
    
    # So: (AB)[i,k] = sum_t W[k*n+i,t] * (sum_{i',j'} U[i'*m+j',t] A[i',j']) * (sum_{j',k'} V[j'*p+k',t] B[j',k'])
    
    # Let's verify with random A, B.
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    r = U_arr.shape[1]
    
    # Compute C_decomp[i,k] = sum_t W[k*n+i,t] * (sum_{i',j'} U[i'*m+j',t] A[i',j']) * (sum_{j',k'} V[j'*p+k',t] B[j',k'])
    C_decomp = np.zeros((n, p))
    for t in range(r):
        u_val = np.sum(U_arr[:, t] * A.flatten())
        v_val = np.sum(V_arr[:, t] * B.flatten())
        for k in range(p):
            for i in range(n):
                C_decomp[i, k] += W_arr[k * n + i, t] * u_val * v_val
    
    return np.sum((C_true - C_decomp) ** 2)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school, rank_school = _build_schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Try to find lower rank decompositions using gradient descent
    # We'll try ranks from 83 down to some minimum
    for target_rank in range(rank_school - 1, 40, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            
            # Initialize with schoolbook and zero out some columns, then perturb
            U_init = np.zeros((n * m, target_rank))
            V_init = np.zeros((m * p, target_rank))
            W_init = np.zeros((p * n, target_rank))
            
            # Use first target_rank columns of schoolbook as initialization
            U_init[:] = np.array(U_school)[:, :target_rank]
            V_init[:] = np.array(V_school)[:, :target_rank]
            W_init[:] = np.array(W_school)[:, :target_rank]
            
            # Add small random perturbation to escape local minima
            U_init += 0.01 * np.random.randn(n * m, target_rank)
            V_init += 0.01 * np.random.randn(m * p, target_rank)
            W_init += 0.01 * np.random.randn(p * n, target_rank)
            
            # Gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Compute residual and gradients numerically
                U_cur = U_init.tolist()
                V_cur = V_init.tolist()
                W_cur = W_init.tolist()
                res = _residual(U_cur, V_cur, W_cur, n, m, p)
                
                if res < 1e-10:
                    # Found a valid decomposition
                    best_U, best_V, best_W, best_rank = U_cur, V_cur, W_cur, target_rank
                    break
                
                # Compute gradients via finite differences (expensive but simple)
                # For efficiency, only do this for a subset of steps
                if step % 10 == 0:
                    # Try a few random directions
                    improved = False
                    for _ in range(20):
                        if time.time() > deadline - 10:
                            break
                        dU = 0.01 * np.random.randn(n * m, target_rank)
                        dV = 0.01 * np.random.randn(m * p, target_rank)
                        dW = 0.01 * np.random.randn(p * n, target_rank)
                        
                        U_new = (U_init + dU).tolist()
                        V_new = (V_init + dV).tolist()
                        W_new = (W_init + dW).tolist()
                        res_new = _residual(U_new, V_new, W_new, n, m, p)
                        
                        if res_new < res:
                            U_init = U_init + dU
                            V_init = V_init + dV
                            W_init = W_init + dW
                            res = res_new
                            improved = True
                    if not improved:
                        break
                
                # Decay learning rate
                lr *= 0.95
            
            # Check if we found a valid decomposition
            U_final = U_init.tolist()
            V_final = V_init.tolist()
            W_final = W_init.tolist()
            res_final = _residual(U_final, V_final, W_final, n, m, p)
            if res_final < 1e-10:
                best_U, best_V, best_W, best_rank = U_final, V_final, W_final, target_rank
                break
    
    # Snap to exact values if close to integers or half-integers
    def snap_values(arr):
        arr = np.array(arr, dtype=float)
        # Try snapping to nearest integer or half-integer
        candidates = []
        for scale in [1, 2]:
            snapped = np.round(arr * scale) / scale
            candidates.append(snapped)
        # Choose the one with smallest residual
        best_snapped = arr
        best_res = _residual(arr.tolist(), V_final, W_final, n, m, p) if len(arr.shape) == 2 else float('inf')
        # This is getting complicated; just return as-is for now
        return arr
    
    # Convert to lists
    best_U_list = [[float(x) for x in row] for row in np.array(best_U)]
    best_V_list = [[float(x) for x in row] for row in np.array(best_V)]
    best_W_list = [[float(x) for x in row] for row in np.array(best_W)]
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
