"""Matrix multiplication 2x5x6 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# Best known lower bound is max(2*5, 5*6, 2*6) = 30
# But that's just the rank bound from dimensions. For 2x5x6, the known best
# is likely 30 or slightly above. Let's try to find a decomposition of rank 30.

def _verify(U, V, W):
    """Verify that U, V, W form a valid decomposition.
    U shape: (n*m, rank)
    V shape: (m*p, rank)  
    W shape: (p*n, rank)
    For each k,i: sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]... 
    
    Actually the convention is: C = A * B where A is n x m, B is m x p
    C[k*n+i, j*m... wait.
    
    The output factor follows W[k*n+i, r].
    
    The tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    But the matrices are flattened:
    U has shape (n*m, rank), indexed by (i*m + j, r)
    V has shape (m*p, rank), indexed by (j*p + k, r)  
    W has shape (p*n, rank), indexed by (k*n + i, r)
    
    The matrix multiplication tensor has entries T[i,j,k] = 1 if... no.
    The tensor for matrix multiplication is:
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    So the trilinear form is: sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
    
    In the tensor decomposition:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    where T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    And T[i,j,k] should equal 1 if this corresponds to A[i,j]*B[j,k]*C[k... 
    
    Wait. The matrix multiplication tensor has T[i,j,k] = δ (it's 1 for all i,j,k 
    in the sense that the bilinear map is sum_j A_ij B_jk). Actually the tensor 
    T_{i,j,k} is defined such that C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}
    where T_{i,j,k} = 1 for all valid i,j,k. No wait, T_{i,j,k} = 1 always, 
    and C_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    So the condition is: for all i, j, k: 
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    But that's not right either. Let me re-read.
    
    The standard: the multiplication tensor T has T[i,j,k] = 1 for all (i,j,k) in 
    the appropriate range, and the bilinear map is <C, A⊗B> = sum_{i,j,k} C[i,k] T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} C[i,k] A[i,j] B[j,k] = sum_i C[i,k] (AB)[i,k].
    
    So yes, T[i,j,k] = 1 for all i,j,k.
    
    The rank-r decomposition means: T = sum_{r=0}^{rank-1} T^(r) where each T^(r) 
    is a rank-1 tensor with T^(r)[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    So the condition is: for all (i,j,k) in {0..n-1}x{0..m-1}x{0..p-1}:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # leave 10s margin
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = max(10, 30, 12) = 30
    lower_bound = max(n*m, m*p, n*p)
    
    # Known results: for 2x5x6, let's try to see if 30 is achievable
    # The key insight: we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # Strategy: use random initialization and gradient-based optimization
    # to find if a decomposition of a given rank exists.
    
    def try_rank(r):
        """Try to find a decomposition of rank r using optimization."""
        nonlocal best_rank, best_U, best_V, best_W
        
        if r <= 0:
            return False
        
        # Objective: minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        # Use simple iterative method
        
        random.seed(r * 12345)
        np.random.seed(r * 54321)
        
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Normalize to avoid scaling issues
        # Use a fixed-point / iterative approach
        
        lr = 0.01
        for iteration in range(5000):
            if time.time() > deadline:
                return False
            
            # Compute residuals
            residuals = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r_idx in range(r):
                            s += U[i*m + j, r_idx] * V[j*p + k, r_idx] * W[k*n + i, r_idx]
                        residuals[(i,j,k)] = s - 1.0
            
            # Check if we've converged
            max_res = max(abs(v) for v in residuals.values())
            if max_res < 1e-5:
                best_rank = r
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                return True
            
            # Gradient update
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res = residuals[(i,j,k)]
                        for r_idx in range(r):
                            dU[i*m + j, r_idx] += res * V[j*p + k, r_idx] * W[k*n + i, r_idx]
                            dV[j*p + k, r_idx] += res * U[i*m + j, r_idx] * W[k*n + i, r_idx]
                            dW[k*n + i, r_idx] += res * U[i*m + j, r_idx] * V[j*p + k, r_idx]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if iteration > 100 and max_res > 1e-3:
                lr *= 0.5
            
            if lr < 1e-8:
                break
        
        return False
    
    # Try ranks from lower bound upward
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
        if try_rank(r):
            break
    
    # Also try a few random restarts with the best found rank or lower_bound
    if time.time() < deadline:
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            # Try lower_bound with different random seeds
            random.seed(10000 + attempt)
            np.random.seed(20000 + attempt)
            
            r = lower_bound
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            lr = 0.005
            for iteration in range(3000):
                if time.time() > deadline - 2:
                    break
                
                residuals = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r_idx in range(r):
                                s += U[i*m + j, r_idx] * V[j*p + k, r_idx] * W[k*n + i, r_idx]
                            residuals[(i,j,k)] = s - 1.0
                
                max_res = max(abs(v) for v in residuals.values())
                if max_res < 1e-5:
                    best_rank = r
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res = residuals[(i,j,k)]
                            for r_idx in range(r):
                                dU[i*m + j, r_idx] += res * V[j*p + k, r_idx] * W[k*n + i, r_idx]
                                dV[j*p + k, r_idx] += res * U[i*m + j, r_idx] * W[k*n + i, r_idx]
                                dW[k*n + i, r_idx] += res * U[i*m + j, r_idx] * V[j*p + k, r_idx]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if iteration % 200 == 199 and max_res > 1e-3:
                    lr *= 0.5
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
