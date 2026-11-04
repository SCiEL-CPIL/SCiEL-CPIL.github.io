import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 8
    deadline = time.time() + 118  # Safety margin for 150s wall-clock limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        # Check all (a,b,c) where a=i*m+j, b=j'*p+k, c=k'*n+i'
        # For the tensor to be correct, we need:
        # sum_t U[a,t]*V[b,t]*W[c,t] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # and 0 otherwise
        # But actually, the tensor T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
        # Wait, let me re-derive. The matrix multiplication tensor is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[(i,j),(j,k),(k,i)] = 1 for all valid i,j,k, and 0 otherwise.
        # In terms of flat indices: a = i*m+j, b = j*p+k, c = k*n+i
        # The condition is that for each (a,b,c), sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
        # where T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
        # and 0 otherwise.
        
        # Build the target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute the reconstructed tensor
        for t in range(rank):
            u = np.array(U[:, t])
            v = np.array(V[:, t])
            w = np.array(W[:, t])
            T_recon = u[:, None, None] * v[None, :, None] * w[None, None, :]
            T += T_recon
        
        max_err = np.max(np.abs(T - 1.0))
        # Also check that non-target entries are zero
        target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_mask[i*m+j, j*p+k, k*n+i] = True
        non_target_err = np.max(np.abs(T[~target_mask]))
        
        return max_err < tol and non_target_err < tol
    
    def round_to_half_int(arr, tol=1e-4):
        """Round array to nearest half-integer if close enough."""
        arr_r = np.round(np.array(arr) * 2) / 2
        if np.max(np.abs(np.array(arr) - arr_r)) < tol:
            return arr_r.tolist(), True
        return None, False
    
    best_result = schoolbook()
    best_rank = 72
    
    # Try to find a lower rank decomposition using iterative refinement
    # We'll use a combination of random restarts and local optimization
    
    # The key insight: we need sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c] for all (a,b,c)
    # This is a system of cubic equations. We can use a fixed-point iteration approach.
    
    # Try ranks from 60 down to 50
    for target_rank in range(60, 49, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for attempt in range(15):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(attempt * 7919 + target_rank * 13)
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
            
            # Use a multi-stage approach:
            # Stage 1: Coarse optimization with large step size
            # Stage 2: Fine optimization with smaller step size
            
            for stage in range(2):
                lr = 0.05 if stage == 0 else 0.01
                steps = 300 if stage == 0 else 500
                
                for step in range(steps):
                    if time.time() > deadline - 20:
                        break
                    
                    # Compute the current tensor values
                    # For efficiency, we only need to check the target entries
                    # and a sample of non-target entries
                    
                    # Compute gradients for target entries
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    
                    # Target entries
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m + j
                                b = j*p + k
                                c = k*n + i
                                val = np.sum(U[a,:] * V[b,:] * W[c,:])
                                grad = 2.0 * (val - 1.0)
                                dU[a,:] += grad * V[b,:] * W[c,:]
                                dV[b,:] += grad * U[a,:] * W[c,:]
                                dW[c,:] += grad * U[a,:] * V[b,:]
                    
                    # Sample some non-target entries to push them toward zero
                    if stage == 1:
                        np.random.seed(step * 31 + attempt)
                        for _ in range(20):
                            a = np.random.randint(0, n*m)
                            b = np.random.randint(0, m*p)
                            c = np.random.randint(0, p*n)
                            # Check if this is a target entry
                            i = a // m
                            j = a % m
                            j2 = b // p
                            k = b % p
                            k2 = c // n
                            i2 = c % n
                            if j == j2 and k == k2 and i == i2:
                                continue  # Skip target entries
                            val = np.sum(U[a,:] * V[b,:] * W[c,:])
                            grad = 2.0 * val
                            dU[a,:] += grad * V[b,:] * W[c,:]
                            dV[b,:] += grad * U[a,:] * W[c,:]
                            dW[c,:] += grad * U[a,:] * V[b,:]
                    
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                    
                    # Check convergence
                    if step % 100 == 0 or step == steps - 1:
                        # Quick error check
                        max_err = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    a = i*m + j
                                    b = j*p + k
                                    c = k*n + i
                                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                                    max_err = max(max_err, abs(val - 1.0))
                        
                        if max_err < 1e-8:
                            # Try to round to half-integers
                            U_list, U_ok = round_to_half_int(U)
                            V_list, V_ok = round_to_half_int(V)
                            W_list, W_ok = round_to_half_int(W)
                            
                            if U_ok and V_ok and W_ok:
                                # Full verification
                                if verify_decomposition(U_list, V_list, W_list):
                                    result = {"U": U_list, "V": V_list, "W": W_list}
                                    if target_rank < best_rank:
                                        best_rank = target_rank
                                        best_result = result
                                    return best_result
                            break
            
            if best_rank < target_rank:
                break
    
    return best_result
