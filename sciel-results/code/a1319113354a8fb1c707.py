import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (150s limit)
    
    # Schoolbook decomposition
    rank_sb = n * m * p  # 54
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i', r] = delta(i,i') * delta(k,k')
    # Actually the convention is: C[i',k'] = sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] W[k*n+i', r] ... 
    # Let me re-read: W[k*n+i, r] where i is the row of C, k is the column of C
    # So C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Wait, that doesn't look right dimensionally. Let me think again.
    # 
    # Standard tensor decomposition for matrix multiplication:
    # C = A * B, where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # In the tensor formulation:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # 
    # With rank-r decomposition:
    # C[i,k] = sum_{r=1}^R (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k]) W[i,k,r]
    # 
    # But the interface says:
    # U: (n*m, rank) - so U[i*m+j, r]
    # V: (m*p, rank) - so V[j*p+k, r]  
    # W: (p*n, rank) - so W[k*n+i, r]
    # 
    # So: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # 
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r] = delta(i,i) delta(k,k) ... 
    # 
    # Actually it should be: the coefficient of A[i,j] B[j',k] in C[i',k'] is 
    # sum_r W[k'*n+i', r] U[i'*m+j, r] V[j'*p+k, r]
    # This should equal delta(i,i') * delta(j,j') * delta(k,k')
    # 
    # So we need: sum_r W[k'*n+i', r] U[i'*m+j, r] V[j'*p+k, r] = delta(i,i') delta(j,j') delta(k,k')
    
    def compute_error(U, V, W, R):
        """Compute the squared error of the decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                # Coefficient of A[i2,j2] B[j2,k2] in C[i,k]
                                coeff = 0.0
                                for r in range(R):
                                    coeff += W[k * n + i, r] * U[i2 * m + j2, r] * V[j2 * p + k2, r]
                                target = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                err += (coeff - target) ** 2
        return err
    
    def compute_gradient(U, V, W, R):
        """Compute gradients of the error with respect to U, V, W."""
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                target = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                for r in range(R):
                                    coeff = 0.0
                                    for r2 in range(R):
                                        coeff += W[k * n + i, r2] * U[i2 * m + j2, r2] * V[j2 * p + k2, r2]
                                    diff = coeff - target
                                    dU[i2 * m + j2, r] += 2 * diff * W[k * n + i, r] * V[j2 * p + k2, r]
                                    dV[j2 * p + k2, r] += 2 * diff * W[k * n + i, r] * U[i2 * m + j2, r]
                                    dW[k * n + i, r] += 2 * diff * U[i2 * m + j2, r] * V[j2 * p + k2, r]
        
        return dU, dV, dW
    
    # Try to find lower rank decompositions using gradient descent
    # Start from rank 53 down to 40
    for R in range(best_rank - 1, max(30, best_rank - 20), -1):
        if time.time() > deadline:
            break
        
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart * 1000 + R)
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            lr = 0.001
            best_err = float('inf')
            best_U_r, best_V_r, best_W_r = U.copy(), V.copy(), W.copy()
            
            for it in range(500):
                if time.time() > deadline:
                    break
                
                dU, dV, dW = compute_gradient(U, V, W, R)
                
                # Adaptive learning rate
                err = compute_error(U, V, W, R)
                if err < best_err:
                    best_err = err
                    best_U_r, best_V_r, best_W_r = U.copy(), V.copy(), W.copy()
                
                if best_err < 1e-10:
                    break
                
                # Scale gradients
                gnorm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if gnorm < 1e-15:
                    break
                
                U -= lr * dU / gnorm
                V -= lr * dV / gnorm
                W -= lr * dW / gnorm
                
                # Adapt learning rate
                if err < 0.01:
                    lr *= 0.9
                elif err > 1.0:
                    lr *= 0.5
                lr = min(lr, 0.01)
            
            # Check if we found a good solution
            final_err = compute_error(best_U_r, best_V_r, best_W_r, R)
            if final_err < 1e-8:
                best_rank = R
                best_U, best_V, best_W = best_U_r, best_V_r, best_W_r
                break  # Found a solution at this rank, try lower
        
        # If we found a solution at this rank, we can stop searching lower
        # (since we're going from high to low, and we want minimal rank)
        # Actually no, we should continue to find even lower rank
        # But break the restart loop for this R
    
    # Final polish: try to verify and potentially improve
    if best_rank < rank_sb:
        final_err = compute_error(best_U, best_V, best_W, best_rank)
        if final_err > 1e-6:
            # Fallback to schoolbook
            best_rank = rank_sb
            best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
