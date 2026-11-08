import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    r_school = n * m * p  # 56
    
    # Build the schoolbook decomposition (rank 56)
    U = np.zeros((n * m, r_school))
    V = np.zeros((m * p, r_school))
    W = np.zeros((p * n, r_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    max_time = 140  # Leave margin before 150s limit
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = r_school
    
    def compute_reconstruction(U_r, V_r, W_r, r):
        """Vectorized reconstruction of the tensor."""
        Rec = np.zeros((n, m, p))
        for rr in range(r):
            U_reshaped = U_r[:, rr].reshape(n, m)
            V_reshaped = V_r[:, rr].reshape(m, p)
            W_reshaped = W_r[:, rr].reshape(p, n)
            Rec_r = np.einsum('ij,jk,ki->ijk', U_reshaped, V_reshaped, W_reshaped)
            Rec += Rec_r
        return Rec
    
    def compute_gradients(U_r, V_r, W_r, Error, r):
        """Vectorized gradient computation using einsum."""
        dU = np.zeros_like(U_r)
        dV = np.zeros_like(V_r)
        dW = np.zeros_like(W_r)
        
        for rr in range(r):
            U_reshaped = U_r[:, rr].reshape(n, m)
            V_reshaped = V_r[:, rr].reshape(m, p)
            W_reshaped = W_r[:, rr].reshape(p, n)
            
            # dU[i,j] = -2 * sum_k Error[i,j,k] * V[j,k] * W[k,i]
            dU_reshaped = -2.0 * np.einsum('ijk,jk,ki->ij', Error, V_reshaped, W_reshaped)
            dU[:, rr] = dU_reshaped.flatten()
            
            # dV[j,k] = -2 * sum_i Error[i,j,k] * U[i,j] * W[k,i]
            dV_reshaped = -2.0 * np.einsum('ijk,ij,ki->jk', Error, U_reshaped, W_reshaped)
            dV[:, rr] = dV_reshaped.flatten()
            
            # dW[k,i] = -2 * sum_j Error[i,j,k] * U[i,j] * V[j,k]
            dW_reshaped = -2.0 * np.einsum('ijk,ij,jk->ki', Error, U_reshaped, V_reshaped)
            dW[:, rr] = dW_reshaped.flatten()
        
        return dU, dV, dW
    
    # Try ranks from 28 to r_school-1
    for r in range(28, r_school):
        if time.time() - start_time > max_time:
            break
        
        if r >= best_rank:
            continue
        
        # Try more restarts for lower ranks
        n_restarts = 5 if r < 40 else 3
        for restart in range(n_restarts):
            if time.time() - start_time > max_time:
                break
            
            # Random initialization with different seeds
            np.random.seed(42 + r * 1000 + restart * 7)
            U_r = np.random.randn(n * m, r) * 0.3
            V_r = np.random.randn(m * p, r) * 0.3
            W_r = np.random.randn(p * n, r) * 0.3
            
            # Adaptive learning rate gradient descent
            lr = 0.1
            prev_loss = np.inf
            for step in range(500):
                if time.time() - start_time > max_time:
                    break
                
                Rec = compute_reconstruction(U_r, V_r, W_r, r)
                Error = 1.0 - Rec
                loss = np.sum(Error ** 2)
                
                if loss < 1e-12:
                    break
                
                # Adaptive learning rate
                if loss > prev_loss * 1.01:
                    lr *= 0.5
                elif step % 50 == 0 and loss < prev_loss * 0.99:
                    lr *= 1.1
                prev_loss = loss
                
                dU, dV, dW = compute_gradients(U_r, V_r, W_r, Error, r)
                
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
            
            # Check if solution is close enough
            Rec_final = compute_reconstruction(U_r, V_r, W_r, r)
            max_err = np.max(np.abs(Rec_final - 1.0))
            
            if max_err < 1e-8:
                # Try rounding to nearest half-integer
                U_rounded = np.round(U_r * 2) / 2
                V_rounded = np.round(V_r * 2) / 2
                W_rounded = np.round(W_r * 2) / 2
                
                Rec_check = compute_reconstruction(U_rounded, V_rounded, W_rounded, r)
                max_err_rounded = np.max(np.abs(Rec_check - 1.0))
                
                if max_err_rounded < 1e-10:
                    best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    best_rank = r
                    break  # Found valid solution for this rank, try lower rank
    
    return best_result
