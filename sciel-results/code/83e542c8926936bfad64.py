"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 120
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def compute_error(U, V, W, n, m, p):
        """Compute max error of decomposition."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def smooth_loss(U, V, W, n, m, p):
        """Sum of squared errors."""
        rank = U.shape[1]
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    total_err += (val - 1.0) ** 2
        return total_err
    
    def compute_gradients(U, V, W, n, m, p):
        """Compute gradients of smooth loss w.r.t. U, V, W."""
        rank = U.shape[1]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err = 2.0 * (val - 1.0)
                    
                    # d/dU[i*m+j, r] = err * V[j*p+k, r] * W[k*n+i, r]
                    dU[i*m+j, :] += err * v_row * w_row
                    # d/dV[j*p+k, r] = err * U[i*m+j, r] * W[k*n+i, r]
                    dV[j*p+k, :] += err * u_row * w_row
                    # d/dW[k*n+i, r] = err * U[i*m+j, r] * V[j*p+k, r]
                    dW[k*n+i, :] += err * u_row * v_row
        
        return dU, dV, dW
    
    # Strategy: Try to find lower rank decompositions using gradient-based optimization
    # with multiple restarts and adaptive learning rates
    
    target_ranks = [30, 32, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random initializations
        n_restarts = 5
        found = False
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if found:
                break
                
            np.random.seed(42 + restart * 1000 + target_rank)
            
            # Initialize with small random values
            scale = 0.1
            U_try = np.random.randn(n*m, target_rank) * scale
            V_try = np.random.randn(m*p, target_rank) * scale
            W_try = np.random.randn(p*n, target_rank) * scale
            
            # Adaptive gradient descent
            lr = 0.05
            best_loss = smooth_loss(U_try, V_try, W_try, n, m, p)
            patience = 0
            max_patience = 200
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                    
                loss = smooth_loss(U_try, V_try, W_try, n, m, p)
                
                if loss < 1e-10:
                    # Found valid decomposition
                    if verify_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-8):
                        best_U = U_try.copy()
                        best_V = V_try.copy()
                        best_W = W_try.copy()
                        best_rank = target_rank
                        found = True
                        break
                
                if loss < best_loss:
                    best_loss = loss
                    patience = 0
                    # Reduce learning rate slightly
                    lr *= 0.99
                else:
                    patience += 1
                    if patience > max_patience:
                        break
                
                # Compute gradients
                dU, dV, dW = compute_gradients(U_try, V_try, W_try, n, m, p)
                
                # Gradient step
                U_try -= lr * dU
                V_try -= lr * dV
                W_try -= lr * dW
                
                # Occasionally add small random perturbation to escape local minima
                if step % 100 == 0 and step > 0:
                    noise_scale = 0.01 * (1.0 - step / 2000.0)
                    U_try += np.random.randn(n*m, target_rank) * noise_scale
                    V_try += np.random.randn(m*p, target_rank) * noise_scale
                    W_try += np.random.randn(p*n, target_rank) * noise_scale
            
            if found:
                break
    
    # If we found a better rank, try to polish with higher precision
    if best_rank < schoolbook_rank:
        if time.time() < deadline - 10:
            # Try to verify with tighter tolerance and polish
            for _ in range(50):
                if time.time() > deadline:
                    break
                loss = smooth_loss(best_U, best_V, best_W, n, m, p)
                if loss < 1e-15:
                    break
                dU, dV, dW = compute_gradients(best_U, best_V, best_W, n, m, p)
                lr = 0.001
                best_U -= lr * dU
                best_V -= lr * dV
                best_W -= lr * dW
    
    # Convert to list of lists with rounding
    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]
    
    # Clean up near-zero and near-integer/half-integer values
    def clean_list(lst):
        for row in lst:
            for idx in range(len(row)):
                v = row[idx]
                if abs(v) < 1e-10:
                    row[idx] = 0.0
                elif abs(v - round(v)) < 1e-10:
                    row[idx] = float(round(v))
                elif abs(v - round(v*2)/2) < 1e-10:
                    row[idx] = round(v*2)/2
        return lst
    
    U_out = clean_list(to_list(best_U))
    V_out = clean_list(to_list(best_V))
    W_out = clean_list(to_list(best_W))
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
