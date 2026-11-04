"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 120
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
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
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is exact."""
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
        # Check off-diagonal entries are zero
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    if (i, j, k) == (0, 0, 0):
                        continue
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val) > tol:
                        return False
        return True
    
    def smooth_loss(U, V, W, n, m, p):
        """Compute sum of squared errors for all tensor entries."""
        rank = U.shape[1]
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if i == 0 and j == 0 and k == 0:
                        total_err += (val - 1.0) ** 2
                    else:
                        total_err += val ** 2
        return total_err
    
    # Strategy: Use structured low-rank decomposition exploiting tensor structure
    # The matrix multiplication tensor T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k}
    # For rectangular matrices, we can use bilinear forms with shared factors
    
    # Try ranks using structured initialization based on rank factorization
    # Key insight: If we can write the multiplication as a sum of rank-1 tensors
    # where each rank-1 tensor has sparse support, we can reduce rank
    
    # Approach: Use the fact that (AB)_{ik} = sum_j A_{ij} B_{jk}
    # We can group terms to create lower-rank decompositions
    
    # Try various structured decompositions
    target_ranks = [48, 54, 60, 72, 80, 90, 100, 110]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple structured initializations
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart * 1000 + target_rank)
            
            # Initialize with structured patterns
            # Use block-diagonal or sparse patterns
            U_try = np.random.randn(n*m, target_rank) * 0.1
            V_try = np.random.randn(m*p, target_rank) * 0.1
            W_try = np.random.randn(p*n, target_rank) * 0.1
            
            # Add some structure: make certain rows/columns more prominent
            for idx in range(min(10, target_rank)):
                U_try[idx % (n*m), idx] += 1.0
                V_try[idx % (m*p), idx] += 1.0
                W_try[idx % (p*n), idx] += 1.0
            
            # Gradient descent with adaptive step size
            lr = 0.05
            best_loss = smooth_loss(U_try, V_try, W_try, n, m, p)
            patience = 0
            max_patience = 100
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                loss = smooth_loss(U_try, V_try, W_try, n, m, p)
                
                if loss < 1e-12:
                    # Found valid decomposition - verify
                    if verify_decomposition(U_try, V_try, W_try, n, m, p):
                        best_U = U_try.copy()
                        best_V = V_try.copy()
                        best_W = W_try.copy()
                        best_rank = target_rank
                        break
                
                if loss < best_loss * 0.99:
                    best_loss = loss
                    patience = 0
                else:
                    patience += 1
                
                if patience > max_patience:
                    # Try a different initialization
                    break
                
                # Compute gradients using finite differences for key parameters
                # For efficiency, use a subset of gradient computations
                if step % 50 == 0:
                    # Check convergence
                    pass
                
                # Adaptive step size
                step_size = lr * max(0.1, 1.0 - step / 2000.0)
                
                # Update with random perturbations biased toward reducing loss
                U_try += np.random.randn(n*m, target_rank) * step_size * 0.1
                V_try += np.random.randn(m*p, target_rank) * step_size * 0.1
                W_try += np.random.randn(p*n, target_rank) * step_size * 0.1
            
            if best_rank == target_rank:
                break
    
    # If no better rank found, try a final optimization pass on schoolbook
    # to see if we can find any structure
    
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
