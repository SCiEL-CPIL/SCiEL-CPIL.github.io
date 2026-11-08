import numpy as np
import time
from scipy.optimize import minimize
import sys

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i, j, k] is 1 if we are computing C[i,k] from A[i,j] * B[j,k]
    # In the trace convention W[k*n+i, r], the tensor index is (i, j, k) -> (k*n+i, j*p+k, i*m+j)?
    # Let's re-read the convention carefully.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Standard tensor decomposition: T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Here, indices are:
    # Input 1 (A): i (row), j (col) -> flattened index i*m + j?
    # Input 2 (B): j (row), k (col) -> flattened index j*p + k?
    # Output (C): i (row), k (col) -> flattened index k*n + i? (This is the "trace convention" mentioned)
    
    # Let's verify the schoolbook construction in the seed:
    # U[i*m+j, column] = 1
    # V[j*p+k, column] = 1
    # W[k*n+i, column] = 1
    # This implies:
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    # The tensor T is defined by T[i, j, k] = 1.
    # The reconstruction is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For the schoolbook, each (i,j,k) has a unique column r, so the sum is 1.
    
    # We want to find U, V, W of rank R such that this holds.
    
    def get_tensor_flat(U, V, W, R):
        # Compute the flat tensor
        # T_flat[idx] = sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
        # idx_u = i*m + j
        # idx_v = j*p + k
        # idx_w = k*n + i
        
        # This is expensive to compute in a pure Python loop.
        # We can use einsum or matrix operations.
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # Let's create the full tensor T of shape (n, m, p)
        # T[i, j, k] = 0
        # for r in range(R):
        #   u_r = U[:, r]  # shape (n*m,)
        #   v_r = V[:, r]  # shape (m*p,)
        #   w_r = W[:, r]  # shape (p*n,)
        
        # Reshape u_r to (n, m), v_r to (m, p), w_r to (p, n)
        # T += u_r_reshaped * v_r_reshaped * w_r_reshaped? No, it's a product of three vectors.
        # T[i, j, k] += u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        
        # We can vectorize this using numpy.
        T = np.zeros((n, m, p))
        for r in range(R):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            
            # We need to add u_r[i,j] * v_r[j,k] * w_r[k,i] to T[i,j,k]
            # This is an outer product-like operation but with shared indices.
            # T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
            
            # We can compute this by iterating over i, j, k? No, too slow.
            # Let's use einsum.
            # 'ij,jk,ki -> ijk'
            # But w_r is indexed by k*n+i. If we reshape w_r to (p, n), w_r[k, i] is w_r[k*n+i].
            # So yes, w_r.reshape(p, n) gives w_r[k, i].
            
            # T += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
            
            # Let's verify the indices for w_r.
            # Seed: W[k*n+i, column]
            # So W_flat[k*n+i] corresponds to W_reshaped[k, i].
            # So w_r_reshaped = W[:, r].reshape(p, n)
            
            T += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
            
        return T

    def residual_flat(U, V, W, R):
        T = get_tensor_flat(U, V, W, R)
        # The target tensor T_target[i,j,k] is 1 for all i,j,k.
        # Wait, is it?
        # Matrix multiplication C = A * B.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor T is such that C[i,k] = sum_{i,j,k} A[i,j] B[j,k] T[i,j,k].
        # So T[i,j,k] must be 1 for all i,j,k?
        # No.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # This corresponds to T[i,j,k] = 1 if the indices match the summation structure.
        # Actually, the tensor T for matrix multiplication is defined as:
        # T[i, j, k] = 1.
        # Because C[i,k] = sum_j (A[i,j] * B[j,k] * 1).
        # Yes, the tensor of matrix multiplication is the all-ones tensor of size n x m x p.
        
        T_target = np.ones((n, m, p))
        return np.linalg.norm(T - T_target)
    
    # Initial Solution: Schoolbook
    rank_init = n * m * p
    U_init = np.zeros((n * m, rank_init))
    V_init = np.zeros((m * p, rank_init))
    W_init = np.zeros((p * n, rank_init))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[i * m + j, col] = 1
                V_init[j * p + k, col] = 1
                W_init[k * n + i, col] = 1
                col += 1
                
    # We want to try to reduce the rank.
    # Strategy: Start with rank R, optimize, check if we can drop a column.
    # If we start with 125, we can't drop columns easily because they are orthogonal.
    # We need to perturb the initial solution to allow mixing.
    
    # Let's try to find a solution with Rank 124.
    # If we fail, we return 125.
    
    best_rank = rank_init
    best_U = U_init
    best_V = V_init
    best_W = W_init
    
    # Time check
    if time.time() - start_time > timeout:
         return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
         
    # Try to optimize a slightly perturbed version of the schoolbook solution
    # to see if we can collapse two columns into one or reduce rank.
    # This is a heuristic.
    
    # Actually, for n=5, it is highly probable that 125 is the minimal rank or close to it.
    # Searching for rank < 125 is extremely hard.
    # Given the constraints, returning the schoolbook solution is the safest bet to ensure validity and score 1.0.
    # Any attempt to reduce rank that fails will result in an invalid solution (score 0) or timeout (score 0).
    # If we find a valid solution with rank < 125, we win big.
    
    # Let's try one quick optimization step with rank 124.
    # We need to initialize U, V, W of size 124.
    # We can take the first 124 columns of the schoolbook solution.
    # But the residual will be 1 (the missing term).
    # We need to adjust the 124 columns to approximate the missing term.
    
    R_target = 124
    U_opt = U_init[:, :R_target].copy()
    V_opt = V_init[:, :R_target].copy()
    W_opt = W_init[:, :R_target].copy()
    
    # Add some noise to escape local minima
    np.random.seed(42)
    U_opt += np.random.normal(0, 0.1, U_opt.shape)
    V_opt += np.random.normal(0, 0.1, V_opt.shape)
    W_opt += np.random.normal(0, 0.1, W_opt.shape)
    
    x0 = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
    
    def objective(x):
        U_curr = x[:n*m*R_target].reshape(n*m, R_target)
        V_curr = x[n*m*R_target : n*m*R_target + m*p*R_target].reshape(m*p, R_target)
        W_curr = x[n*m*R_target + m*p*R_target :].reshape(p*n, R_target)
        
        # Time check inside objective? No, too slow.
        # Just compute residual.
        T = get_tensor_flat(U_curr, V_curr, W_curr, R_target)
        T_target = np.ones((n, m, p))
        res = T - T_target
        return np.sum(res**2)
        
    # We can't use L-BFGS-B on 15000 variables easily without a long time.
    # n*m*R = 25*124 = 3100
    # m*p*R = 25*124 = 3100
    # p*n*R = 25*124 = 3100
    # Total vars = 9300.
    # This is manageable for L-BFGS-B if the function is fast.
    # get_tensor_flat uses einsum.
    
    try:
        if time.time() - start_time < timeout - 5:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
            
            if result.success or result.fun < 1e-10:
                # Check if it's truly valid
                U_final = result.x[:n*m*R_target].reshape(n*m, R_target)
                V_final = result.x[n*m*R_target : n*m*R_target + m*p*R_target].reshape(m*p, R_target)
                W_final = result.x[n*m*R_target + m*p*R_target :].reshape(p*n, R_target)
                
                T_final = get_tensor_flat(U_final, V_final, W_final, R_target)
                T_target = np.ones((n, m, p))
                if np.linalg.norm(T_final - T_target) < 1e-9:
                    # Round to integers if possible?
                    # The problem allows integer/half-integer entries.
                    # If the values are close to integers, we can round them.
                    # However, the evaluator might check for exact match with float tolerance.
                    # The prompt says "integer/half-integer entries".
                    # So we should round to nearest 0.5.
                    
                    U_final = np.round(U_final * 2) / 2
                    V_final = np.round(V_final * 2) / 2
                    W_final = np.round(W_final * 2) / 2
                    
                    # Re-check validity after rounding
                    T_rounded = get_tensor_flat(U_final, V_final, W_final, R_target)
                    if np.linalg.norm(T_rounded - T_target) < 1e-9:
                        best_rank = R_target
                        best_U = U_final
                        best_V = V_final
                        best_W = W_final
                    else:
                        # Rounding broke it, try without rounding?
                        # The prompt says "integer/half-integer entries".
                        # If we can't satisfy that, the solution is invalid?
                        # "factor matrices ... with integer/half-integer entries"
                        # This implies the output MUST have these entries.
                        # So if we can't round it, it's invalid.
                        pass
                        
    except Exception:
        pass
        
    # If we didn't find a better rank, return the schoolbook solution.
    # The schoolbook solution has integer entries.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
