import numpy as np
import time
import random

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target ranks to try
    target_ranks = [17, 18]
    
    # Construct the target tensor T
    # T[i, j, k] = 1 if we are computing C[i, k] from A[i, j] and B[j, k]
    # The tensor is defined by the indices of the non-zero terms in the multiplication.
    # T is a 3D array of shape (n, m, p)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The term A[i, j] * B[j, k] contributes to C[i, k]
                # In the tensor decomposition T = sum u_r v_r w_r,
                # T[i, j, k] is the coefficient.
                # For matrix multiplication, T[i, j, k] = 1 for all valid i, j, k.
                # Wait, is it?
                # C[i, k] = sum_j A[i, j] B[j, k]
                # So the tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k] * 1
                # Yes, all 20 entries are 1.
                T[i, j, k] = 1.0

    # Flatten T for optimization
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin

    best_result = None
    best_rank = n * m * p # 20

    # Helper to verify and convert to list of lists
    def check_and_convert(U, V, W, rank):
        # Check if entries are close to integers or half-integers
        # and if the decomposition is exact.
        # U shape: (n*m, rank)
        # V shape: (m*p, rank)
        # W shape: (p*n, rank)
        
        # Reshape for tensor product
        # U[i*m + j, r]
        # V[j*p + k, r]
        # W[k*n + i, r]
        
        # Reconstruct T from U, V, W
        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            
            # u is length n*m
            # v is length m*p
            # w is length p*n
            
            for i in range(n):
                for j in range(m):
                    u_val = u[i*m + j]
                    if abs(u_val) < 1e-9: continue
                    for k in range(p):
                        v_val = v[j*p + k]
                        if abs(v_val) < 1e-9: continue
                        w_val = w[k*n + i]
                        T_recon[i, j, k] += u_val * v_val * w_val
                        
        # Check exactness
        if not np.allclose(T_recon, T, atol=1e-6):
            return False, None, None, None

        # Check if entries are integer/half-integer
        # We try to round to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Re-check with rounded values
        T_recon2 = np.zeros((n, m, p))
        for r in range(rank):
            u = U_rounded[:, r]
            v = V_rounded[:, r]
            w = W_rounded[:, r]
            for i in range(n):
                for j in range(m):
                    u_val = u[i*m + j]
                    if abs(u_val) < 1e-9: continue
                    for k in range(p):
                        v_val = v[j*p + k]
                        if abs(v_val) < 1e-9: continue
                        w_val = w[k*n + i]
                        T_recon2[i, j, k] += u_val * v_val * w_val
                        
        if not np.allclose(T_recon2, T, atol=1e-6):
            return False, None, None, None
            
        return True, U_rounded, V_rounded, W_rounded

    def run_als(rank, max_iter=500):
        # Initialize U, V, W randomly
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to avoid scaling issues
        # We can fix the scale of one factor, e.g., make norm of each column 1
        # But ALS handles scaling automatically if we alternate.
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            # For each column r, we want to minimize || T - sum_{r'} u_{r'} v_{r'} w_{r'} ||
            # This is equivalent to for each r, U[:, r] = argmin || T_flat - sum_{r' != r} ... ||
            # Actually, we can update each factor by solving a least squares problem.
            
            # Let's use a simple gradient step or ALS.
            # ALS:
            # Fix V, W. Update U.
            # The term involving U[:, r] is U[:, r] * (V * W_r)
            # Where V * W_r is the outer product of V[:, r] and W[:, r]? No.
            # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            # For a fixed r, the contribution is U[:, r] \otimes V[:, r] \otimes W[:, r]
            # This is not a simple linear update for U unless we flatten the other two.
            
            # Let's flatten V and W for each r?
            # V_r is a vector of size m*p. W_r is a vector of size p*n.
            # The product V_r \otimes W_r is not directly a matrix that multiplies U_r.
            # The tensor product is 3-way.
            
            # We can use the fact that for fixed V, W, the map U -> T is linear.
            # T = sum_r U_r \otimes V_r \otimes W_r
            # We can solve for U_r by least squares if we treat the other terms as noise.
            
            # Let's try a simpler approach: Gradient Descent on flattened parameters.
            # This might be slow but robust for small sizes.
            
            pass

    # Instead of complex ALS, let's use scipy.optimize.least_squares on the flattened parameters.
    # Parameters: x = [U.flatten(), V.flatten(), W.flatten()]
    # Size: 24 * rank.
    # For rank 18, size 432.
    # Residual: T_flat - tensor_product(x)
    
    def tensor_product_from_params(x, rank):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
        
        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    u_val = u[i*m + j]
                    if abs(u_val) < 1e-9: continue
                    for k in range(p):
                        v_val = v[j*p + k]
                        if abs(v_val) < 1e-9: continue
                        w_val = w[k*n + i]
                        T_recon[i, j, k] += u_val * v_val * w_val
        return T_recon.flatten() - T_flat

    for rank in target_ranks:
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for restart in range(50):
            if time.time() > deadline:
                break
                
            x0 = np.random.randn(24 * rank)
            
            # Use least_squares
            try:
                res = scipy.optimize.least_squares(
                    tensor_product_from_params, 
                    x0, 
                    args=(rank,), 
                    method='lm', # Levenberg-Marquardt, good for small problems
                    max_nfev=2000
                )
            except:
                continue
                
            if res.cost < 1e-10:
                # Found a real solution
                U, V, W = res.x[:n*m*rank].reshape(n*m, rank), \
                         res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank), \
                         res.x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
                
                is_valid, U_rounded, V_rounded, W_rounded = check_and_convert(U, V, W, rank)
                if is_valid:
                    # Convert to lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return schoolbook
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

import scipy.optimize
