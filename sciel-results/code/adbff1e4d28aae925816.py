import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110  # 10s margin
    
    # Target tensor T of shape (n, m, p) with all 1s
    # T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Function to compute the tensor from U, V, W
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    # Tensor[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    def compute_tensor(U, V, W):
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is a contraction.
        # Let's use einsum
        # 'ijr,jkr,kir -> ijk'
        return np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    def error(U, V, W):
        T_calc = compute_tensor(U, V, W)
        return np.sum((T_calc - T)**2)
    
    def is_half_integer(x):
        return np.allclose(x * 2, np.round(x * 2), atol=1e-6)
        
    def round_to_half(x):
        return np.round(x * 2) / 2.0

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find a low rank decomposition
    # Start from low rank, increase if not found
    # Rank of J_2,2,2 is 7. J_3,5,7 might be around 20-30?
    # Let's try up to 50.
    
    for r in range(10, 51):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Scale to help convergence?
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Simple gradient descent
            # Or use scipy?
            # Let's use a simple iterative method.
            # Since the function is polynomial, gradient descent might be slow.
            # But for small r, it might work.
            
            # Actually, let's try to use the fact that we can fix U and V and solve for W?
            # No, it's trilinear.
            
            # Let's use scipy.optimize.minimize if available, else custom.
            # Custom gradient descent:
            lr = 0.01
            prev_err = float('inf')
            for step in range(200):
                if time.time() > deadline:
                    break
                    
                T_calc = compute_tensor(U, V, W)
                err = np.sum((T_calc - T)**2)
                
                if err < 1e-9:
                    # Check half-integer
                    if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                        U_r = round_to_half(U)
                        V_r = round_to_half(V)
                        W_r = round_to_half(W)
                        # Verify exactness
                        T_ver = compute_tensor(U_r, V_r, W_r)
                        if np.allclose(T_ver, T, atol=1e-6):
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    else:
                        # Try to snap to half-integers and check
                        U_r = round_to_half(U)
                        V_r = round_to_half(V)
                        W_r = round_to_half(W)
                        T_ver = compute_tensor(U_r, V_r, W_r)
                        if np.allclose(T_ver, T, atol=1e-6):
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
                # Compute gradients
                # d(err)/d(U[i,j,r]) = 2 * sum_{k} (T_calc[i,j,k] - T[i,j,k]) * V[j,k,r] * W[k,i,r]
                # This is expensive.
                # Let's just break if we don't improve.
                if err < prev_err - 1e-6:
                    prev_err = err
                else:
                    break
                
                # Simple gradient step (approximate)
                # For speed, we might skip actual gradient calculation and just use random perturbations?
                # No, that's inefficient.
                
                # Let's use a more robust method: ALS (Alternating Least Squares)
                # Fix U, V, solve for W?
                # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # For fixed U, V, this is linear in W.
                # We can solve for W column by column? No, W is coupled.
                # But we can minimize sum_{i,j,k} (T[i,j,k] - sum_r U[i,j,r] V[j,k,r] W[k,i,r])^2
                # This is a linear system for W.
                
                # Let's implement ALS.
                
                # 1. Fix U, V. Solve for W.
                # Reshape U, V to (n, m, r), (m, p, r).
                # We want to find W (p, n, r) to minimize error.
                # The term is einsum('ijr,jkr,kir->ijk', U, V, W).
                # This is linear in W.
                # We can flatten W to (p*n, r).
                # The coefficient for W[k, i, r] is U[i, j, r] * V[j, k, r].
                # Sum over j.
                # So for each (k, i, r), the coefficient is sum_j U[i, j, r] V[j, k, r].
                # Let A[k, i, r] = sum_j U[i, j, r] V[j, k, r].
                # Then T_calc[i, j, k] = sum_r A[k, i, r] W[k, i, r]? No.
                # T_calc[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r].
                # This is not simply linear in W[k,i,r] with a fixed coefficient because U and V depend on j.
                # Wait, for a fixed r, the term is U[i,j,r] V[j,k,r] W[k,i,r].
                # If we fix r, it's a product of three terms.
                # ALS usually works for bilinear (matrix) or when one mode is fixed.
                # Here, if we fix U and V, the equation is linear in W.
                # T[i,j,k] = sum_r (U[i,j,r] V[j,k,r]) W[k,i,r].
                # Let C_r[i,j,k] = U[i,j,r] V[j,k,r].
                # Then T[i,j,k] = sum_r C_r[i,j,k] W[k,i,r].
                # This is a linear system in W.
                # We can solve for W using least squares.
                
                # Similarly for U and V.
                
                # Let's implement one step of ALS.
                
                # Solve for W
                # We need to form the matrix for the linear system.
                # Variables: W_flat (p*n*r).
                # Equations: for each i, j, k.
                # Sum_r C_r[i,j,k] W_flat[k*n+i + r*p*n] = T[i,j,k].
                
                # This is a large sparse system.
                # Given the small size (105 equations, ~1000 variables), we can use dense linear algebra.
                
                # Construct matrix A_W of shape (105, p*n*r)
                # A_W[index(i,j,k), index(k,i,r)] = C_r[i,j,k]
                
                # This is feasible.
                
                if step % 3 == 0: # Solve for W
                    A_W = np.zeros((n*m*p, p*n*r))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                eq_idx = i*m*p + j*p + k
                                for r_idx in range(r):
                                    # Coefficient is U[i,j,r_idx] * V[j,k,r_idx]
                                    coeff = U[i*m+j, r_idx] * V[j*p+k, r_idx]
                                    var_idx = k*n*r + i*r + r_idx # W[k, i, r] index in flat W
                                    # W is (p, n, r). Flat index: (k*n + i)*r + r_idx
                                    A_W[eq_idx, (k*n + i)*r + r_idx] = coeff
                    # Solve A_W W_flat = T_flat
                    T_flat = T.flatten()
                    try:
                        W_flat, residuals, rank, s = np.linalg.lstsq(A_W, T_flat, rcond=None)
                        W = W_flat.reshape(p, n, r)
                        W = W.reshape(p*n, r)
                    except:
                        pass
                elif step % 3 == 1: # Solve for V
                    A_V = np.zeros((n*m*p, m*p*r))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                eq_idx = i*m*p + j*p + k
                                for r_idx in range(r):
                                    coeff = U[i*m+j, r_idx] * W[k*n+i, r_idx]
                                    var_idx = (j*p + k)*r + r_idx
                                    A_V[eq_idx, var_idx] = coeff
                    T_flat = T.flatten()
                    try:
                        V_flat, residuals, rank, s = np.linalg.lstsq(A_V, T_flat, rcond=None)
                        V = V_flat.reshape(m*p, r)
                    except:
                        pass
                else: # Solve for U
                    A_U = np.zeros((n*m*p, n*m*r))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                eq_idx = i*m*p + j*p + k
                                for r_idx in range(r):
                                    coeff = V[j*p+k, r_idx] * W[k*n+i, r_idx]
                                    var_idx = (i*m + j)*r + r_idx
                                    A_U[eq_idx, var_idx] = coeff
                    T_flat = T.flatten()
                    try:
                        U_flat, residuals, rank, s = np.linalg.lstsq(A_U, T_flat, rcond=None)
                        U = U_flat.reshape(n*m, r)
                    except:
                        pass
                
                # Check error
                T_calc = compute_tensor(U, V, W)
                err = np.sum((T_calc - T)**2)
                
                if err < 1e-8:
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    T_ver = compute_tensor(U_r, V_r, W_r)
                    if np.allclose(T_ver, T, atol=1e-6):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # If no low rank found, return schoolbook
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
