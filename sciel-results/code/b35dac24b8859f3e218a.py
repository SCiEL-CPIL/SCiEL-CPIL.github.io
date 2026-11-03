import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T
    # Dimensions: (n*m, m*p, p*n)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    T_flat = T.flatten()
    
    # Function to compute the tensor product sum for given U, V, W
    def tensor_product(U, V, W):
        # U: (dim_u, R), V: (dim_v, R), W: (dim_w, R)
        # Result: (dim_u, dim_v, dim_w)
        # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # We can compute this efficiently using einsum or loops
        # R = U.shape[1]
        # return np.einsum('ar,br,cr->abc', U, V, W)
        return np.einsum('ar,br,cr->abc', U, V, W)

    def cost(params, R):
        # params is a flat array of size (dim_u + dim_v + dim_w) * R
        # Reshape to U, V, W
        size_u = dim_u * R
        size_v = dim_v * R
        size_w = dim_w * R
        
        U = params[:size_u].reshape((dim_u, R))
        V = params[size_u:size_u+size_v].reshape((dim_v, R))
        W = params[size_u+size_v:].reshape((dim_w, R))
        
        prod = tensor_product(U, V, W)
        diff = prod.flatten() - T_flat
        return np.dot(diff, diff)

    # Trivial solution for fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((dim_u, rank_trivial), dtype=int)
    V_trivial = np.zeros((dim_v, rank_trivial), dtype=int)
    W_trivial = np.zeros((dim_w, rank_trivial), dtype=int)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1

    # Try to find lower rank solutions
    # We try ranks from 16 up to 23
    # Using random restarts
    
    best_solution = None
    
    for R in range(16, rank_trivial):
        found = False
        # Limit time per R, e.g., 5 attempts of 1s each?
        # We'll do a few random starts
        for attempt in range(5):
            # Random initialization
            np.random.seed(attempt + R)
            U_init = np.random.randn(dim_u, R)
            V_init = np.random.randn(dim_v, R)
            W_init = np.random.randn(dim_w, R)
            
            # Scale down to help convergence?
            U_init *= 0.5
            V_init *= 0.5
            W_init *= 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use BFGS or L-BFGS-B
            try:
                res = minimize(cost, params_init, args=(R,), method='L-BFGS-B', options={'maxiter': 100})
                if res.fun < 1e-6:
                    # Found a real solution
                    params_opt = res.x
                    size_u = dim_u * R
                    size_v = dim_v * R
                    size_w = dim_w * R
                    
                    U_real = params_opt[:size_u].reshape((dim_u, R))
                    V_real = params_opt[size_u:size_u+size_v].reshape((dim_v, R))
                    W_real = params_opt[size_u+size_v:].reshape((dim_w, R))
                    
                    # Try to find integer/half-integer solution
                    # Check nearest integers
                    U_int = np.round(U_real).astype(int)
                    V_int = np.round(V_real).astype(int)
                    W_int = np.round(W_real).astype(int)
                    
                    if np.allclose(tensor_product(U_int, V_int, W_int).flatten(), T_flat, atol=1e-9):
                        best_solution = (R, U_int, V_int, W_int)
                        found = True
                        break
                    
                    # Check nearest half-integers (multiply by 2, round, divide by 2)
                    # But the output must be integer or half-integer.
                    # The interface accepts integer/half-integer.
                    # If we use half-integers, we can just return them as floats?
                    # "integer/half-integer entries"
                    # Let's check if U_real * 2 is close to integers.
                    U_half = np.round(U_real * 2) / 2.0
                    V_half = np.round(V_real * 2) / 2.0
                    W_half = np.round(W_real * 2) / 2.0
                    
                    if np.allclose(tensor_product(U_half, V_half, W_half).flatten(), T_flat, atol=1e-9):
                        # Convert to a type that represents half-integers clearly?
                        # The return type is a dict of lists.
                        # If they are half-integers, they are floats like 0.5, 1.0, -1.5.
                        best_solution = (R, U_half, V_half, W_half)
                        found = True
                        break
                        
            except Exception:
                pass

        if found:
            break
            
    if best_solution:
        R, U, V, W = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to trivial solution
        return {
            "U": U_trivial.tolist(),
            "V": V_trivial.tolist(),
            "W": W_trivial.tolist()
        }
# EVOLVE-BLOCK-END
