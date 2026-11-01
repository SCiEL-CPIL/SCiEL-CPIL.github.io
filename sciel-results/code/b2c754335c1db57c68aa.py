import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 4, 4
    
    # Start with the known rank-28 Strassen-based decomposition
    r0 = 28
    U0 = [[0] * r0 for _ in range(n * m)]
    V0 = [[0] * r0 for _ in range(m * p)]
    W0 = [[0] * r0 for _ in range(p * n)]

    def _add_strassen(off, ar0, ac0, br0, bc0, cr0, cc0):
        a00 = ar0 * m + (ac0 + 0)
        a01 = ar0 * m + (ac0 + 1)
        a10 = (ar0 + 1) * m + (ac0 + 0)
        a11 = (ar0 + 1) * m + (ac0 + 1)

        b00 = br0 * p + (bc0 + 0)
        b01 = br0 * p + (bc0 + 1)
        b10 = (br0 + 1) * p + (bc0 + 0)
        b11 = (br0 + 1) * p + (bc0 + 1)

        c00 = cc0 * n + cr0
        c01 = (cc0 + 1) * n + cr0
        c10 = cc0 * n + (cr0 + 1)
        c11 = (cc0 + 1) * n + (cr0 + 1)

        U0[a00][off] += 1
        U0[a11][off] += 1
        V0[b00][off] += 1
        V0[b11][off] += 1
        W0[c00][off] += 1
        W0[c11][off] += 1

        U0[a10][off + 1] += 1
        U0[a11][off + 1] += 1
        V0[b00][off + 1] += 1
        W0[c10][off + 1] += 1
        W0[c11][off + 1] -= 1

        U0[a00][off + 2] += 1
        V0[b01][off + 2] += 1
        V0[b11][off + 2] -= 1
        W0[c01][off + 2] += 1
        W0[c11][off + 2] += 1

        U0[a11][off + 3] += 1
        V0[b10][off + 3] += 1
        V0[b00][off + 3] -= 1
        W0[c00][off + 3] += 1
        W0[c10][off + 3] += 1

        U0[a00][off + 4] += 1
        U0[a01][off + 4] += 1
        V0[b11][off + 4] += 1
        W0[c00][off + 4] -= 1
        W0[c01][off + 4] += 1

        U0[a10][off + 5] += 1
        U0[a00][off + 5] -= 1
        V0[b00][off + 5] += 1
        V0[b01][off + 5] += 1
        W0[c11][off + 5] += 1

        U0[a01][off + 6] += 1
        U0[a11][off + 6] -= 1
        V0[b10][off + 6] += 1
        V0[b11][off + 6] += 1
        W0[c00][off + 6] += 1

    _add_strassen(0, 0, 0, 0, 0, 0, 0)
    _add_strassen(7, 0, 2, 2, 0, 0, 0)
    _add_strassen(14, 0, 0, 0, 2, 0, 2)
    _add_strassen(21, 0, 2, 2, 2, 0, 2)

    # Convert to numpy arrays for optimization
    U_np = np.array(U0, dtype=float)
    V_np = np.array(V0, dtype=float)
    W_np = np.array(W0, dtype=float)

    # Precompute the target tensor T_{i,j,k} = delta_{i,k} (for C_{i,k} = sum_j A_{i,j} B_{j,k})
    # Tensor dimensions: (n*m, m*p, p*n) = (8, 16, 8)
    # Index mapping:
    # U index: i*m + j  (i in 0..n-1, j in 0..m-1)
    # V index: j*p + k  (j in 0..m-1, k in 0..p-1)
    # W index: k*n + i  (k in 0..p-1, i in 0..n-1)
    
    # We want to check if U @ diag(lambda) @ V^T reconstructs the tensor when contracted with W
    # Actually, the tensor T is defined by T[a_idx, b_idx, c_idx] = 1 if the indices are consistent, 0 otherwise
    # a_idx = i*m + j
    # b_idx = j*p + k
    # c_idx = k*n + i
    # So T[a_idx, b_idx, c_idx] = 1 if (a_idx % m == b_idx // p) and (b_idx % p == c_idx // n) and (a_idx // m == c_idx % n)
    
    # Let's build the target tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # The decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We want to minimize the rank by finding a rank-r decomposition that is close to T
    # But we already have a valid rank-28 decomposition. Can we do better?
    
    # Let's try to find a rank-24 or rank-27 decomposition using optimization
    # The objective is to minimize || T - sum_r U[:,r] V[:,r]^T W[:,r] ||_F^2
    # This is a non-convex problem. We can use gradient-based optimization.
    
    # Let's try rank 24 first, then 25, 26, 27
    # Time budget: 120s total, so we need to be careful
    
    start_time = time.time()
    time_limit = 100  # Leave 20s margin
    
    best_rank = 28
    best_U = U_np
    best_V = V_np
    best_W = W_np
    
    # Try to find lower rank decompositions
    for target_rank in [24, 25, 26, 27]:
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with random values, scaled
        np.random.seed(42)
        U_r = np.random.randn(n*m, target_rank) * 0.1
        V_r = np.random.randn(m*p, target_rank) * 0.1
        W_r = np.random.randn(p*n, target_rank) * 0.1
        
        # Flatten for optimization
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            size_U = n*m * target_rank
            size_V = m*p * target_rank
            U_flat = x[:size_U]
            V_flat = x[size_U:size_U+size_V]
            W_flat = x[size_U+size_V:]
            return U_flat.reshape(n*m, target_rank), V_flat.reshape(m*p, target_rank), W_flat.reshape(p*n, target_rank)
        
        def objective(x):
            U_r, V_r, W_r = unflatten(x)
            # Compute tensor reconstruction
            # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # This is the outer product of U, V, W along the rank dimension
            # T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            return np.sum((T - T_recon)**2)
        
        def gradient(x):
            U_r, V_r, W_r = unflatten(x)
            # Gradient of sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
            # d/dU[a,r] = -2 sum_{b,c} (T[a,b,c] - T_recon[a,b,c]) V[b,r] W[c,r]
            T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            diff = T - T_recon
            
            dU = -2 * np.einsum('abc,br,cr->ar', diff, V_r, W_r)
            dV = -2 * np.einsum('abc,ar,cr->br', diff, U_r, W_r)
            dW = -2 * np.einsum('abc,ar,br->cr', diff, U_r, V_r)
            
            return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        x0 = flatten(U_r, V_r, W_r)
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500})
            
            if time.time() - start_time > time_limit:
                break
                
            final_obj = result.fun
            if final_obj < 1e-6:
                # Found a valid decomposition
                U_r, V_r, W_r = unflatten(result.x)
                best_rank = target_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break  # Found the lowest possible rank we're trying
        except Exception:
            pass
    
    # Convert back to list format
    if best_rank < 28:
        # Try to round to integers/half-integers
        U_out = np.round(best_U * 2) / 2
        V_out = np.round(best_V * 2) / 2
        W_out = np.round(best_W * 2) / 2
        
        # Verify
        T_recon = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
        if np.max(np.abs(T - T_recon)) < 1e-9:
            # Valid
            U_list = U_out.tolist()
            V_list = V_out.tolist()
            W_list = W_out.tolist()
        else:
            # Fallback to original
            U_list = U0
            V_list = V0
            W_list = W0
    else:
        U_list = U0
        V_list = V0
        W_list = W0
    
    return {"U": U_list, "V": V_list, "W": W_list}
