import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    # Dimensions of the factor matrices
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T (dense)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Function to check if U, V, W decompose T exactly
    # with half-integer entries
    def check_exact(U, V, W):
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Reconstruct tensor
        # T_rec = sum_r U_r[:, r] * V_r[:, r] * W_r[:, r]
        # Use einsum for speed
        T_rec = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        
        # Check exact equality
        return np.allclose(T_rec, T, atol=1e-9)

    def als_step(U, V, W, T, R):
        # Update U
        # U[a, r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / sum_{b,c} V[b,r]^2 W[c,r]^2
        # We can precompute V_r * W_r? No, depends on r.
        # For each r, U[:, r] is a linear solve? No, it's a direct sum.
        # U[:, r] = sum_{b,c} T[:, b, c] * V[b, r] * W[c, r]
        # This is equivalent to:
        # Temp = sum_r (V[:, r] * W[:, r])_{b,c} * T[:, b, c] ... no.
        
        # Efficient update:
        # For each r:
        #   vec = V[:, r] * W[:, r]  # size dim_v * dim_w
        #   U[:, r] = np.tensordot(T, vec, axes=([1, 2], [0, 1]))
        
        # This is slow if done in a loop over R.
        # We can use einsum.
        # U_new = np.einsum('abc, br, cr -> ar', T, V, W)
        # Denominator: D_r = sum_{b,c} V[b,r]^2 W[c,r]^2
        # D = np.einsum('br, cr -> r', V**2, W**2)
        # U_new = U_new / (D + 1e-12)[np.newaxis, :]
        
        U_new = np.einsum('abc, br, cr -> ar', T, V, W)
        D_u = np.einsum('br, cr -> r', V**2, W**2)
        U_new /= (D_u + 1e-12)[np.newaxis, :]
        
        V_new = np.einsum('abc, ar, cr -> br', T, U_new, W)
        D_v = np.einsum('ar, cr -> r', U_new**2, W**2)
        V_new /= (D_v + 1e-12)[np.newaxis, :]
        
        W_new = np.einsum('abc, ar, br -> cr', T, U_new, V_new)
        D_w = np.einsum('ar, br -> r', U_new**2, V_new**2)
        W_new /= (D_w + 1e-12)[np.newaxis, :]
        
        return U_new, V_new, W_new

    # Search for minimal rank
    # Lower bound is max(dim_u, dim_v, dim_w) = 20
    # Try ranks from 20 to 60
    best_solution = None
    
    for R in range(20, 61):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Initialize
            np.random.seed(restart + R)
            U = np.random.randn(dim_u, R)
            V = np.random.randn(dim_v, R)
            W = np.random.randn(dim_w, R)
            
            # Run ALS
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                U, V, W = als_step(U, V, W, T, R)
                
                # Check for exact solution
                if check_exact(U, V, W):
                    # Found a solution
                    # Convert to list of lists as required
                    U_list = np.round(U * 2) / 2
                    V_list = np.round(V * 2) / 2
                    W_list = np.round(W * 2) / 2
                    return {"U": U_list.tolist(), "V": V_list.tolist(), "W": W_list.tolist()}
            
            # If not found in 50 iterations, try next restart
            pass

    # If no better solution found, return schoolbook
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
