import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_u = n * m # 8
    dim_v = m * p # 16
    dim_w = p * n # 8
    
    # Target tensor T
    # T is a 3D array of shape (dim_u, dim_v, dim_w)
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    
    # Populate T
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Try ranks from 26 up to 32
    # We want the lowest rank.
    # Use a randomized local search / gradient descent approach.
    # Since the problem is small, we can try a few restarts.
    
    def find_decomposition(rank):
        # Initialize U, V, W randomly
        # Scale initialization to help convergence
        best_error = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        # Number of restarts depends on time budget
        # Let's allow ~10 seconds per rank attempt
        start_time = time.time()
        max_restarts = 10
        max_iter = 1000
        lr = 0.01
        
        for restart in range(max_restarts):
            if time.time() - start_time > 10:
                break
                
            U = np.random.randn(dim_u, rank) * 0.1
            V = np.random.randn(dim_v, rank) * 0.1
            W = np.random.randn(dim_w, rank) * 0.1
            
            # Gradient descent
            # Error E = sum (T - U V W)^2
            # We can update one matrix at a time (Alternating Least Squares or Gradient)
            # ALS is more stable.
            # Fix V, W, update U:
            # U_{u,c} = sum_{v,w} T_{u,v,w} V_{v,c} W_{w,c} / sum_{v,w} V_{v,c}^2 W_{w,c}^2
            # This is linear least squares for each column c of U.
            
            for it in range(max_iter):
                if time.time() - start_time > 10:
                    break
                    
                # Update U
                for c in range(rank):
                    # Construct RHS for U[:, c]
                    # U[:, c] = argmin || T[:,:,c_part] - U[:,c] V[:,c] W[:,c] ||
                    # This is not quite right because V[:,c] and W[:,c] are vectors.
                    # The term is U[:,c] (V[:,c] outer W[:,c])
                    # Let M_c = V[:,c] outer W[:,c] (shape dim_v x dim_w)
                    # We want U[:,c] such that sum_c U[:,c] M_c approx T
                    # This is coupled. ALS usually fixes two and solves for the third.
                    # Fix V, W. Solve for U.
                    # T = U V W = sum_c U[:,c] V[:,c] W[:,c]^T ? No.
                    # T_{u,v,w} = sum_c U_{u,c} V_{v,c} W_{w,c}
                    # For fixed c, the contribution to T is U[:,c] (V[:,c] outer W[:,c])
                    # Let A_c = V[:,c] outer W[:,c]
                    # We want U such that sum_c U[:,c] A_c approx T
                    # This is a linear system for U if we vectorize.
                    # But A_c depends on c.
                    # Standard ALS for CP decomposition:
                    # Fix V, W. For each column c of U, the contribution is U[:,c] * (V[:,c] outer W[:,c])
                    # The residual for column c is T - sum_{c' != c} U[:,c'] (V[:,c'] outer W[:,c'])
                    # We solve min_U || Residual_c - U[:,c] (V[:,c] outer W[:,c]) ||_F
                    # This is a least squares problem: U[:,c] (V[:,c] outer W[:,c]) approx Residual_c
                    # Let M_c = V[:,c] outer W[:,c]
                    # We want vec(U[:,c] M_c) = vec(Residual_c)
                    # vec(U[:,c] M_c) = (M_c^T \otimes I) vec(U[:,c])
                    # So we solve (M_c^T \otimes I) vec(U[:,c]) = vec(Residual_c)
                    
                    # This is computationally heavy for every iteration.
                    # Given the small size (8x16x8), it might be okay.
                    
                    pass # Skipping detailed ALS implementation for brevity in thought, will use simpler gradient or precomputed solution.
            
            # If we can't implement ALS quickly, use a known solution if available.
            # Or, just return the schoolbook rank 32 if search fails.
            
            # Let's try a different approach:
            # The rank 26 solution for 2x4x4 is known.
            # Can I hardcode it?
            # No, I don't have it memorized.
            
            # Let's try a simple heuristic: 
            # If we can't find a lower rank, return 32.
            # But 32 is the baseline. We need to beat it.
            
            # Is there a rank 28 solution?
            # 2x2x2 is 7.
            # 2x4x4 can be seen as 4 blocks of 2x2x2?
            # A = [A1 A2], B = [B11 B12; B21 B22]
            # C1 = A1 B11 + A2 B21
            # C2 = A1 B12 + A2 B22
            # Each 2x2 mult is rank 7.
            # Total 28.
            # Can we share computations?
            # A1 B11 and A1 B12 share A1.
            # A2 B21 and A2 B22 share A2.
            # This structure might allow rank reduction below 28.
            
            # Let's try to find a rank 26 solution using a specific library or algorithm?
            # No external libs other than numpy/scipy.
            
            # I will implement a simple local search using gradient descent on the flattened parameters.
            # Parameters: U, V, W flattened.
            # Loss: || T - U V W ||_F^2
            # Gradient calculation is feasible.
            
            def loss_and_grad(params):
                # Unpack
                U = params[0:dim_u*rank].reshape(dim_u, rank)
                V = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
                W = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
                
                # Compute product T_calc
                # T_calc[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
                # Efficient computation:
                # T_calc = np.einsum('uc,vc,wc->uvw', U, V, W)
                T_calc = np.einsum('uc,vc,wc->uvw', U, V, W)
                
                diff = T - T_calc
                loss = np.sum(diff**2)
                
                # Gradient
                # dL/dU[u,c] = -2 * sum_{v,w} diff[u,v,w] V[v,c] W[w,c]
                # dL/dV[v,c] = -2 * sum_{u,w} diff[u,v,w] U[u,c] W[w,c]
                # dL/dW[w,c] = -2 * sum_{u,v} diff[u,v,w] U[u,c] V[v,c]
                
                grad_U = -2.0 * np.einsum('uvw,vc,wc->uc', diff, V, W)
                grad_V = -2.0 * np.einsum('uvw,uc,wc->vc', diff, U, W)
                grad_W = -2.0 * np.einsum('uvw,uc,vc->wc', diff, U, V)
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return loss, grad

            # Simple gradient descent with momentum
            for restart in range(5):
                if time.time() - start_time > 10:
                    break
                params = np.random.randn(dim_u*rank + dim_v*rank + dim_w*rank) * 0.1
                v_mom = np.zeros_like(params)
                lr = 0.001
                for it in range(2000):
                    loss, grad = loss_and_grad(params)
                    if loss < 1e-10:
                        break
                    v_mom = 0.9 * v_mom - lr * grad
                    params = params + v_mom
                    if it % 100 == 0:
                        # Check time
                        if time.time() - start_time > 10:
                            break
                
                # Check if solution is valid
                U = params[0:dim_u*rank].reshape(dim_u, rank)
                V = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
                W = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
                T_calc = np.einsum('uc,vc,wc->uvw', U, V, W)
                if np.allclose(T_calc, T, atol=1e-6):
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    # Verify rounded solution
                    T_r = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r)
                    if np.allclose(T_r, T, atol=1e-9):
                        return U_r, V_r, W_r, loss
        
        return None, None, None, np.inf

    # Try ranks
    for r in range(26, 33):
        U, V, W, err = find_decomposition(r)
        if U is not None:
            # Convert to lists of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    rank = 32
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
