import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    n_triplets = len(triplets)
    
    # Pre-compute index arrays for vectorized operations
    tri_i = np.array([t[0] for t in triplets])
    tri_j = np.array([t[1] for t in triplets])
    tri_k = np.array([t[2] for t in triplets])
    
    tri_u = tri_i * m + tri_j
    tri_v = tri_j * p + tri_k
    tri_w = tri_k * n + tri_i
    
    def get_residuals(U, V, W):
        # Vectorized: compute sum over rank for each triplet
        # U[tri_u, :], V[tri_v, :], W[tri_w, :] are (n_triplets, rank)
        res = np.einsum('tr,tr,tr->t', U[tri_u], V[tri_v], W[tri_w]) - 1.0
        return res

    def total_error(U, V, W):
        res = get_residuals(U, V, W)
        return float(np.sum(res * res))

    def optimize(U, V, W, max_iters=500, tol=1e-12, deadline=None):
        # Adaptive Gradient Descent with momentum
        lr = 0.05
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
                
            res = get_residuals(U, V, W)
            err = float(np.sum(res * res))
            if err < tol:
                break
            
            # Compute gradients vectorized
            # dU[u] = 2 * sum_t res[t] * V[tri_v[t], :] * W[tri_w[t], :]
            # We can do this efficiently with einsum or loops over rank
            
            # term_VW[t, r] = V[tri_v[t], r] * W[tri_w[t], r]
            # dU[tri_u[t], r] += 2 * res[t] * term_VW[t, r]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # For each rank component r, compute contributions
            # This is O(n_triplets * rank) which is fine for small sizes
            
            U_t = U[tri_u]  # (n_triplets, rank)
            V_t = V[tri_v]  # (n_triplets, rank)
            W_t = W[tri_w]  # (n_triplets, rank)
            
            # dU[tri_u[t], r] = 2 * res[t] * V_t[t, r] * W_t[t, r]
            # We need to scatter-add these back to dU
            
            # Using np.add.at for scatter addition
            dU_contrib = 2.0 * res[:, None] * V_t * W_t  # (n_triplets, rank)
            dV_contrib = 2.0 * res[:, None] * U_t * W_t
            dW_contrib = 2.0 * res[:, None] * U_t * V_t
            
            np.add.at(dU, tri_u, dU_contrib)
            np.add.at(dV, tri_v, dV_contrib)
            np.add.at(dW, tri_w, dW_contrib)
            
            # Momentum update
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            
            U += vU
            V += vV
            W += vW
            
            # Adaptive learning rate
            if err > prev_err:
                lr *= 0.5
            else:
                lr = min(lr * 1.05, 0.1)
            prev_err = err
            
        return U, V, W

    def polish(U, V, W, max_iters=1000, tol=1e-14, deadline=None):
        # High-precision polish with smaller learning rate
        lr = 0.01
        momentum = 0.95
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
                
            res = get_residuals(U, V, W)
            err = float(np.sum(res * res))
            if err < tol:
                break
            
            U_t = U[tri_u]
            V_t = V[tri_v]
            W_t = W[tri_w]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            dU_contrib = 2.0 * res[:, None] * V_t * W_t
            dV_contrib = 2.0 * res[:, None] * U_t * W_t
            dW_contrib = 2.0 * res[:, None] * U_t * V_t
            
            np.add.at(dU, tri_u, dU_contrib)
            np.add.at(dV, tri_v, dV_contrib)
            np.add.at(dW, tri_w, dW_contrib)
            
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            
            U += vU
            V += vV
            W += vW
            
            if err > prev_err:
                lr *= 0.5
            prev_err = err
            
        return U, V, W

    def schoolbook(rank=None):
        if rank is None:
            rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[u_idx(i, j), col] = 1.0
                    V[v_idx(j, k), col] = 1.0
                    W[w_idx(k, i), col] = 1.0
                    col += 1
                if col >= rank:
                    break
            if col >= rank:
                break
        return U, V, W

    def randomized_algebraic_search(max_restarts=200, max_iters=200):
        best_rank = n * m * p
        best_U, best_V, best_W = schoolbook()
        best_err = 0.0
        
        start_time = time.time()
        deadline = 140  # 10s margin before 150s kill
        
        # Try ranks from 40 to 48
        for r in range(40, 49):
            if time.time() > deadline:
                break
            
            for restart in range(max_restarts):
                if time.time() > deadline:
                    break
                
                # Initialize with schoolbook terms (first r)
                U_init, V_init, W_init = schoolbook(r)
                
                # Add small noise to help escape local minima
                noise_level = 0.1
                U_init = U_init + noise_level * np.random.randn(n*m, r)
                V_init = V_init + noise_level * np.random.randn(m*p, r)
                W_init = W_init + noise_level * np.random.randn(p*n, r)
                
                # Run optimization
                U_opt, V_opt, W_opt = optimize(U_init, V_init, W_init, max_iters=max_iters, deadline=deadline)
                
                err = total_error(U_opt, V_opt, W_opt)
                
                if err < 1e-6:
                    # Polish for higher precision
                    U_opt, V_opt, W_opt = polish(U_opt, V_opt, W_opt, max_iters=500, tol=1e-14, deadline=deadline)
                    err = total_error(U_opt, V_opt, W_opt)
                    if err < 1e-6:
                        return U_opt, V_opt, W_opt, r
                
                # Keep the best found so far (lowest error)
                if best_err == 0.0 or err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = r
        
        return best_U, best_V, best_W, best_rank

    U, V, W, rank = randomized_algebraic_search()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
