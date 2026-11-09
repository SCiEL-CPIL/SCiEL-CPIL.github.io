import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # Target tensor T
    # T is a 9x9x9 tensor, but sparse.
    # Indices: u = i*m + j, v = j*p + k, w = k*n + i
    # Non-zeros at (u, v, w) = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Rank-23 construction for 3x3 matrix multiplication
    # Based on the known result that rank(M_3) <= 23
    # This is a non-trivial construction. Let me implement it via
    # a more systematic approach using the structure of the tensor.

    # The key insight: the 3x3 matrix multiplication tensor has rank at most 23.
    # This was shown by various authors. The construction typically involves
    # combining smaller Strassen-like constructions with careful handling of cross terms.

    # Let me try a different approach: use the known fact that we can write
    # the 3x3 product as a combination of smaller products.
    # 
    # One standard construction:
    # - Use 7 Strassen terms for a 2x2 block
    # - Use 1 term for the (3,3) corner
    # - Use 15 terms for the cross-coupling
    # Total: 7 + 1 + 15 = 23
    #
    # But the cross-coupling terms need to be carefully designed.
    #
    # Alternative: use a known explicit rank-23 decomposition from the literature.
    # The construction by Landsberg (1988) or later refinements give explicit
    # factor matrices.
    #
    # Let me try to find a valid rank-23 decomposition numerically with a better
    # initialization strategy.

    def try_find_decomposition(rank, max_iters=1000, restarts=10):
        np.random.seed(42)  # For reproducibility in debugging
        
        for restart in range(restarts):
            if time.time() > deadline:
                return None
            
            # Better initialization: start from rank-27 and try to merge terms
            # Or use a structured initialization based on the tensor structure
            
            # Strategy: initialize U, V, W with small random perturbations around
            # a structured base
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # First, try to use a subset of the rank-27 terms as a starting point
            # and then optimize the rest
            
            # For rank 23, we can start with 23 of the 27 diagonal terms
            # and add some off-diagonal structure
            base_terms = min(rank, n*m*p)
            for t in range(base_terms):
                i = t // (m*p)
                j = (t // p) % m
                k = t % p
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, t] = 1.0
                V[v, t] = 1.0
                W[w, t] = 1.0
            
            # Add small random perturbations to all entries
            if rank > base_terms:
                U[:, base_terms:] += np.random.randn(n*m, rank - base_terms) * 0.1
                V[:, base_terms:] += np.random.randn(m*p, rank - base_terms) * 0.1
                W[:, base_terms:] += np.random.randn(p*n, rank - base_terms) * 0.1
            
            # Also perturb the base terms slightly
            U += np.random.randn(n*m, rank) * 0.05
            V += np.random.randn(m*p, rank) * 0.05
            W += np.random.randn(p*n, rank) * 0.05
            
            # Gradient descent with momentum
            lr = 0.1
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Compute current tensor estimate
                Est = np.einsum('ur,vr,wr->uvw', U, V, W)
                
                # Error
                Err = Est - T
                
                # Gradients
                G_U = np.einsum('uvw,vr,wr->ur', Err, V, W)
                G_V = np.einsum('uvw,ur,wr->vr', Err, U, W)
                G_W = np.einsum('uvw,ur,vr->wr', Err, U, V)
                
                # Momentum update
                vU = momentum * vU + lr * G_U
                vV = momentum * vV + lr * G_V
                vW = momentum * vW + lr * G_W
                
                U -= vU
                V -= vV
                W -= vW
                
                # Adaptive learning rate
                norm_err = np.linalg.norm(Err)
                if norm_err < 1e-6:
                    break
                
                # Check for divergence
                if not np.all(np.isfinite(U)) or not np.all(np.isfinite(V)) or not np.all(np.isfinite(W)):
                    break
                
                # Decay learning rate
                if it % 100 == 99:
                    lr *= 0.8
            
            # Final verification with rounding
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify exactness
            Est_r = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            if np.allclose(Est_r, T, atol=1e-10):
                return U_r, V_r, W_r
            
            # Also try without rounding to see if we're close
            Est_unrounded = np.einsum('ur,vr,wr->uvw', U, V, W)
            final_err = np.linalg.norm(Est_unrounded - T)
            if final_err < 1e-5:
                # Try more aggressive rounding
                for scale in [2, 4, 8, 16]:
                    U_r2 = np.round(U * scale) / scale
                    V_r2 = np.round(V * scale) / scale
                    W_r2 = np.round(W * scale) / scale
                    Est_r2 = np.einsum('ur,vr,wr->uvw', U_r2, V_r2, W_r2)
                    if np.allclose(Est_r2, T, atol=1e-10):
                        return U_r2, V_r2, W_r2
        
        return None

    # Try to find a rank-23 decomposition
    result = try_find_decomposition(23)
    if result:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: try rank 24, then 25, then 26, then 27
    for fallback_rank in [24, 25, 26, 27]:
        if time.time() > deadline:
            break
        result = try_find_decomposition(fallback_rank, max_iters=500, restarts=3)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Ultimate fallback: rank-27 trivial solution
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, column] = 1
                V[v, column] = 1
                W[w, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
