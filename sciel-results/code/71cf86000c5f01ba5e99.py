"""Matrix multiplication 2x3x3 - rank-minimal tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    
    # We want to decompose the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # T = sum_{t=0}^{r-1} u_t ⊗ v_t ⊗ w_t
    # i.e., for each (a,b,c): T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(6,9,6) = 9
    # But for 2x3x3, we know rank is between 9 and 18.
    # Let's try to find a rank 10 or 11 decomposition using optimization.
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from 10 to 15 (skip 9 as it's very unlikely to work with small coefficients)
    # Use a randomized approach: initialize with small values and use gradient descent
    # on the loss function sum (T - sum_t U[:,t] V[:,t] W[:,t])^2
    
    def try_optimize_rank(r, max_iter=5000, lr=0.01, seed=42):
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        for it in range(max_iter):
            # Compute current tensor estimate
            # T_est[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
            # Loss = sum (T - T_est)^2
            
            # Compute gradients
            # dLoss/dU[a,t] = 2 * sum_{b,c} (T[a,b,c] - T_est[a,b,c]) * V[b,t] * W[c,t]
            # Similarly for V and W
            
            # For efficiency, compute intermediate terms
            # Let's just do a simple forward-backward for small dimensions
            
            # Compute T_est
            T_est = np.zeros((n*m, m*p, p*n))
            for t in range(r):
                T_est += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
            
            # Check convergence
            loss = np.sum((T - T_est)**2)
            if loss < 1e-10:
                return U, V, W, loss
            
            # Compute gradients
            dU = np.zeros((n*m, r))
            dV = np.zeros((m*p, r))
            dW = np.zeros((p*n, r))
            
            # dLoss/dU[a,t] = 2 * sum_{b,c} (T[a,b,c] - T_est[a,b,c]) * V[b,t] * W[c,t]
            # = 2 * sum_t' ... wait, let's think about this more carefully
            # Actually, for each t, the contribution to T_est is U[:,t] V[:,t] W[:,t]
            # dLoss/dU[a,t] = 2 * sum_{b,c} (T[a,b,c] - T_est[a,b,c]) * V[b,t] * W[c,t]
            
            diff = T - T_est
            
            for t in range(r):
                # dU[:,t] = 2 * V[:,t] * W[:,t] reshaped and contracted with diff
                # For fixed t and a: dU[a,t] = 2 * sum_{b,c} diff[a,b,c] * V[b,t] * W[c,t]
                # = 2 * (diff * (V[:,t,None,None] * W[:,t][None,:,None] * np.ones((1,m*p,p*n))[None,...]))
                
                # More efficiently:
                # dU[:,t] = 2 * np.einsum('abc,b,c->a', diff, V[:,t], W[:,t])
                dU[:,t] = 2.0 * np.einsum('abc,b,c->a', diff, V[:,t], W[:,t])
                dV[:,t] = 2.0 * np.einsum('abc,a,c->b', diff, U[:,t], W[:,t])
                dW[:,t] = 2.0 * np.einsum('abc,a,b->c', diff, U[:,t], V[:,t])
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Final loss
        T_est = np.zeros((n*m, m*p, p*n))
        for t in range(r):
            T_est += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
        loss = np.sum((T - T_est)**2)
        return U, V, W, loss
    
    # Try ranks from 10 to 14
    best_result = None
    for r in range(10, 15):
        for seed in range(5):
            U, V, W, loss = try_optimize_rank(r, max_iter=2000, lr=0.005, seed=seed)
            if loss < 1e-8:
                if best_result is None or r < best_result[0]:
                    best_result = (r, U, V, W)
                break
    
    if best_result is not None:
        r, U, V, W = best_result
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
