import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Fallback seed construction
    def get_seed():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a lower rank solution using gradient descent
    # Target: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # Known lower bounds suggest rank >= 17 for 3x4x4
    for R in range(17, 20):
        if time.time() - start_time > 100:
            break
            
        for attempt in range(50):
            if time.time() - start_time > 110:
                break
                
            # Random initialization
            # Scale matters for convergence
            U = np.random.randn(n, m, R) * 0.5
            V = np.random.randn(m, p, R) * 0.5
            W = np.random.randn(p, n, R) * 0.5
            
            lr = 0.05
            converged = False
            
            for step in range(2000):
                # Compute S[i,j,k]
                S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                E = S - 1.0
                err = np.sum(E**2)
                
                if err < 1e-8:
                    converged = True
                    break
                    
                # Gradients
                # dU[i,j,r] = sum_k E[i,j,k] V[j,k,r] W[k,i,r]
                dU = np.einsum('ijk,jkr,kir->ijr', E, V, W)
                # dV[j,k,r] = sum_i E[i,j,k] U[i,j,r] W[k,i,r]
                dV = np.einsum('ijk,ijr,kir->jkr', E, U, W)
                # dW[k,i,r] = sum_j E[i,j,k] U[i,j,r] V[j,k,r]
                dW = np.einsum('ijk,ijr,jkr->kir', E, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Simple regularization to keep values bounded?
                # Not strictly necessary if lr is small
                
            if converged:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness
                S_r = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                if np.allclose(S_r, 1.0, atol=1e-9):
                    U_flat = U_r.reshape(n*m, R).tolist()
                    V_flat = V_r.reshape(m*p, R).tolist()
                    W_flat = W_r.reshape(p*n, R).tolist()
                    return {"U": U_flat, "V": V_flat, "W": W_flat}
                    
    # If no solution found, return seed
    return get_seed()
# EVOLVE-BLOCK-END
