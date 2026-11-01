import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Seed solution (Rank 24) - schoolbook
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed
    
    start_time = time.time()
    timeout = 130.0  # Safety margin
    
    # Constraints: (u_idx, v_idx, w_idx)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
                
    num_constraints = len(constraints)
    
    # Pre-compute indices for vectorization
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])

    def error_function(U, V, W):
        # Vectorized error calculation
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)

    def try_rank(r, time_limit=10.0, seed=42):
        np.random.seed(seed)
        # Try a few random restarts
        for attempt in range(15):
            if time.time() - start_time > timeout - 5.0:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Gradient Descent with Momentum
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for step in range(300):
                if time.time() - start_time > timeout - 5.0:
                    return None
                
                # Compute gradients
                vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
                diffs = 2 * (vals - 1.0)
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Scatter gradients
                np.add.at(dU, u_idxs, diffs[:, None] * V[v_idxs, :] * W[w_idxs, :])
                np.add.at(dV, v_idxs, diffs[:, None] * U[u_idxs, :] * W[w_idxs, :])
                np.add.at(dW, w_idxs, diffs[:, None] * U[u_idxs, :] * V[v_idxs, :])
                
                # Update with momentum
                vU = momentum * vU - lr * dU
                vV = momentum * vV - lr * dV
                vW = momentum * vW - lr * dW
                
                U += vU
                V += vV
                W += vW
                
                if step % 10 == 0:
                    err = error_function(U, V, W)
                    if err < 1e-8:
                        # Round to nearest 0.5
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        # Verify exactness
                        if error_function(U_rounded, V_rounded, W_rounded) < 1e-10:
                            return U_rounded, V_rounded, W_rounded
                        break
        return None

    # Try lower ranks
    for r in [23, 22, 21]:
        sol = try_rank(r, time_limit=10.0)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
