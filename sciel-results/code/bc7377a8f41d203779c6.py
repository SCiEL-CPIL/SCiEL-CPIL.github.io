"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Known lower bounds: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Schoolbook gives 63. We'll search for better ranks.
    
    best_result = None
    best_rank = None
    
    # Try ranks from lower bound upward
    for rank in range(21, 45):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations for this rank
        max_restarts = 50 if rank <= 25 else 25
        if time.time() > deadline - 10:
            max_restarts = 10
            
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            try:
                result = _try_rank(rank, n, m, p, restart, deadline)
                if result is not None:
                    found = True
                    if best_result is None or rank < best_rank:
                        best_rank = rank
                        best_result = result
                    break
            except Exception:
                continue
        
        if found and best_rank == rank:
            continue
        elif not found:
            pass
    
    # If nothing found, return schoolbook
    if best_result is None:
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
    
    return best_result

def _try_rank(rank, n, m, p, restart_idx, deadline):
    """Try to find a rank-rank decomposition using random init + adaptive gradient descent."""
    np.random.seed(42 + restart_idx * 1000)
    
    # Initialize U, V, W with different strategies
    strategy = restart_idx % 3
    
    if strategy == 0:
        # Random Gaussian
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        scale = 1.0 / np.sqrt(rank)
        U *= scale
        V *= scale
        W *= scale
    elif strategy == 1:
        # Sparse random initialization
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c in range(rank):
            u_idx = np.random.choice(n * m, size=min(3, n * m), replace=False)
            v_idx = np.random.choice(m * p, size=min(3, m * p), replace=False)
            w_idx = np.random.choice(p * n, size=min(3, p * n), replace=False)
            U[u_idx, c] = np.random.choice([-1, 1], size=len(u_idx))
            V[v_idx, c] = np.random.choice([-1, 1], size=len(v_idx))
            W[w_idx, c] = np.random.choice([-1, 1], size=len(w_idx))
    else:
        # Structured initialization based on schoolbook pattern
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        # Initialize with partial schoolbook structure
        for c in range(rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            if i < n and j < m and k < p:
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
        # Add random perturbations
        U += 0.1 * np.random.randn(n * m, rank)
        V += 0.1 * np.random.randn(m * p, rank)
        W += 0.1 * np.random.randn(p * n, rank)
    
    # Use adaptive gradient descent to minimize ||tensor - sum_c U_c ⊗ V_c ⊗ W_c||
    lr = 0.05
    lr_decay = 0.999
    momentum = 0.9
    max_iter = 500
    
    # Momentum buffers
    vel_U = np.zeros_like(U)
    vel_V = np.zeros_like(V)
    vel_W = np.zeros_like(W)
    
    prev_loss = np.inf
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
            
        # Compute current tensor estimate
        T_est = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            u3d = u_c[:, :, np.newaxis]
            v3d = v_c[np.newaxis, :, :]
            w3d = np.broadcast_to(w_c.T[np.newaxis, :, :], (n, 1, p))
            
            T_est += u3d * v3d * w3d
        
        # Target tensor: all ones
        T_target = np.ones((n, m, p))
        
        # Loss
        diff = T_est - T_target
        loss = np.sum(diff ** 2)
        
        # Check convergence
        if loss < 1e-10:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Compute gradients
        dU = np.zeros((n * m, rank))
        dV = np.zeros((m * p, rank))
        dW = np.zeros((p * n, rank))
        
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            D = diff * w_c.T[np.newaxis, np.newaxis, :]
            C_u = np.einsum('ijk,jk->ij', D, v_c)
            dU[c] = 2 * C_u.ravel()
            
            E = diff * u_c[:, :, np.newaxis]
            C_v = np.einsum('ijk,ki->jk', E, w_c)
            dV[c] = 2 * C_v.ravel()
            
            F = diff * v_c[np.newaxis, :, :]
            C_w = np.einsum('ijk,ij->ki', F, u_c)
            dW[c] = 2 * C_w.ravel()
        
        # Adaptive learning rate based on loss change
        if prev_loss > 0 and loss < prev_loss:
            lr *= 1.01  # Slightly increase if improving
        elif prev_loss > 0 and loss > prev_loss:
            lr *= 0.95  # Decrease if worsening
        
        lr = max(0.001, min(0.1, lr))
        prev_loss = loss
        
        # Update with momentum
        vel_U = momentum * vel_U + (1 - momentum) * dU
        vel_V = momentum * vel_V + (1 - momentum) * dV
        vel_W = momentum * vel_W + (1 - momentum) * dW
        
        U -= lr * vel_U
        V -= lr * vel_V
        W -= lr * vel_W
    
    # Final check
    T_est = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        u3d = u_c[:, :, np.newaxis]
        v3d = v_c[np.newaxis, :, :]
        w3d = np.broadcast_to(w_c.T[np.newaxis, :, :], (n, 1, p))
        T_est += u3d * v3d * w3d
    
    loss = np.sum((T_est - 1.0) ** 2)
    if loss < 1e-8:
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
