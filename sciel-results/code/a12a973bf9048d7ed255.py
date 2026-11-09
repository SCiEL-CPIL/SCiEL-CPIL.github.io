"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Known lower bounds: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Schoolbook gives 63.
    
    best_result = None
    best_rank = None
    
    # We know rank 55 is possible (23+23+9).
    # We know rank 59 is possible (23+36).
    # We will search from 21 upwards.
    
    # To speed up, we prioritize ranks that are likely to be found.
    # But the task asks for minimal rank.
    # We'll iterate 21 to 63.
    
    for rank in range(21, 64):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations for this rank
        # Increase restarts for lower ranks as they are harder but more valuable
        max_restarts = 10 if rank < 30 else 5
        if time.time() > deadline - 10:
            max_restarts = 2
            
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            try:
                result = _try_rank_vectorized(rank, n, m, p, restart, deadline)
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

def _try_rank_vectorized(rank, n, m, p, restart_idx, deadline):
    """Try to find a rank-rank decomposition using random init + vectorized gradient descent."""
    np.random.seed(42 + restart_idx * 1000)
    
    # Initialize U, V, W randomly
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Scale to reasonable magnitude
    scale = 1.0 / np.sqrt(rank)
    U *= scale
    V *= scale
    W *= scale
    
    # Reshape for einsum
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    lr = 0.05
    max_iter = 200
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
            
        # Compute current tensor estimate T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # einsum: 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        
        # Target tensor: all ones
        diff = T - 1.0
        loss = np.sum(diff ** 2)
        
        # Check convergence
        if loss < 1e-10:
            # Optional: Snap to nearest 0.5 or integer to ensure exactness if close
            # U = np.round(U * 2) / 2.0
            # V = np.round(V * 2) / 2.0
            # W = np.round(W * 2) / 2.0
            # Re-verify after snapping
            # Ur_snap = U.reshape(n, m, rank)
            # Vr_snap = V.reshape(m, p, rank)
            # Wr_snap = W.reshape(p, n, rank)
            # T_snap = np.einsum('ijc,jkc,kic->ijk', Ur_snap, Vr_snap, Wr_snap)
            # if np.allclose(T_snap, 1.0, atol=1e-9):
            #     return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Compute gradients using einsum
        # dU[i,j,c] = 2 * sum_{j,k} diff[i,j,k] * V[j,k,c] * W[k,i,c]
        # Note: V is (m,p,r), W is (p,n,r)
        # We need sum over j and k.
        # diff is (n,m,p)
        # dU = 2 * einsum('ijk,jkc,kic->ijc', diff, Vr, Wr)
        dU = 2 * np.einsum('ijk,jkc,kic->ijc', diff, Vr, Wr)
        
        # dV[j,k,c] = 2 * sum_{i,k} diff[i,j,k] * U[i,j,c] * W[k,i,c]
        # dV = 2 * einsum('ijk,ijc,kic->jkc', diff, Ur, Wr)
        dV = 2 * np.einsum('ijk,ijc,kic->jkc', diff, Ur, Wr)
        
        # dW[k,i,c] = 2 * sum_{i,j} diff[i,j,k] * U[i,j,c] * V[j,k,c]
        # dW = 2 * einsum('ijk,ijc,jkc->kic', diff, Ur, Vr)
        dW = 2 * np.einsum('ijk,ijc,jkc->kic', diff, Ur, Vr)
        
        # Update
        U -= lr * dU.reshape(n * m, rank)
        V -= lr * dV.reshape(m * p, rank)
        W -= lr * dW.reshape(p * n, rank)
        
        # Update reshaped views
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        
        # Adaptive learning rate
        if it % 20 == 19:
            lr *= 0.8
            
    # Final check
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    loss = np.sum((T - 1.0) ** 2)
    if loss < 1e-8:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None
# EVOLVE-BLOCK-END
