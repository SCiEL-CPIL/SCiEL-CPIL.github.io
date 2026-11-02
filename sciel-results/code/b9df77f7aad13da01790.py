"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Build target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook baseline
    rank_sb = n * m * p  # 52
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Precompute indices for fast evaluation
    # For each (i,j,k), we need U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    idx_U = []
    idx_V = []
    idx_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U.append(i*m+j)
                idx_V.append(j*p+k)
                idx_W.append(k*n+i)
    idx_U = np.array(idx_U)
    idx_V = np.array(idx_V)
    idx_W = np.array(idx_W)
    
    def compute_residual(U, V, W):
        """Compute residual R[i,j,k] = sum_r U[idx_U,r]*V[idx_V,r]*W[idx_W,r] - 1"""
        # U[idx_U, :] has shape (num_entries, rank)
        U_sel = U[idx_U, :]
        V_sel = V[idx_V, :]
        W_sel = W[idx_W, :]
        # Element-wise product then sum over rank dimension
        prod = U_sel * V_sel * W_sel
        return prod.sum(axis=1) - 1.0
    
    def max_error(U, V, W):
        res = compute_residual(U, V, W)
        return np.max(np.abs(res))
    
    np.random.seed(12345)
    
    # Try to find lower rank decompositions
    # Use a more efficient optimization approach
    for target_rank in range(8, min(rank_sb + 1, 40)):
        if time.time() > deadline - 15:
            break
        
        found = False
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize
            U = np.random.randn(n*m, target_rank) * 0.3
            V = np.random.randn(m*p, target_rank) * 0.3
            W = np.random.randn(p*n, target_rank) * 0.3
            
            # Adaptive learning rate gradient descent
            lr = 0.005
            for iteration in range(300):
                if time.time() > deadline - 8:
                    break
                
                # Compute residual and gradients using vectorized ops
                U_sel = U[idx_U, :]
                V_sel = V[idx_V, :]
                W_sel = W[idx_W, :]
                
                prod = U_sel * V_sel * W_sel
                residual = prod.sum(axis=1) - 1.0  # shape (n*m*p,)
                
                # Gradients
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                # d/dU: residual * V * W
                np.add.at(U_grad, idx_U, residual[:, None] * V_sel * W_sel)
                np.add.at(V_grad, idx_V, residual[:, None] * U_sel * W_sel)
                np.add.at(W_grad, idx_W, residual[:, None] * U_sel * V_sel)
                
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                # Check convergence every 10 iterations
                if iteration % 10 == 0:
                    err = max_error(U, V, W)
                    if err < 1e-9:
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                        found = True
                        break
            
            if found:
                break
    
    # Convert to lists
    U_list = [[float(best_U[i, r]) for r in range(best_rank)] for i in range(n*m)]
    V_list = [[float(best_V[i, r]) for r in range(best_rank)] for i in range(m*p)]
    W_list = [[float(best_W[i, r]) for r in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
