import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 50
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try ranks from 48 down to 40
    for target_rank in range(48, 39, -1):
        if time.time() > deadline:
            break
        
        found = False
        num_restarts = 8
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            seed = restart * 10000 + target_rank * 100
            np.random.seed(seed)
            
            # Initialize with perturbation of a subset of schoolbook terms
            # Pick target_rank random terms from schoolbook
            indices = np.random.choice(rank_school, target_rank, replace=False)
            U = U_school[:, indices].copy()
            V = V_school[:, indices].copy()
            W = W_school[:, indices].copy()
            
            # Add small perturbation
            U += np.random.randn(nm, target_rank) * 0.1
            V += np.random.randn(mp, target_rank) * 0.1
            W += np.random.randn(pn, target_rank) * 0.1
            
            # Gradient descent with adaptive learning rate
            lr = 0.01
            for step in range(10000):
                if time.time() > deadline:
                    break
                
                T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
                err = T_recon - T_target
                error = np.sum(err**2)
                
                if error < 1e-12:
                    found = True
                    break
                
                # Gradients
                dU = 2 * np.einsum('uvw,vr,wr->ur', err, V, W)
                dV = 2 * np.einsum('uvw,ur,wr->vr', err, U, W)
                dW = 2 * np.einsum('uvw,ur,vr->wr', err, U, V)
                
                # Adaptive step size
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 1e-10:
                    step_size = lr * 0.1 / (grad_norm + 1e-10)
                else:
                    step_size = 0.01
                
                U -= step_size * dU
                V -= step_size * dV
                W -= step_size * dW
            
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break
    
    # Verify
    T_recon = np.einsum('ur,vr,wr->uvw', best_U, best_V, best_W)
    final_error = np.sum((T_recon - T_target)**2)
    
    if final_error > 1e-8:
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_school
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
