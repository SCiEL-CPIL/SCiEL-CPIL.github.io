import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-6):
        r = U.shape[1]
        # Check condition 1: valid entries should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        # Check condition 2 (j mismatch)
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    if j == jp: continue
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[k*n+i, :])
                        if abs(val) > tol:
                            return False
        # Check condition 3 (k mismatch)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for kp in range(p):
                        if k == kp: continue
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[kp*n+i, :])
                        if abs(val) > tol:
                            return False
        # Check condition 4 (i mismatch)
        for i in range(n):
            for ip in range(n):
                if i == ip: continue
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+ip, :])
                        if abs(val) > tol:
                            return False
        return True

    def compute_loss(U, V, W):
        """Compute the sum of squared violations."""
        r = U.shape[1]
        loss = 0.0
        # Valid entries should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    loss += (val - 1) ** 2
        # j mismatch should be 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    if j == jp: continue
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[k*n+i, :])
                        loss += val ** 2
        # k mismatch should be 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for kp in range(p):
                        if k == kp: continue
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[kp*n+i, :])
                        loss += val ** 2
        # i mismatch should be 0
        for i in range(n):
            for ip in range(n):
                if i == ip: continue
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+ip, :])
                        loss += val ** 2
        return loss

    def optimize(U, V, W, max_iters=200, lr=0.01):
        """Simple gradient-free optimization using coordinate descent."""
        r = U.shape[1]
        best_loss = compute_loss(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Try small perturbations
            improved = False
            for _ in range(3):
                # Perturb U
                U_trial = U.copy()
                idx = np.random.randint(0, U_trial.shape[0])
                jdx = np.random.randint(0, r)
                delta = np.random.randn() * lr
                U_trial[idx, jdx] += delta
                loss_trial = compute_loss(U_trial, V, W)
                if loss_trial < best_loss:
                    U = U_trial
                    best_loss = loss_trial
                    improved = True
                # Perturb V
                V_trial = V.copy()
                idx = np.random.randint(0, V_trial.shape[0])
                jdx = np.random.randint(0, r)
                delta = np.random.randn() * lr
                V_trial[idx, jdx] += delta
                loss_trial = compute_loss(U, V_trial, W)
                if loss_trial < best_loss:
                    V = V_trial
                    best_loss = loss_trial
                    improved = True
                # Perturb W
                W_trial = W.copy()
                idx = np.random.randint(0, W_trial.shape[0])
                jdx = np.random.randint(0, r)
                delta = np.random.randn() * lr
                W_trial[idx, jdx] += delta
                loss_trial = compute_loss(U, V, W_trial)
                if loss_trial < best_loss:
                    W = W_trial
                    best_loss = loss_trial
                    improved = True
            if not improved:
                # Decrease learning rate
                lr *= 0.9
                if lr < 1e-8:
                    break
        
        return U, V, W, best_loss

    best_rank = rank_schoolbook
    best_solution = (U_sb, V_sb, W_sb)
    
    # Try ranks from 59 down to 30
    for r in range(rank_schoolbook - 1, max(29, 0), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Optimize
            U_opt, V_opt, W_opt, loss = optimize(U, V, W, max_iters=100, lr=0.05)
            
            if loss < 1e-8:
                # Check if it's a valid solution
                if check_decomposition(U_opt, V_opt, W_opt, tol=1e-6):
                    # Round to half-integers
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    if check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-9):
                        best_rank = r
                        best_solution = (U_rounded, V_rounded, W_rounded)
                        break
        if best_rank < rank_schoolbook:
            break
            
    # Convert best solution to list of lists
    U_list = best_solution[0].tolist()
    V_list = best_solution[1].tolist()
    W_list = best_solution[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
