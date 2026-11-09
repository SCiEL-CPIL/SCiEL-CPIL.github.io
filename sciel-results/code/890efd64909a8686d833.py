import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p

    # Target tensor indices
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Construct the full target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1.0

    # Verify function
    def verify(U, V, W):
        T_approx = np.einsum('ut,vt,wt->uvw', U, V, W)
        diff = np.abs(T_approx - T_target)
        return np.all(diff < 1e-9)

    # Schoolbook solution (Rank 40)
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_solution = {'U': U_school, 'V': V_school, 'W': W_school}
    best_rank = rank_schoolbook

    # New Strategy: Decoupled Linear Solve
    # For fixed U, V, the system for W is decoupled by k.
    # For each k, we solve B_k w_k = t_k
    # B_k is (n*m, r) with entries B_k[i*m+j, t] = U[i*m+j, t] * V[j*p+k, t]
    # t_k is (n*m, 1) with entries t_k[i*m+j] = T_target[i*m+j, j*p+k, k*n+i]
    
    def try_random_decomposition(r, max_restarts=100):
        # We search for U, V with small integer entries
        # Entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
        
        # Pre-compute t_k vectors
        t_vectors = []
        for k in range(p):
            t_k = np.zeros(n*m)
            for i in range(n):
                for j in range(m):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    t_k[idx_u] = T_target[idx_u, idx_v, idx_w]
            t_vectors.append(t_k)
            
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                return None
            
            # Generate random sparse U, V
            # Use {-1, 0, 1}
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            
            # Randomize U
            for idx in range(n*m):
                if random.random() < 0.5: # 50% non-zero
                    U[idx, :] = np.random.choice([-1, 1], size=r)
                    
            # Randomize V
            for idx in range(m*p):
                if random.random() < 0.5:
                    V[idx, :] = np.random.choice([-1, 1], size=r)

            # Solve for W
            W = np.zeros((p*n, r))
            valid = True
            
            for k in range(p):
                # Construct B_k
                # B_k[i*m+j, t] = U[i*m+j, t] * V[j*p+k, t]
                B_k = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        B_k[idx_u, :] = U[idx_u, :] * V[idx_v, :]
                
                # Solve B_k w_k = t_k
                # If B_k is singular or no solution, fail
                try:
                    # Use least squares if not square/invertible, but we want exact integer solution.
                    # If r < n*m, it's underdetermined. 
                    # Wait, if r < n*m, there are infinite solutions.
                    # We need *an* integer solution.
                    # If r >= n*m, it might be overdetermined.
                    # Usually r < n*m for low rank?
                    # n*m = 8. r is ~30-39. So r > n*m.
                    # So it's overdetermined. We need exact solution.
                    
                    # Check if t_k is in column space of B_k
                    # Solve B_k w = t_k
                    w_k, residuals, rank, s = np.linalg.lstsq(B_k, t_vectors[k], rcond=None)
                    
                    # Check residual
                    if np.linalg.norm(B_k @ w_k - t_vectors[k]) > 1e-9:
                        valid = False
                        break
                    
                    # Check if w_k is close to integer or half-integer
                    # Try snapping
                    w_snap = np.round(w_k * 2) / 2
                    if np.linalg.norm(B_k @ w_snap - t_vectors[k]) < 1e-9:
                        W[k*n:(k+1)*n, :] = w_snap
                    else:
                        valid = False
                        break
                        
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if valid:
                # Final verification
                if verify(U, V, W):
                    return U, V, W
                    
        return None

    # Search for lower ranks
    # We try ranks from 39 down to 30
    # We increase restarts for lower ranks as they are harder
    
    for r in range(rank_schoolbook - 1, 25, -1):
        if time.time() > deadline - 10:
            break
            
        # Increase restarts for lower ranks
        max_restarts = 50 if r > 35 else 200
        
        result = try_random_decomposition(r, max_restarts)
        
        if result is not None:
            U, V, W = result
            best_solution = {'U': U, 'V': V, 'W': W}
            best_rank = r
            # Continue searching for even lower rank
            continue
        else:
            # If we fail to find a solution for this rank, we might still try lower?
            # Usually if rank r is impossible, rank r-1 is also impossible?
            # No, rank is the minimum. If we can't find r, we might not find r-1 either.
            # But we might have just failed to find the specific structure for r.
            # Let's continue to r-1 just in case, but with less hope.
            pass

    return {
        "U": best_solution['U'].tolist(),
        "V": best_solution['V'].tolist(),
        "W": best_solution['W'].tolist()
    }
