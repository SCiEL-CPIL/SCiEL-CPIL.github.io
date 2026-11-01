import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_solution = None
    best_rank = rank_school
    
    # Try ranks from 15 up to 17
    for rank in range(15, 18):
        if time.time() > deadline:
            break
            
        # Multiple restarts
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # ALS iterations
            max_iters = 50
            converged = False
            for it in range(max_iters):
                # Update U
                for i in range(n):
                    for j in range(m):
                        idx_u = i * m + j
                        # M[k, r] = V[j, k, r] * W[k, i, r]
                        # V is (m*p, rank). Index for (j,k) is j*p + k
                        # W is (p*n, rank). Index for (k,i) is k*n + i
                        M = np.zeros((p, rank))
                        for k in range(p):
                            idx_v = j * p + k
                            idx_w = k * n + i
                            M[k, :] = V[idx_v, :] * W[idx_w, :]
                        
                        # Solve M @ u = 1
                        # u is (rank, 1)
                        try:
                            # Minimum norm solution if rank > p
                            # u = M.T @ inv(M @ M.T) @ ones
                            MMt = M @ M.T
                            # Add regularization for stability
                            MMt += 1e-8 * np.eye(p)
                            u = M.T @ np.linalg.solve(MMt, np.ones(p))
                            U[idx_u, :] = u
                        except np.linalg.LinAlgError:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        idx_v = j * p + k
                        N = np.zeros((n, rank))
                        for i in range(n):
                            idx_u = i * m + j
                            idx_w = k * n + i
                            N[i, :] = U[idx_u, :] * W[idx_w, :]
                        
                        try:
                            NNt = N @ N.T
                            NNt += 1e-8 * np.eye(n)
                            v = N.T @ np.linalg.solve(NNt, np.ones(n))
                            V[idx_v, :] = v
                        except np.linalg.LinAlgError:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        idx_w = k * n + i
                        O = np.zeros((m, rank))
                        for j in range(m):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            O[j, :] = U[idx_u, :] * V[idx_v, :]
                        
                        try:
                            OOt = O @ O.T
                            OOt += 1e-8 * np.eye(m)
                            w = O.T @ np.linalg.solve(OOt, np.ones(m))
                            W[idx_w, :] = w
                        except np.linalg.LinAlgError:
                            pass

            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        error += (val - 1.0) ** 2
            
            if error < 1e-6:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.dot(U_snap[idx_u, :], V_snap[idx_v, :] * W_snap[idx_w, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
                
                if valid:
                    best_solution = (U_snap, V_snap, W_snap)
                    best_rank = rank
                    break # Found valid solution for this rank, go to next rank (lower is better, so break inner loop)
        
        if best_solution is not None and best_rank == rank:
            break # Found solution for current rank, which is the lowest we tried so far. Since we iterate 15,16,17, if we find 15, we stop.
            
    if best_solution is not None:
        U, V, W = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
