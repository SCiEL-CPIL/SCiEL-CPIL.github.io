"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import scipy
import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T
    # T is defined by the schoolbook decomposition terms
    # We construct the tensor T as a 3D array
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def error(U, V, W):
        # Compute the reconstructed tensor
        # U: (dim_u, r), V: (dim_v, r), W: (dim_w, r)
        # Reconstructed: sum_c U[:,c] outer V[:,c] outer W[:,c]
        # We can do this efficiently using einsum
        # Rec[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        Rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.sum((T - Rec) ** 2)

    def snap_to_half_int(x):
        # Snap values to nearest half-integer
        return np.round(x * 2) / 2.0

    best_rank = 12
    best_U = None
    best_V = None
    best_W = None
    best_err = np.inf

    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Try ranks from 9 upwards
    # It is known that rank(2,2,3) is 9. 
    # If we can't find 9, we try 10, 11.
    for rank in [9, 10, 11, 12]:
        if time.time() > deadline:
            break
            
        found = False
        # Multiple restarts for this rank
        num_restarts = 20 if rank == 9 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            # Use small integers to encourage integer solutions
            U = np.random.choice([-1, 0, 1], size=(dim_u, rank))
            V = np.random.choice([-1, 0, 1], size=(dim_v, rank))
            W = np.random.choice([-1, 0, 1], size=(dim_w, rank))
            
            # Optimization
            # Flatten U, V, W into a single vector
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_opt = x[:dim_u*rank].reshape(dim_u, rank)
                V_opt = x[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
                W_opt = x[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
                return error(U_opt, V_opt, W_opt)
            
            try:
                # Use BFGS or L-BFGS-B
                # Since we want integer/half-integer, we might need to be careful.
                # Let's just optimize and snap at the end.
                res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'gtol': 1e-8})
                x_opt = res.x
                U_opt = x_opt[:dim_u*rank].reshape(dim_u, rank)
                V_opt = x_opt[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
                W_opt = x_opt[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
                
                err = res.fun
                
                # Snap to half-integers and check error again
                U_snap = snap_to_half_int(U_opt)
                V_snap = snap_to_half_int(V_opt)
                W_snap = snap_to_half_int(W_opt)
                err_snap = error(U_snap, V_snap, W_snap)
                
                if err_snap < 1e-9:
                    # Found an exact decomposition
                    if rank < best_rank:
                        best_rank = rank
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        best_err = err_snap
                        found = True
                        break # Break restart loop
                elif err < best_err:
                    # Keep track of best approximate solution in case we don't find exact
                    if rank < best_rank or (rank == best_rank and err < best_err):
                        # We only update best_rank if we find a valid exact solution
                        # But we can store the best approximate for fallback
                        pass
                        
            except Exception:
                continue

        if found:
            break

    # If no exact solution found, return the schoolbook solution (rank 12)
    if best_U is None:
        rank = 12
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
