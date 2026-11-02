import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    
    start_time = time.time()
    deadline = 140
    
    # Build target tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    # Schoolbook fallback
    rank_school = n * m * p  # 56
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for (u_idx, v_idx, w_idx) in tensor_entries:
        U_school[u_idx, col] = 1
        V_school[v_idx, col] = 1
        W_school[w_idx, col] = 1
        col += 1
    
    # Try to find a valid decomposition with rank < 56
    # Using a greedy approach with exact verification
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try ranks from 50 down to 45
    for target_rank in range(50, 44, -1):
        if time.time() - start_time > deadline:
            break
        
        # Try multiple random attempts
        for attempt in range(5):
            if time.time() - start_time > deadline:
                break
            
            # Generate random U and V with small integer entries
            U = np.random.randint(-2, 3, size=(dim_u, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(dim_v, target_rank)).astype(float)
            
            # Solve for W: the system is
            # For each (u,v,w): sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w]
            # This is linear in W.
            
            # Build the linear system A * W_vec = b
            # Where W_vec is the vectorization of W (length dim_w * target_rank)
            num_unk = dim_w * target_rank
            num_nonzero = len(tensor_entries)
            
            # Only the nonzero entries of T give equations with b=1
            # All other entries give equations with b=0
            # We'll only enforce the nonzero equations for efficiency
            
            # For each nonzero entry (u,v,w), the equation is:
            # sum_r U[u,r] * V[v,r] * W[w,r] = 1
            # This involves only the W[w,:] variables
            
            # So we have a block-diagonal-like system: for each w, the variables W[w,:] 
            # appear in equations for all (u,v) with that w.
            
            # Let's set up the system for each w separately.
            # For fixed w, the equations are:
            # For all (u,v) such that T[u,v,w] = 1: sum_r U[u,r]*V[v,r]*W[w,r] = 1
            # For all (u,v) such that T[u,v,w] = 0: sum_r U[u,r]*V[v,r]*W[w,r] = 0
            
            # This is a system of dim_u*dim_v equations in target_rank unknowns.
            # dim_u*dim_v = 8*28 = 224 equations, target_rank unknowns.
            # Overdetermined. We need the system to be consistent.
            
            # Let's try to solve it using least squares and check if the residual is zero.
            
            W_sol = np.zeros((dim_w, target_rank))
            valid = True
            
            for w in range(dim_w):
                # Build the system for this w
                # Equations: for each (u,v), sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w]
                A_w = np.zeros((dim_u * dim_v, target_rank))
                b_w = np.zeros(dim_u * dim_v)
                
                idx = 0
                for u in range(dim_u):
                    for v in range(dim_v):
                        # T[u,v,w] = 1 if there exists i,j,k such that 
                        # u = i*m+j, v = j*p+k, w = k*n+i
                        # Check if T[u,v,w] is nonzero
                        t_val = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if i*m+j == u and j*p+k == v and k*n+i == w:
                                        t_val = 1.0
                        A_w[idx, :] = U[u, :] * V[v, :]
                        b_w[idx] = t_val
                        idx += 1
                
                # Solve A_w * W[w,:] = b_w
                # Use least squares
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                    W_sol[w, :] = sol
                    # Check if the solution is exact
                    if residuals.size > 0 and residuals[0] > 1e-10:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify the full decomposition
            # Check that sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w] for all u,v,w
            error = 0.0
            for u in range(dim_u):
                for v in range(dim_v):
                    for w in range(dim_w):
                        val = np.dot(U[u, :] * V[v, :], W_sol[w, :])
                        # Compute T[u,v,w]
                        t_val = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if i*m+j == u and j*p+k == v and k*n+i == w:
                                        t_val = 1.0
                        error += (val - t_val) ** 2
            
            if error < 1e-10:
                # Found a valid decomposition!
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W_sol
                break
        
        if best_rank < rank_school:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
