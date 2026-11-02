"""Construct rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    # Target rank to try. Schoolbook is 72.
    # We try to find a decomposition with rank < 72.
    # Based on 3x3x3 rank 19, 3x3x8 might be around 50-60.
    
    # Define the indices for the tensor equations
    # Equation for each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    idx_u = []
    idx_v = []
    idx_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i * m + j)
                idx_v.append(j * p + k)
                idx_w.append(k * n + i)
                
    idx_u = np.array(idx_u)
    idx_v = np.array(idx_v)
    idx_w = np.array(idx_w)
    
    num_eqs = len(idx_u)
    
    def residual(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Compute sum_r U[idx_u, r] V[idx_v, r] W[idx_w, r]
        # This is element-wise multiplication of rows and sum over R
        # U[idx_u] is (num_eqs, R)
        # V[idx_v] is (num_eqs, R)
        # W[idx_w] is (num_eqs, R)
        # Product is (num_eqs, R)
        # Sum over axis 1 gives (num_eqs,)
        prod = U[idx_u] * V[idx_v] * W[idx_w]
        return np.sum(prod, axis=1) - 1.0

    # Try to find a decomposition with rank R
    # We use a simple gradient descent / random restart
    # To respect the "BP" directive, we use an iterative refinement
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to improve
    # We try ranks from 60 down to 40
    for R in range(60, 39, -2):
        if time.time() > 100: break # Leave time for finalization
        
        for attempt in range(5):
            # Initialize with random small values
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Iterative refinement (Gradient Descent)
            lr = 0.01
            for step in range(1000):
                res = residual(U, V, W)
                if np.max(np.abs(res)) < 1e-5:
                    # Found a valid decomposition
                    if R < best_rank:
                        best_rank = R
                        best_U = U
                        best_V = V
                        best_W = W
                    break
                
                # Compute gradients
                # d/dU[idx_u, r] = V[idx_v, r] * W[idx_w, r] * res
                # We need to update U, V, W
                # This is expensive to do element-wise in Python loops
                # We use vectorized ops
                
                # grad_U is sparse, only at idx_u
                # grad_U[idx_u, :] = res[:, None] * V[idx_v] * W[idx_w]
                # But U has other rows. We only care about the rows used in equations?
                # No, all rows of U are used in some equation.
                # Rows of U not in idx_u are never used?
                # idx_u contains all i*m+j for i,j. Yes, all rows are used.
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # res is (num_eqs,)
                # V[idx_v] is (num_eqs, R)
                # W[idx_w] is (num_eqs, R)
                
                # grad_U[idx_u, r] = sum_eq res[eq] * V[idx_v[eq], r] * W[idx_w[eq], r]
                # This is not a simple outer product because idx_u has duplicates?
                # idx_u = [0,1,2, 3,4,5, ...]
                # For each i,j, there are p equations.
                # So row i*m+j of U appears in p equations.
                
                # We can compute this by scattering the contributions.
                # contrib_U[eq, r] = res[eq] * V[idx_v[eq], r] * W[idx_w[eq], r]
                # Then grad_U[idx_u[eq], r] += contrib_U[eq, r]
                
                contrib_U = res[:, None] * V[idx_v] * W[idx_w]
                contrib_V = res[:, None] * U[idx_u] * W[idx_w]
                contrib_W = res[:, None] * U[idx_u] * V[idx_v]
                
                # Scatter add
                # np.add.at
                np.add.at(grad_U, idx_u, contrib_U)
                np.add.at(grad_V, idx_v, contrib_V)
                np.add.at(grad_W, idx_w, contrib_W)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Optional: normalize to prevent blowup
                # U /= np.linalg.norm(U)
                
            else:
                # Check if it converged sufficiently
                res = residual(U, V, W)
                if np.max(np.abs(res)) < 1e-4:
                    if R < best_rank:
                        best_rank = R
                        best_U = U
                        best_V = V
                        best_W = W

    # Convert to list of lists as required
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
