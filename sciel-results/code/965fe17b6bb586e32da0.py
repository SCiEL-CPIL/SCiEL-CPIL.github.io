import time
import numpy as np
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
                
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 24 to 30
    for r in range(24, min(31, best_rank + 1)):
        if time.time() > deadline - 10:
            break
            
        # Random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + restart * 100 + r)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Simple gradient descent / local search
            # We minimize sum (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - T[i,j,k])^2
            # This is equivalent to minimizing || U @ V.T @ W.T ... ||? No.
            # We can use a simple iterative method.
            
            # Let's use scipy.optimize if available, otherwise a simple step.
            # Since we are in a restricted environment, let's use a simple coordinate descent or random walk.
            # Actually, let's try a few steps of a simple gradient approximation.
            
            # A better approach: Use the fact that for small r, we can solve linear systems?
            # No, it's bilinear.
            
            # Let's try a simple random walk with acceptance criterion.
            current_U = U.copy()
            current_V = V.copy()
            current_W = W.copy()
            
            def calc_error(U, V, W):
                # Reconstruct tensor
                # Efficient way: 
                # Tensor = sum_r U[:, r] V[:, r] W[:, r]
                # This is slow in Python loops.
                # Use einsum?
                # T_rec[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # Indices: i is 0..8, j is 0..11, k is 0..11
                # U is 9xR, V is 12xR, W is 12xR
                # T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
                T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
                return np.sum((T_rec - T)**2)
            
            current_error = calc_error(current_U, current_V, current_W)
            
            # Local search
            step_size = 0.5
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                # Randomly pick one of U, V, W to update
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    mat = current_U
                elif mat_idx == 1:
                    mat = current_V
                else:
                    mat = current_W
                    
                row = np.random.randint(mat.shape[0])
                col = np.random.randint(mat.shape[1])
                
                # Try small perturbations
                improved = False
                for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                    new_val = mat[row, col] + delta
                    old_val = mat[row, col]
                    mat[row, col] = new_val
                    new_error = calc_error(current_U, current_V, current_W)
                    if new_error < current_error:
                        current_error = new_error
                        improved = True
                        break
                    else:
                        mat[row, col] = old_val
                
                if not improved:
                    step_size *= 0.9
                    if step_size < 1e-4:
                        break
                
                if current_error < 1e-6:
                    break
            
            if current_error < 1e-6:
                # Try to snap to half-integers
                U_snap = np.round(current_U * 2) / 2
                V_snap = np.round(current_V * 2) / 2
                W_snap = np.round(current_W * 2) / 2
                
                # Check if snapped solution is exact
                T_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T_snap, T, atol=1e-9):
                    # Convert to list of lists
                    best_rank = r
                    best_U = U_snap.tolist()
                    best_V = V_snap.tolist()
                    best_W = W_snap.tolist()
                    found = True
                    break
        
        if found:
            break

    # If no better solution found, return the naive one
    if best_U is None:
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
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
