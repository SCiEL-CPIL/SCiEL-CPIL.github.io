import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Target tensor T[a, b, c]
    # a: i*m + j
    # b: j'*p + k
    # c: k'*n + i'
    # T[a, b, c] = 1 if i=i', j=j', k=k' else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    best_rank = n * m * p
    best_result = None
    
    # Try ranks from 16 upwards
    # We know 16 is the target
    for rank in range(16, n*m*p + 1):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple restarts for this rank
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Use small integers or half-integers
            # Scale: entries around 1
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            # ALS Iterations
            max_iters = 100
            prev_err = float('inf')
            current_err = prev_err
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Solve for U
                # A shape: (m*p * p*n, rank)
                # A[(b,c), r] = V[b,r] * W[c,r]
                A_U = np.zeros((m*p * p*n, rank))
                for b in range(m*p):
                    for c in range(p*n):
                        idx = b * p*n + c
                        A_U[idx, :] = V[b, :] * W[c, :]
                
                # Solve for each a
                for a in range(n*m):
                    y = T[a, :, :].flatten()
                    # A_U is same for all a, but y depends on a
                    # Solve A_U x = y
                    x, residuals, rank_u, s = np.linalg.lstsq(A_U, y, rcond=None)
                    U[a, :] = x
                
                # Solve for V
                A_V = np.zeros((n*m * p*n, rank))
                for a in range(n*m):
                    for c in range(p*n):
                        idx = a * p*n + c
                        A_V[idx, :] = U[a, :] * W[c, :]
                        
                for b in range(m*p):
                    y = T[:, b, :].flatten()
                    x, residuals, rank_v, s = np.linalg.lstsq(A_V, y, rcond=None)
                    V[b, :] = x

                # Solve for W
                A_W = np.zeros((n*m * m*p, rank))
                for a in range(n*m):
                    for b in range(m*p):
                        idx = a * m*p + b
                        A_W[idx, :] = U[a, :] * V[b, :]
                        
                for c in range(p*n):
                    y = T[:, :, c].flatten()
                    x, residuals, rank_w, s = np.linalg.lstsq(A_W, y, rcond=None)
                    W[c, :] = x

                # Calculate Error
                # Reconstruct tensor
                # T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Efficient calculation:
                # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                current_err = np.linalg.norm(T_rec - T)
                
                if abs(current_err - prev_err) < 1e-6:
                    break
                prev_err = current_err
                
            # Check if solution is close to exact
            if current_err < 1e-5:
                # Round to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness
                T_rec_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                if np.allclose(T_rec_int, T, atol=1e-9):
                    # Found a valid solution
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    
        # If not found for this rank, continue to next rank
        # But we expect 16 to work
        
    # Fallback to schoolbook if no better solution found
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
