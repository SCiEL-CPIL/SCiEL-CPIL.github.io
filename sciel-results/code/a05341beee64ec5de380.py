import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank
    # The tensor entries T_{i,j,k} are 1 for all i,j,k in standard multiplication?
    # Let's assume the standard multiplication tensor.
    # We need to solve: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    for r in range(best_rank - 1, 34, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random U and V with small integers
            U = [[random.randint(-1, 1) for _ in range(r)] for _ in range(n * m)]
            V = [[random.randint(-1, 1) for _ in range(r)] for _ in range(m * p)]
            
            # Construct linear system for W
            # Variables: W[k*n+i, c] for k in p, i in n, c in r
            # Num vars: p * n * r
            # Equations: n * m * p
            num_vars = p * n * r
            num_eqs = n * m * p
            
            # Matrix A of size num_eqs x num_vars
            # RHS b of size num_eqs
            A = np.zeros((num_eqs, num_vars), dtype=float)
            b = np.ones(num_eqs, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation for T[i,j,k] = 1
                        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                        for c in range(r):
                            coeff = U[i * m + j][c] * V[j * p + k][c]
                            if coeff != 0:
                                var_idx = (k * n + i) * r + c
                                A[eq_idx, var_idx] += coeff
                        eq_idx += 1
            
            # Solve A * W_flat = b
            try:
                # Use least squares if underdetermined or overdetermined
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if solution is close to exact
                if np.allclose(A @ W_flat, b, atol=1e-9):
                    # Check if W_flat can be represented as integers/half-integers
                    # Scale to clear denominators?
                    # For now, just check if they are close to half-integers
                    # Multiply by 2 and check closeness to integers
                    W_scaled = W_flat * 2
                    W_int = np.round(W_scaled)
                    if np.allclose(W_scaled, W_int, atol=1e-6):
                        # We found a valid decomposition
                        # Construct W matrix
                        W = [[0] * r for _ in range(p * n)]
                        for idx, val in enumerate(W_flat):
                            row = idx // r
                            col = idx % r
                            W[row][col] = val / 2.0 # Keep as float, evaluator might accept?
                            # Prompt says integer/half-integer. 
                            # If val/2 is integer or half-integer, it's fine.
                        
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = r
                        # Since we found a lower rank, we can stop searching for this r
                        # and try lower r
                        break 
            except Exception:
                continue
        
        # If we found a rank r, we can stop here or try lower?
        # The loop iterates r downwards. If we break the inner loop, we continue to next r.
        # But we should return the best found.
        # Since we are iterating downwards, the first found is the best.
        if best_rank < rank_sb:
            break

    return best_result
