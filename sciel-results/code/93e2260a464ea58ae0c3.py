import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    schoolbook_rank = n * m * p
    
    # Try ranks from a reasonable lower bound to schoolbook
    # Lower bound is max(n*m, m*p, n*p) = 25
    # We will search from 30 upwards.
    
    best_result = None
    best_rank = schoolbook_rank
    
    for R in range(30, schoolbook_rank + 1):
        if time.time() > deadline:
            break
            
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R * 1000 + restart)
            # Initialize
            U = np.random.randn(n, m, R) * 0.5
            V = np.random.randn(m, p, R) * 0.5
            W = np.random.randn(p, n, R) * 0.5
            
            prev_error = float('inf')
            
            for it in range(200):
                if time.time() > deadline:
                    break
                    
                # Update U
                # For each i, j, solve min_u sum_k (u . (V[j,k,:] * W[k,i,:]) - 1)^2
                for i in range(n):
                    for j in range(m):
                        # B[k, r] = V[j, k, r] * W[k, i, r]
                        B = V[j, :, :] * W[:, i, :] # Shape (p, R)
                        # Solve B u = 1 (least squares)
                        # B is p x R. We want u in R^R.
                        # If R > p, rank deficient. lstsq handles it.
                        try:
                            u = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                        except:
                            continue
                        U[i, j, :] = u
                        
                # Update V
                # For each j, k, solve min_v sum_i (v . (U[i,j,:] * W[k,i,:]) - 1)^2
                for j in range(m):
                    for k in range(p):
                        # B[i, r] = U[i, j, r] * W[k, i, r]
                        B = U[:, j, :] * W[k, :, :] # Shape (n, R)
                        try:
                            v = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                        except:
                            continue
                        V[j, k, :] = v
                        
                # Update W
                # For each k, i, solve min_w sum_j (w . (U[i,j,:] * V[j,k,:]) - 1)^2
                for k in range(p):
                    for i in range(n):
                        # B[j, r] = U[i, j, r] * V[j, k, r]
                        B = U[i, :, :] * V[:, k, :] # Shape (m, R)
                        try:
                            w = np.linalg.lstsq(B, np.ones(m), rcond=None)[0]
                        except:
                            continue
                        W[k, i, :] = w
                        
                # Calculate error
                # S[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Use einsum for speed
                S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                err = np.sum((S - 1.0)**2)
                
                if err < 1e-10:
                    found = True
                    break
                
                if err > prev_error:
                    # Divergence or oscillation, might need to break or adjust
                    # But least squares step should decrease error for that block.
                    # Total error might increase temporarily? No, block coordinate descent on convex subproblems usually decreases global objective if steps are exact.
                    pass
                prev_error = err
                
            if found:
                # Convert to lists with integer/half-integer rounding
                # The problem asks for integer/half-integer entries.
                # We need to check if the solution can be rounded.
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness with rounded values
                S_rounded = np.einsum('ijr,jkr,kir->ijk', U_int, V_int, W_int)
                if np.allclose(S_rounded, 1.0, atol=1e-9):
                    # Flatten and return
                    U_flat = U_int.reshape(n * m, R).tolist()
                    V_flat = V_int.reshape(m * p, R).tolist()
                    W_flat = W_int.reshape(p * n, R).tolist()
                    return {"U": U_flat, "V": V_flat, "W": W_flat}
                else:
                    # If rounding breaks it, we might need to search for exact integer solution
                    # Or accept that this rank doesn't have a "nice" solution and try next rank?
                    # The problem requires integer/half-integer entries.
                    # If the continuous solution is close to half-integers, rounding works.
                    # If not, this rank might not be achievable with such entries, or we need a different solution.
                    # Let's try to find a solution that is exactly integer/half-integer.
                    # This is a discrete problem.
                    # However, if the continuous solution is very close, it's likely the intended one.
                    # If not, we skip this restart.
                    pass
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
