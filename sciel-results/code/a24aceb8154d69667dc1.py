import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    # Try to find lower rank decompositions
    # Known lower bound is 15. Try 15 to 23.
    for rank in range(15, 24):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
                
            np.random.seed(seed)
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # ALS iterations
            for _ in range(500):
                if time.time() > deadline:
                    break
                
                # Solve for W
                # Equation: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
                # Vectorize W: w[k*n+i, c]
                # For each (i,j,k), we have a linear equation in w.
                # Coeffs for w are: coeff[c] = U[i*m+j, c] * V[j*p+k, c]
                # We build a matrix A_w of shape (n*m*p, p*n*rank)
                # Rows: index for (i,j,k)
                # Cols: index for (k,i,c) -> (k*n+i)*rank + c
                
                A_w = np.zeros((n * m * p, p * n * rank))
                b_w = np.ones(n * m * p)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Row idx corresponds to (i,j,k)
                            # Coefficients for W[k*n+i, c]
                            row_start = (k * n + i) * rank
                            coeffs = U[i * m + j] * V[j * p + k]
                            A_w[idx, row_start:row_start + rank] = coeffs
                            idx += 1
                
                # Solve A_w w = b_w
                w_sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W = w_sol.reshape(p * n, rank)
                
                # Solve for U
                # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Coeffs for U[i*m+j, c] are: V[j*p+k, c] * W[k*n+i, c]
                A_u = np.zeros((n * m * p, n * m * rank))
                b_u = np.ones(n * m * p)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_start = (i * m + j) * rank
                            coeffs = V[j * p + k] * W[k * n + i]
                            A_u[idx, row_start:row_start + rank] = coeffs
                            idx += 1
                
                u_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = u_sol.reshape(n * m, rank)
                
                # Solve for V
                # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Coeffs for V[j*p+k, c] are: U[i*m+j, c] * W[k*n+i, c]
                A_v = np.zeros((n * m * p, m * p * rank))
                b_v = np.ones(n * m * p)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_start = (j * p + k) * rank
                            coeffs = U[i * m + j] * W[k * n + i]
                            A_v[idx, row_start:row_start + rank] = coeffs
                            idx += 1
                
                v_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = v_sol.reshape(m * p, rank)
                
                # Check residual
                # Residual for each (i,j,k)
                # res = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1
                # We can compute this efficiently
                # For each c, compute M_c = U[:,c] * V[:,c] * W[:,c]? No.
                # Let's just compute the sum directly for the final check.
                # For intermediate checks, we can approximate or skip.
                # Let's do a quick check every 10 iterations to avoid overhead.
                
            # Final check and snap
            # Compute residuals
            residuals = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        sum_val = 0
                        for c in range(rank):
                            sum_val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        residuals += (sum_val - 1) ** 2
            
            if residuals < 1e-6:
                # Snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify snapped solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            sum_val = 0
                            for c in range(rank):
                                sum_val += U_snapped[i * m + j, c] * V_snapped[j * p + k, c] * W_snapped[k * n + i, c]
                            if abs(sum_val - 1) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to lists
                    U_list = U_snapped.tolist()
                    V_list = V_snapped.tolist()
                    W_list = W_snapped.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    # Return fallback
    return best_result
