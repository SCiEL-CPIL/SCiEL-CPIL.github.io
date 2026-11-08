import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, r) = (10, r)
    # V: (m*p, r) = (25, r)
    # W: (p*n, r) = (10, r)
    
    # Try ranks from low to high
    for r in range(5, 51):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(1000):
            if time.time() > deadline - 2:
                break
            
            # Random half-integer initialization
            # Use small values to keep numbers manageable
            U = (np.random.randint(-4, 5, size=(n*m, r)) + np.random.randint(0, 2, size=(n*m, r))) / 2.0
            V = (np.random.randint(-4, 5, size=(m*p, r)) + np.random.randint(0, 2, size=(m*p, r))) / 2.0
            
            # For each (i,k), solve for W[k*2+i, :]
            W = np.zeros((p*n, r))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build 5×r system: for j=0..4, sum_c U[i*5+j,c]*V[j*5+k,c]*W[k*2+i,c] = 1
                    A_mat = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_mat[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    
                    # Solve A_mat @ W[k*2+i, :] = ones(5)
                    try:
                        # Use least squares if r > m, or direct solve if r == m
                        if r == m:
                            W_row = np.linalg.solve(A_mat, np.ones(m))
                        else:
                            W_row, residuals, rank, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                            if rank < m:
                                valid = False
                                break
                        W[k*n + i, :] = W_row
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition
            # Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        if abs(val - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Convert to lists with half-integer rounding
                U_list = [[round(x * 2) / 2 for x in row] for row in U]
                V_list = [[round(x * 2) / 2 for x in row] for row in V]
                W_list = [[round(x * 2) / 2 for x in row] for row in W]
                
                # Re-verify with rounded values
                ok2 = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c] for c in range(r))
                            if abs(val - 1.0) > 1e-9:
                                ok2 = False
                                break
                        if not ok2:
                            break
                    if not ok2:
                        break
                
                if ok2:
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
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
