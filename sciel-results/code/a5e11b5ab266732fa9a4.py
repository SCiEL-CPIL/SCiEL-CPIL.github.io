import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 7]

def strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    n, m, p = 2, 2, 2
    R = 7
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen formulas for 2x2x2
    # U: (i,j) -> combinations
    # V: (j,k) -> combinations  
    # W: (k,i) -> combinations
    
    # M1 = (A11+A22)(B11+B22)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1; W[1,0] = -1; W[2,0] = -1
    
    # M2 = (A21+A22)B12
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = 1
    
    # M3 = A11(B21-B11)
    U[0,2] = 1
    V[2,2] = 1; V[0,2] = -1
    W[0,2] = 1; W[2,2] = 1
    
    # M4 = A22(B12-B22)
    U[3,3] = 1
    V[1,3] = 1; V[3,3] = -1
    W[1,3] = 1; W[3,3] = 1
    
    # M5 = (A11+A12)B22
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[2,4] = 1; W[3,4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1; W[2,5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[1,6] = 1; W[3,6] = 1
    
    return U, V, W

def get_decomp(n, m, p):
    """
    Get a decomposition for n x m times m x p.
    Uses recursive Strassen-like approach where possible.
    """
    # Base cases: small problems use schoolbook or known optimal
    if n == 1 and m == 1 and p == 1:
        R = 1
        U = np.ones((1, R))
        V = np.ones((1, R))
        W = np.ones((1, R))
        return U, V, W
    
    if n == 1 and m == 1:
        R = p
        U = np.ones((1, R))
        V = np.eye(p).flatten().reshape(p, R)
        W = np.eye(p).flatten().reshape(p, R)
        return U, V, W
    
    if n == 1 and p == 1:
        R = m
        U = np.eye(m).flatten().reshape(m, R)
        V = np.ones((m, R))
        W = np.ones((1, R))
        return U, V, W
    
    if m == 1:
        R = n * p
        U = np.zeros((n, R))
        V = np.zeros((p, R))
        W = np.zeros((p, R))
        col = 0
        for i in range(n):
            for k in range(p):
                U[i, col] = 1
                V[k, col] = 1
                W[k, col] = 1
                col += 1
        return U, V, W
    
    # For 2x2x2, use Strassen
    if n == 2 and m == 2 and p == 2:
        return strassen_2x2x2()
    
    # For 2x2x3, try to combine two 2x2x2 with one extra
    if n == 2 and m == 2 and p == 3:
        # Use Strassen for first 2 columns, schoolbook for 3rd
        U1, V1, W1 = strassen_2x2x2()
        # Add 3rd column
        R = 7 + 2  # Strassen rank + schoolbook for remaining
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        U[:, :7] = U1
        V[:, :7] = V1
        W[:, :7] = W1
        
        # Schoolbook for k=2 (0-indexed, so 3rd column)
        col = 7
        for i in range(2):
            for j in range(2):
                U[i*2+j, col] = 1
                V[j*2+2, col] = 1
                W[2*2+i, col] = 1
                col += 1
        return U, V, W
    
    # For 3x3x3, use known optimal rank 23 (or try to find)
    if n == 3 and m == 3 and p == 3:
        # Use schoolbook as fallback, rank 27
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # General case: use schoolbook
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Target tensor is all 1s
    # T[i, j, k] = 1
    
    best_result = None
    best_rank = n * m * p # 63
    
    # Schoolbook is always valid
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Get a good initial decomposition using recursive approach
    U_init_rec, V_init_rec, W_init_rec = get_decomp(n, m, p)
    R_init = U_init_rec.shape[1]
    
    # Search for lower rank
    # We will try to find integer/half-integer solutions
    # by optimizing floats and rounding
    
    # Start from min(R_init, 62) down to 30
    max_search_rank = min(R_init, 62)
    for R in range(max_search_rank, 29, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations with different strategies
        n_attempts = 8 if R < 50 else 5
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
                
            # Different initialization strategies
            if attempt == 0:
                # Use recursive decomposition padded/truncated to rank R
                if R >= R_init:
                    U_init = np.zeros((n*m, R))
                    V_init = np.zeros((m*p, R))
                    W_init = np.zeros((p*n, R))
                    U_init[:, :R_init] = U_init_rec
                    V_init[:, :R_init] = V_init_rec
                    W_init[:, :R_init] = W_init_rec
                else:
                    U_init = U_init_rec[:, :R].copy()
                    V_init = V_init_rec[:, :R].copy()
                    W_init = W_init_rec[:, :R].copy()
            elif attempt == 1:
                # Random small integers
                U_init = np.random.randint(-1, 2, size=(n*m, R))
                V_init = np.random.randint(-1, 2, size=(m*p, R))
                W_init = np.random.randint(-1, 2, size=(p*n, R))
            elif attempt == 2:
                # Random in {-1, 0, 0.5, 1}
                vals = np.array([-1, -0.5, 0, 0.5, 1])
                U_init = vals[np.random.randint(0, len(vals), size=(n*m, R))]
                V_init = vals[np.random.randint(0, len(vals), size=(m*p, R))]
                W_init = vals[np.random.randint(0, len(vals), size=(p*n, R))]
            elif attempt == 3:
                # Gaussian with small std
                U_init = np.random.randn(n*m, R) * 0.5
                V_init = np.random.randn(m*p, R) * 0.5
                W_init = np.random.randn(p*n, R) * 0.5
            elif attempt == 4:
                # Sparse random
                U_init = np.random.choice([0, 1, -1], size=(n*m, R), p=[0.7, 0.15, 0.15])
                V_init = np.random.choice([0, 1, -1], size=(m*p, R), p=[0.7, 0.15, 0.15])
                W_init = np.random.choice([0, 1, -1], size=(p*n, R), p=[0.7, 0.15, 0.15])
            else:
                # Perturb the recursive solution
                if R >= R_init:
                    U_init = np.zeros((n*m, R))
                    V_init = np.zeros((m*p, R))
                    W_init = np.zeros((p*n, R))
                    U_init[:, :R_init] = U_init_rec + np.random.randn(n*m, R_init) * 0.1
                    V_init[:, :R_init] = V_init_rec + np.random.randn(m*p, R_init) * 0.1
                    W_init[:, :R_init] = W_init_rec + np.random.randn(p*n, R_init) * 0.1
                else:
                    U_init = U_init_rec[:, :R].copy() + np.random.randn(n*m, R) * 0.1
                    V_init = V_init_rec[:, :R].copy() + np.random.randn(m*p, R) * 0.1
                    W_init = W_init_rec[:, :R].copy() + np.random.randn(p*n, R) * 0.1
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residuals(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Efficiently using einsum
                # U shape (n, m, R)
                # V shape (m, p, R)
                # W shape (p, n, R)
                U_3d = U.reshape(n, m, R)
                V_3d = V.reshape(m, p, R)
                W_3d = W.reshape(p, n, R)
                
                # Sum over r: U_{ijr} V_{jkr} W_{kir}
                # Note: W index is k*n + i, so W[k, i, r]
                # We want T_{ijk} = sum_r U_{ijr} V_{jkr} W_{kir}
                
                # Use einsum
                # 'ijr,jkr,kir->ijk'
                try:
                    prod = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                except Exception:
                    return np.ones(n*m*p)
                
                # Target is 1
                return (prod - 1).flatten()
            
            try:
                res = least_squares(residuals, x0, method='lm', max_nfev=2000)
            except Exception:
                continue
                
            if res.success or np.linalg.norm(res.fun) < 1e-5:
                # Check if we can round to integers/half-integers
                U_opt = res.x[:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Try multiple rounding strategies
                found = False
                for scale in [2, 4, 1]:
                    U_round = np.round(U_opt * scale) / scale
                    V_round = np.round(V_opt * scale) / scale
                    W_round = np.round(W_opt * scale) / scale
                    
                    # Verify exactness
                    U_3d = U_round.reshape(n, m, R)
                    V_3d = V_round.reshape(m, p, R)
                    W_3d = W_round.reshape(p, n, R)
                    
                    try:
                        prod = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                    except Exception:
                        continue
                    
                    if np.allclose(prod, 1, atol=1e-9):
                        # Convert to list of lists of ints/floats
                        U_list = [[int(x) if x == int(x) else float(x) for x in row] for row in U_round]
                        V_list = [[int(x) if x == int(x) else float(x) for x in row] for row in V_round]
                        W_list = [[int(x) if x == int(x) else float(x) for x in row] for row in W_round]
                        
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = R
                        found = True
                        break
                
                if found:
                    # Found a solution for this rank, continue to lower R
                    break
        
        # Continue to next (lower) R
    
    return best_result
