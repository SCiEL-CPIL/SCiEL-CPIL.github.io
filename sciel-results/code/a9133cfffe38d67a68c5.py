"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=2000, seed=42):
        """Try to find a decomposition of rank r using alternating least squares."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    A_sys = np.zeros((p, r))
                    for k in range(p):
                        A_sys[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b_sys = np.ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve linear system
            for j in range(m):
                for k in range(p):
                    # Build system: for each i, sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1
                    A_sys = np.zeros((n, r))
                    for i in range(n):
                        A_sys[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b_sys = np.ones(n)
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (i,k), solve linear system
            for i in range(n):
                for k in range(p):
                    # Build system: for each j, sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
                    A_sys = np.zeros((m, r))
                    for j in range(m):
                        A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b_sys = np.ones(m)
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                    except:
                        pass
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0)**2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-10:
                break
        
        if best_error < 1e-8:
            return best_U, best_V, best_W
        return None
    
    def round_to_half_integers(mat, tol=1e-4):
        """Try to round matrix entries to half-integers and check validity."""
        U_list = [[float(x) for x in row] for row in mat]
        for i in range(len(U_list)):
            for j in range(len(U_list[i])):
                U_list[i][j] = round(U_list[i][j] * 2) / 2.0
        return U_list
    
    # Try ranks from low to high with multiple random restarts
    best_result = None
    best_rank = n * m * p  # Schoolbook rank as fallback
    
    for r in range(24, min(49, n*m*p + 1)):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(42, 42 + 20):  # 20 random restarts per rank
            if time.time() > deadline:
                break
            
            result = try_rank(r, max_iters=500, seed=seed)
            if result is not None:
                U, V, W = result
                
                # Try to round to half-integers
                U_list = round_to_half_integers(U)
                V_list = round_to_half_integers(V)
                W_list = round_to_half_integers(W)
                
                # Verify after rounding
                if check_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), tol=1e-3):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
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
# EVOLVE-BLOCK-END
