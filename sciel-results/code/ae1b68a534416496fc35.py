"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=500):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize
        for c in range(r):
            norm = np.sqrt(np.sum(U[:,c]**2) * np.sum(V[:,c]**2) * np.sum(W[:,c]**2))
            if norm > 1e-10:
                U[:,c] /= norm**(1/3)
                V[:,c] /= norm**(1/3)
                W[:,c] /= norm**(1/3)
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Alternating minimization
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is a linear system
            
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                    A_sys = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A_sys[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    b_sys = np.ones(p)
                    try:
                        U[i*m+j, :], residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A_sys = np.zeros((n*m, r))
                    for i in range(n):
                        for jj in range(m):
                            for c in range(r):
                                A_sys[i*m+jj, c] = U[i*m+jj, c] * W[k*n+i, c] if jj == j else 0
                    # This is getting complicated. Let me use a different approach.
                    
                    # Actually, for V[j*p+k, :], we need:
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                    # So: for each i, sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1
                    A_sys = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A_sys[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    b_sys = np.ones(n)
                    try:
                        V[j*p+k, :], residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    except:
                        pass
            
            # Update W
            for i in range(n):
                for k in range(p):
                    # For W[k*n+i, :], we need:
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                    A_sys = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_sys[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    b_sys = np.ones(m)
                    try:
                        W[k*n+i, :], residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
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
            
            if error < 1e-8:
                break
        
        if best_error < 1e-6:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from low to high
    for r in range(24, 49):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=200)
        if result is not None:
            U, V, W = result
            # Convert to lists with integer/half-integer entries if possible
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Try to round to half-integers
            for lst in [U_list, V_list, W_list]:
                for i in range(len(lst)):
                    for j in range(len(lst[i])):
                        lst[i][j] = round(lst[i][j] * 2) / 2.0
            
            # Verify
            if check_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), tol=1e-4):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
