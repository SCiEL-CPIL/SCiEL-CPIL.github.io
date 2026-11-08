"""Fresh design for 2x4 by 4x5 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]
DEADLINE = 140.0  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, rank):
    """Verify that U, V, W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = start_time + DEADLINE
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school, "rank": rank_school}
    
    def _solve_for_rank(r):
        """Try to find a decomposition of rank r."""
        if time.time() > time_limit:
            return None
        
        for attempt in range(100):
            if time.time() > time_limit - 5:
                break
            
            U_rand = np.random.randn(n*m, r)
            V_rand = np.random.randn(m*p, r)
            
            W_rand = np.zeros((p*n, r))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    o = k*n + i
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j,:] = U_rand[i*m+j,:] * V_rand[j*p+k,:]
                    b = np.ones(m)
                    
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_rand[o,:] = sol
                        if not np.allclose(A @ sol, b, atol=1e-8):
                            valid = False
                    except:
                        valid = False
            
            if valid:
                U_out = U_rand.tolist()
                V_out = V_rand.tolist()
                W_out = W_rand.tolist()
                
                if _verify(U_out, V_out, W_out, n, m, p, r):
                    return {"U": U_out, "V": V_out, "W": W_out, "rank": r}
        
        return None
    
    # Try ranks from low to high
    for r in range(10, rank_school + 1):
        if time.time() > time_limit - 10:
            break
        
        result = _solve_for_rank(r)
        if result is not None:
            best = result
            break
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}

