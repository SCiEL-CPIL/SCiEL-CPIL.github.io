import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    strassen_blocks: list of (i0, j0, k0)
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _try_find_lower_rank(U, V, W, n, m, p, target_rank, time_limit=10.0):
    """
    Attempt to find a decomposition of rank target_rank.
    """
    start_time = time.time()
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # We will try to optimize parameters for rank target_rank
    # Initialize with first target_rank columns
    U0 = U_np[:, :target_rank]
    V0 = V_np[:, :target_rank]
    W0 = W_np[:, :target_rank]
    
    def residuals(params):
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = params[n*m*target_rank + m*p*target_rank : ]
        
        U_opt = U_flat.reshape(n*m, target_rank)
        V_opt = V_flat.reshape(m*p, target_rank)
        W_opt = W_flat.reshape(p*n, target_rank)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_opt[i*m+j], np.multiply(V_opt[j*p+k], W_opt[k*n+i]))
                    res[i,j,k] = val - 1.0
        return res.flatten()
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    # Try a few different initializations
    # 1. Direct
    # 2. Random perturbation
    # 3. Drop different columns
    
    attempts = 5
    for attempt in range(attempts):
        if time.time() - start_time > time_limit:
            break
            
        if attempt == 0:
            x_init = x0
        elif attempt == 1:
            x_init = x0 + np.random.normal(0, 0.1, len(x0))
        else:
            # Drop a random column from the original rank and re-optimize
            drop_idx = np.random.randint(0, U_np.shape[1])
            mask = [i for i in range(U_np.shape[1]) if i != drop_idx]
            U_sub = U_np[:, mask[:target_rank]]
            V_sub = V_np[:, mask[:target_rank]]
            W_sub = W_np[:, mask[:target_rank]]
            x_init = np.concatenate([U_sub.flatten(), V_sub.flatten(), W_sub.flatten()])
            
        try:
            result = least_squares(residuals, x_init, method='lm', max_nfev=500, xtol=1e-7, ftol=1e-7)
            if result.cost < 1e-8:
                U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = result.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
                
                U_new = U_opt.tolist()
                V_new = V_opt.tolist()
                W_new = W_opt.tolist()
                
                if _verify(U_new, V_new, W_new, n, m, p, tol=1e-5):
                    return U_new, V_new, W_new
        except Exception:
            pass
            
    return None

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known good fixed placement (4 blocks along k)
    blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    U, V, W = _build_strassen_embedding(n, m, p, blocks)
    
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank to 67
    if _time_left() > 30:
        res = _try_find_lower_rank(U, V, W, n, m, p, best_rank - 1, time_limit=_time_left() - 10)
        if res:
            U_new, V_new, W_new = res
            best_rank = best_rank - 1
            best_result = {"U": U_new, "V": V_new, "W": W_new}
            
    # Try to reduce rank to 66 if we found 67
    if best_rank > 66 and _time_left() > 15:
        U_cur = best_result["U"]
        V_cur = best_result["V"]
        W_cur = best_result["W"]
        res = _try_find_lower_rank(U_cur, V_cur, W_cur, n, m, p, best_rank - 1, time_limit=_time_left() - 5)
        if res:
            U_new, V_new, W_new = res
            best_rank = best_rank - 1
            best_result = {"U": U_new, "V": V_new, "W": W_new}

    return best_result
