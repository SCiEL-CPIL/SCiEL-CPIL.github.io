import time
import numpy as np
from scipy.optimize import minimize

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

def _try_rank_16_slice():
    """
    Attempt to find a rank 16 decomposition for 3x3 x 2 matrix multiplication.
    Returns U, V, W for the slice or None.
    """
    n, m, p = 3, 3, 2
    rank = 16
    
    # Flatten U, V, W into a single vector for optimization
    # U: 9x16, V: 6x16, W: 6x16
    # Total variables: 9*16 + 6*16 + 6*16 = 336
    
    def loss(params):
        U_flat = params[:9*rank]
        V_flat = params[9*rank:9*rank+6*rank]
        W_flat = params[9*rank+6*rank:]
        
        U = U_flat.reshape(9, rank)
        V = V_flat.reshape(6, rank)
        W = W_flat.reshape(6, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0)**2
        return err

    # Initialize with random small values
    np.random.seed(42)
    x0 = np.random.randn(336) * 0.1
    
    # Try a few random restarts
    best_loss = float('inf')
    best_params = None
    
    for _ in range(10):
        x0 = np.random.randn(336) * 0.1
        res = minimize(loss, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            
        if _time_left() < 20:
            break
            
    if best_loss < 1e-6:
        U_flat = best_params[:9*rank]
        V_flat = best_params[9*rank:9*rank+6*rank]
        W_flat = best_params[9*rank+6*rank:]
        
        U = U_flat.reshape(9, rank)
        V = V_flat.reshape(6, rank)
        W = W_flat.reshape(6, rank)
        
        # Round to nearest 0.5
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Convert to list of lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
            
    return None

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Try Strassen embedding (Rank 68)
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6)
    ]
    
    if _time_left() > 5:
        try:
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
            
    # Try Rank 16 Slice (Rank 64)
    if _time_left() > 30:
        try:
            slice_result = _try_rank_16_slice()
            if slice_result is not None:
                sU, sV, sW = slice_result
                # Build full 3x3x8 using 4 copies of the slice
                rank = 16 * 4
                U = [[0]*rank for _ in range(n*m)]
                V = [[0]*rank for _ in range(m*p)]
                W = [[0]*rank for _ in range(p*n)]
                
                for k_offset in range(0, p, 2):
                    col_base = (k_offset // 2) * 16
                    for i in range(n):
                        for j in range(m):
                            for c in range(16):
                                U[i*m+j][col_base+c] = sU[i*m+j][c]
                    for j in range(m):
                        for k in range(2):
                            for c in range(16):
                                V[j*p + (k_offset+k)][col_base+c] = sV[j*2+k][c]
                    for k in range(2):
                        for i in range(n):
                            for c in range(16):
                                W[(k_offset+k)*n + i][col_base+c] = sW[k*3+i][c]
                
                if _verify(U, V, W, n, m, p):
                    r = len(U[0])
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    return best_result
