import time
import numpy as np

def _find_3x3_decomposition(rank, max_attempts=500, seed=42):
    """
    Try to find a rank-r decomposition for 3x3x3 matrix multiplication.
    Uses randomized approach: fix U and V randomly, solve for W.
    Returns (U, V, W) as lists of lists, or None if not found.
    """
    rng = np.random.RandomState(seed)
    
    for attempt in range(max_attempts):
        U = rng.randn(9, rank)
        V = rng.randn(9, rank)
        
        W = np.zeros((9, rank))
        valid = True
        for k in range(3):
            for i in range(3):
                A = np.zeros((3, rank))
                for j in range(3):
                    for c in range(rank):
                        A[j, c] = U[i*3+j, c] * V[j*3+k, c]
                
                try:
                    W_row, residuals, rank_val, sv = np.linalg.lstsq(A, np.ones(3), rcond=None)
                    if np.allclose(A @ W_row, np.ones(3), atol=1e-10):
                        W[k*3+i, :] = W_row
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            ok = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                        if abs(s - 1) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                U_list = [[float(U[r, c]) for c in range(rank)] for r in range(9)]
                V_list = [[float(V[r, c]) for c in range(rank)] for r in range(9)]
                W_list = [[float(W[r, c]) for c in range(rank)] for r in range(9)]
                return U_list, V_list, W_list
    
    return None

def _try_rank_decomposition(n, m, p, rank, start_time, deadline_margin=30):
    """
    Try to find a rank-r decomposition for n x m * m x p matrix multiplication.
    Returns (U, V, W) or None.
    """
    if n == 3 and m == 3 and p == 6:
        # Try to find rank-23 decomposition for 3x3x3, then duplicate
        result = _find_3x3_decomposition(23, max_attempts=200, seed=1)
        if result is not None:
            U3, V3, W3 = result
            rank_total = 46
            U_full = [[0.0] * rank_total for _ in range(9)]
            V_full = [[0.0] * rank_total for _ in range(18)]
            W_full = [[0.0] * rank_total for _ in range(18)]
            
            for c in range(23):
                for i in range(3):
                    for j in range(3):
                        U_full[i*3+j][c] = U3[i*3+j][c]
                for j in range(3):
                    for k1 in range(3):
                        V_full[j*3+k1][c] = V3[j*3+k1][c]
                for k1 in range(3):
                    for i in range(3):
                        W_full[k1*3+i][c] = W3[k1*3+i][c]
            
            for c in range(23):
                c_full = 23 + c
                for i in range(3):
                    for j in range(3):
                        U_full[i*3+j][c_full] = U3[i*3+j][c]
                for j in range(3):
                    for k2 in range(3):
                        V_full[j*6 + 3 + k2][c_full] = V3[j*3+k2][c]
                for k2 in range(3):
                    for i in range(3):
                        W_full[(3+k2)*3 + i][c_full] = W3[k2*3+i][c]
            
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U_full[i*m+j][c] * V_full[j*p+k][c] * W_full[k*n+i][c] for c in range(rank_total))
                        if abs(s - 1) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                return U_full, V_full, W_full
        
        for r in [22, 21, 20, 19]:
            if time.time() > start_time + (150 - deadline_margin):
                break
            result = _find_3x3_decomposition(r, max_attempts=100, seed=r*100+1)
            if result is not None:
                U3, V3, W3 = result
                rank_total = 2 * r
                U_full = [[0.0] * rank_total for _ in range(9)]
                V_full = [[0.0] * rank_total for _ in range(18)]
                W_full = [[0.0] * rank_total for _ in range(18)]
                
                for c in range(r):
                    for i in range(3):
                        for j in range(3):
                            U_full[i*3+j][c] = U3[i*3+j][c]
                    for j in range(3):
                        for k1 in range(3):
                            V_full[j*3+k1][c] = V3[j*3+k1][c]
                    for k1 in range(3):
                        for i in range(3):
                            W_full[k1*3+i][c] = W3[k1*3+i][c]
                
                for c in range(r):
                    c_full = r + c
                    for i in range(3):
                        for j in range(3):
                            U_full[i*3+j][c_full] = U3[i*3+j][c]
                    for j in range(3):
                        for k2 in range(3):
                            V_full[j*6 + 3 + k2][c_full] = V3[j*3+k2][c]
                    for k2 in range(3):
                        for i in range(3):
                            W_full[(3+k2)*3 + i][c_full] = W3[k2*3+i][c]
                
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U_full[i*m+j][c] * V_full[j*p+k][c] * W_full[k*n+i][c] for c in range(rank_total))
                            if abs(s - 1) > 1e-9:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    return U_full, V_full, W_full
        
        return None
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy: Try to find low-rank decompositions for 3x3x3, then combine
    for 3x3x6. Fall back to schoolbook (rank 54) if no better found.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Try to find improved decomposition
    result = _try_rank_decomposition(n, m, p, 54, start_time, deadline_margin=30)
    
    if result is not None:
        U, V, W = result
        rank = len(U[0])
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
