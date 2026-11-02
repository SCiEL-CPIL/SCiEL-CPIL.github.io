import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]
_START_TIME = time.time()
TIME_LIMIT = 120  # seconds

def _time_remaining():
    return TIME_LIMIT - 10 - (time.time() - _START_TIME)

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) as lists.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # U: A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V: B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W: C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] += 1
    U[3][0] += 1
    V[0][0] += 1
    V[3][0] += 1
    W[0][0] += 1
    W[3][0] += 1
    
    # M2: (A10+A11)B00
    U[2][1] += 1
    U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    W[3][1] -= 1
    
    # M3: A00(B01-B11)
    U[0][2] += 1
    V[1][2] += 1
    V[3][2] -= 1
    W[1][2] += 1
    W[3][2] += 1
    
    # M4: A11(B10-B00)
    U[3][3] += 1
    V[2][3] += 1
    V[0][3] -= 1
    W[0][3] += 1
    W[2][3] += 1
    
    # M5: (A00+A01)B11
    U[0][4] += 1
    U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1
    W[1][4] += 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] += 1
    U[0][5] -= 1
    V[0][5] += 1
    V[1][5] += 1
    W[3][5] += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] += 1
    U[3][6] -= 1
    V[2][6] += 1
    V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _build_decomposed_2x3x3():
    """
    Decompose 2x3x3 into:
    - Part 1: 2x2x2 (columns j=0,1; rows k=0,1) -> Strassen rank 7
    - Part 2: 2x2x1 (columns j=0,1; row k=2) -> schoolbook rank 4
    - Part 3: 2x1x3 (column j=2; rows k=0,1,2) -> schoolbook rank 6
    Total rank: 17
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x2 block
    U1, V1, W1 = _build_strassen_2x2x2()
    for r in range(7):
        for a_idx in range(4):
            i_local = a_idx // 2
            j_local = a_idx % 2
            a_global = i_local * 3 + j_local
            U[a_global][r] = U1[a_idx][r]
        for b_idx in range(4):
            j_local = b_idx // 2
            k_local = b_idx % 2
            b_global = j_local * 3 + k_local
            V[b_global][r] = V1[b_idx][r]
        for c_idx in range(4):
            k_local = c_idx // 2
            i_local = c_idx % 2
            c_global = k_local * 2 + i_local
            W[c_global][r] = W1[c_idx][r]
    
    # Part 2: 2x2x1 block (j in {0,1}, k=2)
    col_offset = 7
    for i in range(2):
        for j in range(2):
            col = col_offset + i * 2 + j
            a_global = i * 3 + j
            b_global = j * 3 + 2
            c_global = 4 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    # Part 3: 2x1x3 block (j=2, k in {0,1,2})
    col_offset = 11
    for i in range(2):
        for k in range(3):
            col = col_offset + i * 3 + k
            a_global = i * 3 + 2
            b_global = 6 + k
            c_global = k * 2 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We check if T matches the multiplication tensor (1 for all i,j,k)
    
    # Vectorized check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=1000):
    """
    Try to reduce rank by finding a lower rank approximation.
    """
    if _time_remaining() < 5:
        return None
        
    U_arr = np.array(U, dtype=float).copy()
    V_arr = np.array(V, dtype=float).copy()
    W_arr = np.array(W, dtype=float).copy()
    
    current_rank = U_arr.shape[1]
    if current_rank <= target_rank:
        return U_arr, V_arr, W_arr
        
    # Simple heuristic: try to zero out columns and see if we can compensate
    # This is a placeholder for a more sophisticated search
    # For now, we just return the original if we can't find a better one quickly
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the known rank-17 decomposition
    try:
        U, V, W = _build_decomposed_2x3x3()
        if _verify_fast(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = len(U[0])
            
            # Try to reduce rank
            # Note: The current search is weak, so we might not improve
            # But we ensure we return a valid solution
            
            return {"U": best_U, "V": best_V, "W": best_W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
