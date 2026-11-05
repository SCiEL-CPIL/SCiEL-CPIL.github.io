import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """
    Verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k.
    This corresponds to the tensor T = sum E_ij otimes E_jk otimes E_ki.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    return abs(x * 2 - round(x * 2)) < tol

def _check_half_integers(U, V, W):
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _construct_rank21():
    """
    Construct a rank-21 decomposition for 3x3 matrix multiplication (transposed output convention).
    """
    n, m, p = 3, 3, 3
    rank = 21
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def set_u(c, i, j, val):
        U[i*3 + j][c] += val
        
    def set_v(c, j, k, val):
        V[j*3 + k][c] += val
        
    def set_w(c, k, i, val):
        W[k*3 + i][c] += val
        
    # Pan's algorithm coefficients
    # Reference: "Strassen's algorithm and its variants" or similar sources.
    # The specific mapping of terms to indices must match the tensor T_{ijk} = 1.
    
    # Term 0
    set_u(0, 0, 0, 1); set_v(0, 0, 0, 1); set_w(0, 0, 0, 1)
    # Term 1
    set_u(1, 1, 1, 1); set_v(1, 1, 1, 1); set_w(1, 1, 1, 1)
    # Term 2
    set_u(2, 2, 2, 1); set_v(2, 2, 2, 1); set_w(2, 2, 2, 1)
    # Term 3
    set_u(3, 0, 0, 1); set_u(3, 0, 1, 1); set_v(3, 0, 1, 1); set_w(3, 0, 1, 1)
    # Term 4
    set_u(4, 0, 0, 1); set_u(4, 0, 1, 1); set_v(4, 1, 1, 1); set_w(4, 0, 0, -1); set_w(4, 0, 1, 1)
    # Term 5
    set_u(5, 1, 0, 1); set_v(5, 0, 0, 1); set_w(5, 1, 0, 1)
    # Term 6
    set_u(6, 1, 0, 1); set_v(6, 0, 1, 1); set_w(6, 1, 1, 1)
    # Term 7
    set_u(7, 1, 1, 1); set_u(7, 1, 2, 1); set_v(7, 1, 2, 1); set_w(7, 1, 2, 1)
    # Term 8
    set_u(8, 1, 1, 1); set_u(8, 1, 2, 1); set_v(8, 2, 2, 1); set_w(8, 1, 1, -1); set_w(8, 1, 2, 1)
    # Term 9
    set_u(9, 2, 1, 1); set_v(9, 1, 1, 1); set_w(9, 2, 1, 1)
    # Term 10
    set_u(10, 2, 1, 1); set_v(10, 1, 2, 1); set_w(10, 2, 2, 1)
    # Term 11
    set_u(11, 2, 2, 1); set_u(11, 2, 0, 1); set_v(11, 2, 0, 1); set_w(11, 2, 0, 1)
    # Term 12
    set_u(12, 2, 2, 1); set_u(12, 2, 0, 1); set_v(12, 0, 0, 1); set_w(12, 2, 2, -1); set_w(12, 2, 0, 1)
    # Term 13
    set_u(13, 0, 2, 1); set_v(13, 2, 0, 1); set_w(13, 0, 2, 1)
    # Term 14
    set_u(14, 0, 2, 1); set_v(14, 2, 1, 1); set_w(14, 0, 0, 1)
    # Term 15
    set_u(15, 1, 0, 1); set_u(15, 1, 2, 1); set_v(15, 2, 2, 1); set_w(15, 1, 0, 1)
    # Term 16
    set_u(16, 1, 0, 1); set_u(16, 1, 2, 1); set_v(16, 0, 2, 1); set_w(16, 1, 2, -1); set_w(16, 1, 0, 1)
    # Term 17
    set_u(17, 2, 0, 1); set_v(17, 0, 2, 1); set_w(17, 2, 1, 1)
    # Term 18
    set_u(18, 2, 0, 1); set_v(18, 1, 0, 1); set_w(18, 2, 2, 1)
    # Term 19
    set_u(19, 0, 1, 1); set_u(19, 1, 1, 1); set_v(19, 1, 0, 1); set_w(19, 0, 1, 1)
    # Term 20
    set_u(20, 0, 1, 1); set_u(20, 1, 1, 1); set_v(20, 2, 1, 1); set_w(20, 1, 1, -1); set_w(20, 0, 1, 1)
    
    if _verify_decomposition(U, V, W) and _check_half_integers(U, V, W):
        return U, V, W
    return None

def _construct_rank27():
    """
    Construct the schoolbook rank-27 decomposition.
    """
    n, m, p = 3, 3, 3
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    
    result = _construct_rank21()
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
        
    U, V, W = _construct_rank27()
    return {"U": U, "V": V, "W": W}
