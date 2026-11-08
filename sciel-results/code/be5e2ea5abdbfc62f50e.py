import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 140  # Leave 10s margin for the 150s limit

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    Uses integer arithmetic to avoid floating point issues.
    """
    r = len(U[0])
    # Convert to numpy arrays for fast computation
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    # Reshape
    U_r = U_arr.reshape(n, m, r)
    V_r = V_arr.reshape(m, p, r)
    W_r = W_arr.reshape(p, n, r)
    
    # Compute reconstruction
    # Result[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    try:
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    except Exception:
        return False
    
    return np.all(reconstructed == 1)

def _verify_decomposition_float(U, V, W, n, m, p):
    """
    Fast float verification. Returns True if likely valid (needs exact check).
    """
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_r = U_arr.reshape(n, m, r)
    V_r = V_arr.reshape(m, p, r)
    W_r = W_arr.reshape(p, n, r)
    
    try:
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    except Exception:
        return False
    
    # Check if close to 1
    return np.allclose(reconstructed, 1.0, atol=1e-6)

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (A10+A11)(B00) -> C10 += 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C[1,0]
    
    # m3 = A00*(B01-B11) -> C01 += 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1  # C[0,1]
    
    # m4 = A11*(B10-B00) -> C00 += 1, C10 += 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1  # C[0,0] and C[1,0]
    
    # m5 = (A00+A01)*B11 -> C00 -= 1, C01 += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1  # C[0,0] -= 1, C[0,1] += 1
    
    # m6 = (A10-A00)*(B00+B01) -> C11 += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C[1,1]
    
    # m7 = (A01-A11)*(B10+B11) -> C00 += 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C[0,0]
    
    return U, V, W

def _strassen_2x4_4x4():
    """
    Construct a rank-28 decomposition for 2x4 * 4x4 using four Strassen 2x2*2x2 decompositions.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 8x28
    V = [[0]*rank for _ in range(m*p)]  # 16x28
    W = [[0]*rank for _ in range(p*n)]  # 8x28
    
    # Get Strassen 2x2*2x2 decomposition
    S_U, S_V, S_W = _strassen_2x2()
    
    # For M1 = A_L * B11:
    for c in range(7):
        base = 0  # M1 uses rank indices 0-6
        for row in range(4):
            U[row][base + c] += S_U[row][c]
            V[row][base + c] += S_V[row][c]
            W[row][base + c] += S_W[row][c]
    
    # For M2 = A_R * B21:
    for c in range(7):
        base = 7  # M2 uses rank indices 7-13
        for row in range(4):
            U[4 + row][base + c] += S_U[row][c]
            V[8 + row][base + c] += S_V[row][c]
            W[row][base + c] += S_W[row][c]
    
    # For M3 = A_L * B12:
    for c in range(7):
        base = 14  # M3 uses rank indices 14-20
        for row in range(4):
            U[row][base + c] += S_U[row][c]
            V[4 + row][base + c] += S_V[row][c]
            W[4 + row][base + c] += S_W[row][c]
    
    # For M4 = A_R * B22:
    for c in range(7):
        base = 21  # M4 uses rank indices 21-27
        for row in range(4):
            U[4 + row][base + c] += S_U[row][c]
            V[12 + row][base + c] += S_V[row][c]
            W[4 + row][base + c] += S_W[row][c]
    
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(r, n, m, p, time_limit, start_time):
    """
    Try to find a rank-r decomposition using random search.
    """
    best_result = None
    
    # Number of random trials
    trials = 500
    
    for trial in range(trials):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with small integers
        U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
        
        # Quick float check
        if not _verify_decomposition_float(U, V, W, n, m, p):
            continue
        
        # Exact check
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # First try the structured Strassen-based decomposition
    start_time = time.time()
    
    # Try rank 28 (four Strassen 2x2*2x2)
    result_28 = _strassen_2x4_4x4()
    if _verify_decomposition(result_28["U"], result_28["V"], result_28["W"], n, m, p):
        best_rank = 28
        best_result = result_28
    else:
        # Fallback to schoolbook
        best_result = _schoolbook()
        best_rank = rank_schoolbook
    
    # Try to find rank 26 (theoretical minimum for 2x4x4)
    if best_rank > 26:
        result_26 = _try_rank_decomposition(26, n, m, p, TIME_LIMIT, start_time)
        if result_26 is not None:
            best_rank = 26
            best_result = result_26
    
    # Try to find rank 25 (unlikely but possible if 26 is not minimal)
    if best_rank > 25:
        result_25 = _try_rank_decomposition(25, n, m, p, TIME_LIMIT, start_time)
        if result_25 is not None:
            best_rank = 25
            best_result = result_25

    return best_result
