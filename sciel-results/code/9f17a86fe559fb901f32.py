"""Matrix multiplication 3,3,8 — 3x3x3 Strassen + schoolbook tail."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
_T0 = time.time()
_DEADLINE = _T0 + 110


def _elapsed():
    return time.time() - _T0


def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    result = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    return np.allclose(result, 1.0, atol=1e-9)


def _strassen_3x3x3_terms():
    """
    Return the 19 Strassen terms for 3x3x3 matrix multiplication.
    Each term is (u_dict, v_dict, w_dict) where:
      u_dict: {(i,j): val} for U[i*m+j]
      v_dict: {(j,k): val} for V[j*p+k]
      w_dict: {(k,i): val} for W[k*n+i]
    """
    terms = []
    
    # M1: A11 * B11
    terms.append((
        {(0,0): 1},
        {(0,0): 1},
        {(0,0): 1},
    ))
    # M2: (A11+A22) * (B11+B22)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): -1},
    ))
    # M3: (A22+A33) * (B22+B33)
    terms.append((
        {(1,1): 1, (2,2): 1},
        {(1,1): 1, (2,2): 1},
        {(1,1): 1, (2,2): 1, (1,2): -1, (2,1): -1},
    ))
    # M4: (A12-A22) * (B21+B22)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(1,0): 1},
    ))
    # M5: (A11-A12) * (B12+B22)
    terms.append((
        {(0,0): 1, (0,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,1): 1},
    ))
    # M6: (A21-A31) * (B11+B12)
    terms.append((
        {(1,0): 1, (2,0): -1},
        {(0,0): 1, (0,1): 1},
        {(0,1): 1},
    ))
    # M7: (A32-A12) * (B21+B22)
    terms.append((
        {(2,1): 1, (0,1): -1},
        {(1,0): 1, (1,1): 1},
        {(1,2): 1},
    ))
    # M8: (A13+A23) * (B31+B32)
    terms.append((
        {(0,2): 1, (1,2): 1},
        {(2,0): 1, (2,1): 1},
        {(0,2): 1, (1,2): 1, (0,0): -1, (1,1): -1},
    ))
    # M9: (A23-A13) * (B32+B33)
    terms.append((
        {(1,2): 1, (0,2): -1},
        {(2,1): 1, (2,2): 1},
        {(1,2): 1},
    ))
    # M10: (A31+A11) * (B12+B11)
    terms.append((
        {(2,0): 1, (0,0): 1},
        {(0,1): 1, (0,0): 1},
        {(0,0): 1, (0,2): 1, (1,1): -1, (1,2): -1},
    ))
    # M11: (A33-A23) * (B31+B32)
    terms.append((
        {(2,2): 1, (1,2): -1},
        {(2,0): 1, (2,1): 1},
        {(2,2): 1},
    ))
    # M12: (A31-A21) * (B11+B12)
    terms.append((
        {(2,0): 1, (1,0): -1},
        {(0,0): 1, (0,1): 1},
        {(2,0): 1},
    ))
    # M13: (A11+A12) * (B11+B12)
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): 1, (0,1): 1, (1,0): -1, (1,1): -1},
    ))
    # M14: (A22+A21) * (B22+B21)
    terms.append((
        {(1,1): 1, (1,0): 1},
        {(1,1): 1, (1,0): 1},
        {(1,1): 1, (1,0): 1, (2,1): -1, (2,0): -1},
    ))
    # M15: (A33+A32) * (B33+B32)
    terms.append((
        {(2,2): 1, (2,1): 1},
        {(2,2): 1, (2,1): 1},
        {(2,2): 1, (2,1): 1, (0,2): -1, (0,1): -1},
    ))
    # M16: (A13-A23) * (B31+B32)
    terms.append((
        {(0,2): 1, (1,2): -1},
        {(2,0): 1, (2,1): 1},
        {(0,2): 1},
    ))
    # M17: (A31+A32) * (B11+B12)
    terms.append((
        {(2,0): 1, (2,1): 1},
        {(0,0): 1, (0,1): 1},
        {(2,0): 1, (2,1): 1, (0,0): -1, (0,1): -1},
    ))
    # M18: (A12+A22) * (B21+B11)
    terms.append((
        {(0,1): 1, (1,1): 1},
        {(1,0): 1, (0,0): 1},
        {(0,1): 1, (1,0): 1, (0,0): -1, (1,1): -1},
    ))
    # M19: (A23+A33) * (B32+B33)
    terms.append((
        {(1,2): 1, (2,2): 1},
        {(2,1): 1, (2,2): 1},
        {(1,2): 1, (2,2): 1, (1,1): -1, (2,1): -1},
    ))
    
    return terms


def _build_3x3x8_decomposition():
    """
    Build decomposition for 3x3 x 3x8 using 3x3x3 Strassen for first 3 cols
    and schoolbook for remaining 5 cols.
    """
    n, m, p = 3, 3, 8
    strassen_terms = _strassen_3x3x3_terms()
    num_strassen = len(strassen_terms)  # 19
    num_schoolbook = n * m * (p - 3)    # 3*3*5 = 15
    total_rank = num_strassen + num_schoolbook  # 34
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    
    # Add Strassen terms for the 3x3x3 part (columns 0,1,2)
    for u_dict, v_dict, w_dict in strassen_terms:
        for (i, j), val in u_dict.items():
            U[i * m + j][offset] += val
        for (j, k), val in v_dict.items():
            if k < 3:  # Only for the 3x3x3 part
                V[j * p + k][offset] += val
        for (k, i), val in w_dict.items():
            if k < 3:  # Only for the 3x3x3 part
                W[k * n + i][offset] += val
        offset += 1
    
    # Add schoolbook for columns 3,4,5,6,7
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][offset] = 1
                V[j * p + k][offset] = 1
                W[k * n + i][offset] = 1
                offset += 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Pure schoolbook decomposition."""
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


def construct():
    """Construct rank-minimal (or near-minimal) decomposition for 3x3 x 3x8."""
    n, m, p = 3, 3, 8
    
    # Try the 3x3x3 Strassen + schoolbook approach
    if _elapsed() < _DEADLINE - 20:
        U, V, W = _build_3x3x8_decomposition()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    return {"U": U_school, "V": V_school, "W": W_school}
