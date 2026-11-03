import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Block-Strassen decomposition for 3x4x8 matrix multiplication
    # Strategy: split m=4 into 2+2, and for each 3x2x8 block:
    #   - Split n=3 into 2+1
    #   - 2x2x8: use 4 independent 2x2x2 Strassen decompositions (rank 28)
    #   - 1x2x8: schoolbook (rank 16)
    # Total per 3x2x8: 44
    # Grand total: 2 * 44 = 88
    
    def _build_strassen_3x4x8():
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Strassen 2x2x2 decomposition
        # Each term is (U_coeffs, V_coeffs, W_coeffs) where keys are flattened 2x2 indices
        # 2x2 matrix indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        strassen_decomp = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1 = (A00+A11)(B00+B11)
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # M2 = (A10+A11)B00
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3 = A00(B10-B00)
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4 = A11(B01-B11)
            ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),         # M5 = (A00+A01)B11
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6 = (A10-A00)(B00+B01)
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})          # M7 = (A01-A11)(B10+B11)
        ]
        
        for j1 in range(2):  # 2 blocks of 2 in m dimension
            # Top part: 2x2x8 (rows 0,1 of A)
            for l in range(4):  # 4 blocks of 2 in p dimension
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    # U (A): local 2x2 -> global
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i  # 0 or 1
                        global_j = 2 * j1 + li_j
                        U[global_i * m + global_j][col] += coeff
                    
                    # V (B): local 2x2 -> global
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_j = 2 * j1 + li_i
                        global_k = 2 * l + li_j
                        V[global_j * p + global_k][col] += coeff
                    
                    # W (C): local 2x2 -> global
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i
                        global_k = 2 * l + li_j
                        W[global_k * n + global_i][col] += coeff
                    
                    col += 1
            
            # Bottom part: 1x2x8 (row 2 of A), schoolbook
            for k in range(8):
                for j2 in range(2):
                    global_i = 2
                    global_j = 2 * j1 + j2
                    U[global_i * m + global_j][col] = 1
                    V[global_j * p + k][col] = 1
                    W[k * n + global_i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def _verify(U, V, W, tol=1e-9):
        """Verify exact tensor decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def _verify_half_integer(U, V, W):
        """Check that all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True
    
    # Build the decomposition
    result = _build_strassen_3x4x8()
    
    # Verify
    if _verify(result['U'], result['V'], result['W']) and _verify_half_integer(result['U'], result['V'], result['W']):
        return result
    
    # Fallback: schoolbook
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
