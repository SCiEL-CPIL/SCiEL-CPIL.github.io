import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()

def check_time_and_return(best_result):
    if best_result is not None and (time.time() - START_TIME) > TIME_LIMIT - 10.0:
        return best_result
    return None

def construct():
    best_result = None
    try:
        best_result = _construct_strassen_28()
        if best_result is not None:
            if _verify_decomposition(best_result):
                return best_result
    except Exception:
        pass
    if best_result is None:
        best_result = _construct_schoolbook_32()
    return best_result

def _verify_decomposition(result, trials=5):
    import random
    n, m, p = DIMENSIONS
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = 0.0
            for i in range(n):
                for j in range(m):
                    u_r += U[i * m + j, r] * A[i, j]
            v_r = 0.0
            for j in range(m):
                for k in range(p):
                    v_r += V[j * p + k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W[k * n + i, r] * u_r * v_r
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _construct_schoolbook_32():
    n, m, p = DIMENSIONS
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

def _construct_strassen_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definition
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # P1 = (x00+x11)(y00+y11)
    # P2 = (x10+x11)y00
    # P3 = x00(y01-y11)
    # P4 = x11(y10-y00)
    # P5 = (x00+x01)y11
    # P6 = (x10-x00)(y00+y01)
    # P7 = (x01-x11)(y10+y11)
    
    # Z = [[z00, z01], [z10, z11]]
    # z00 = P1 + P4 - P5 + P7
    # z01 = P3 + P5
    # z10 = P2 + P4
    # z11 = P1 - P2 + P3 + P6
    
    # Each term is a tuple (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {local_idx: coeff} for X
    # v_coeffs: dict {local_idx: coeff} for Y
    # w_coeffs: dict {local_idx: coeff} for Z
    # local_idx: 0=x00, 1=x01, 2=x10, 3=x11
    
    strassen_terms = [
        # P1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    col = 0
    
    # Define the 4 sub-products
    # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
    # Note: for W, index is k*n + i. So we need to be careful with "row" and "col" of C.
    # C is n x p. i is row (0..1), k is col (0..3).
    # W index = k * n + i.
    # In sub-product definitions, we map local C row r (0,1) to global i, and local C col c (0,1) to global k.
    
    sub_products = [
        # 1. A1 * B11 -> C1
        # A1: rows 0-1, cols 0-1. Global A[i][j] -> local X[r][c] where i=r, j=c.
        # B11: rows 0-1, cols 0-1. Global B[j][k] -> local Y[r][c] where j=r, k=c.
        # C1: rows 0-1, cols 0-1. Global C[i][k] -> local Z[r][c] where i=r, k=c.
        # W index for global C[i][k] is k*2 + i.
        # local Z[r][c] corresponds to i=r, k=c. W index = c*2 + r.
        
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        # 2. A2 * B21 -> C1
        # A2: rows 0-1, cols 2-3. Global A[i][j] -> local X[r][c] where i=r, j=c+2.
        # B21: rows 2-3, cols 0-1. Global B[j][k] -> local Y[r][c] where j=r+2, k=c.
        # C1: rows 0-1, cols 0-1. Global C[i][k] -> local Z[r][c] where i=r, k=c.
        # W index = c*2 + r.
        
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        # 3. A1 * B12 -> C2
        # A1: rows 0-1, cols 0-1. Global A[i][j] -> local X[r][c] where i=r, j=c.
        # B12: rows 0-1, cols 2-3. Global B[j][k] -> local Y[r][c] where j=r, k=c+2.
        # C2: rows 0-1, cols 2-3. Global C[i][k] -> local Z[r][c] where i=r, k=c+2.
        # W index = k*2 + i = (c+2)*2 + r.
        
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
        # 4. A2 * B22 -> C2
        # A2: rows 0-1, cols 2-3. Global A[i][j] -> local X[r][c] where i=r, j=c+2.
        # B22: rows 2-3, cols 2-3. Global B[j][k] -> local Y[r][c] where j=r+2, k=c+2.
        # C2: rows 0-1, cols 2-3. Global C[i][k] -> local Z[r][c] where i=r, k=c+2.
        # W index = (c+2)*2 + r.
        
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
    ]
    
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                r = loc_u // 2
                c = loc_u % 2
                global_u = sp['a_map'](r, c)
                U[global_u][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                r = loc_v // 2
                c = loc_v % 2
                global_v = sp['b_map'](r, c)
                V[global_v][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                r = loc_w // 2
                c = loc_w % 2
                global_w = sp['c_map'](r, c)
                W[global_w][col] += coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
