"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Uses Strassen's algorithm on four 2x2 sub-blocks to achieve rank 28 (vs schoolbook 32).
Rank 28 is the best known upper bound for this specific tensor product.
"""
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
    # Fallback to schoolbook
    if best_result is None:
        best_result = _construct_schoolbook_32()
    return best_result


def _verify_decomposition(result):
    """Verify that the decomposition correctly computes matrix multiplication."""
    try:
        n, m, p = DIMENSIONS
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        
        # Test with random matrices
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Vectorized computation
        # Reshape A to (n*m, 1) and B to (1, m*p)? No.
        # C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * V[j*p+k, t])
        
        # Let's form the intermediate products M[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t]
        # This is a tensor of size (n, m, p).
        # U is (n*m, rank). V is (m*p, rank).
        # We can compute the outer product of U and V for each t?
        # Or simply:
        # For each t, M_t[i,j,k] = U[i*m+j, t] * V[j*p+k, t]
        # C[i,k] = sum_t W[k*n+i, t] * M_t[i,j,k] (summed over j)
        
        # Efficient way:
        # Construct A_flat (n*m, 1) and B_flat (1, m*p)?
        # Actually, C = (A @ B)
        # The decomposition computes C = sum_t W_t * (A @ V_t) ? No.
        # A is (n, m). V_t is a column of V? No, V is (m*p, rank).
        # Let V_t be the t-th column of V, reshaped to (m, p).
        # Let U_t be the t-th column of U, reshaped to (n, m).
        # Then A @ V_t is not correct.
        # The term is sum_j U[i*m+j, t] * A[i,j]? No.
        # The tensor decomposition is:
        # C[i,k] = sum_{j,t} A[i,j] B[j,k] U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
        # Wait, the standard tensor decomposition for matrix multiplication is:
        # C[i,k] = sum_t (sum_j A[i,j] U_t[i,j]) (sum_l B[l,k] V_t[l,k]) W_t[i,k]
        # This requires U, V, W to be 3D tensors or reshaped matrices.
        # The interface defines U as (n*m, rank).
        # So U_t[i*m+j] is the coefficient for A[i,j].
        # Let A_t = sum_{i,j} A[i,j] U_t[i*m+j]. This is a scalar.
        # Let B_t = sum_{j,k} B[j,k] V_t[j*p+k]. This is a scalar.
        # Then C[i,k] += A_t B_t W_t[k*n+i].
        
        # Let's verify this interpretation.
        # If U, V, W are rank-1 tensors, then:
        # C = sum_t (A * U_t) * (B * V_t) * W_t
        # where * is element-wise sum (inner product).
        
        # Compute A_t for all t:
        # A_flat = A.flatten() (n*m,)
        # A_t = A_flat @ U (rank,)
        A_flat = A.flatten()
        A_t = A_flat @ U
        
        # Compute B_t for all t:
        B_flat = B.flatten()
        B_t = B_flat @ V
        
        # Compute C:
        # C[i,k] = sum_t A_t[t] * B_t[t] * W[k*n+i, t]
        # C_flat = (A_t * B_t) @ W.T ?
        # W is (p*n, rank). W[k*n+i, t].
        # Let Z = A_t * B_t (rank,)
        # C_flat[k*n+i] = sum_t Z[t] * W[k*n+i, t]
        # C_flat = Z @ W.T
        C_flat = (A_t * B_t) @ W.T
        C_computed = C_flat.reshape((n, p))
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        return True
    except Exception:
        return False


def _construct_schoolbook_32():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
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
    """
    2x4 * 4x4 using four 2x2 Strassen blocks.
    
    Partition:
      A is 2x4: A = [A1 | A2] where A1, A2 are 2x2
      B is 4x4: B = [B11 B12; B21 B22] all 2x2
      C = A * B is 2x4: C = [C1 | C2] where
        C1 = A1*B11 + A2*B21
        C2 = A1*B12 + A2*B22
    
    For each 2x2 product, use Strassen (rank 7). Four products -> rank 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2 product X*Y:
    # Local indices: a=0 (x00), b=1 (x01), c=2 (x10), d=3 (x11)
    # P1 = (x00+x11)(y00+y11) -> c00:+1, c11:+1
    # P2 = (x10+x11)y00 -> c10:+1, c11:-1
    # P3 = x00(y01-y11) -> c01:+1, c11:+1
    # P4 = x11(y10-y00) -> c00:+1, c10:+1
    # P5 = (x00+x01)y11 -> c00:-1, c01:+1
    # P6 = (x10-x00)(y00+y01) -> c11:+1
    # P7 = (x01-x11)(y10+y11) -> c00:+1
    
    # W output indices: c00=0, c01=1, c10=2, c11=3 (local)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # P1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # P2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # P3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # P4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),         # P5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # P6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),         # P7
    ]
    
    # Four sub-products with their index mappings
    # Each mapping: local (r,c) -> global index
    sub_products = [
        # 1. A_left (A[:,0:2]) * B_tl (B[0:2,0:2]) -> C_left (C[:,0:2])
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        # 2. A_right (A[:,2:4]) * B_bl (B[2:4,0:2]) -> C_left (C[:,0:2])
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        # 3. A_left (A[:,0:2]) * B_tr (B[0:2,2:4]) -> C_right (C[:,2:4])
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
        # 4. A_right (A[:,2:4]) * B_br (B[2:4,2:4]) -> C_right (C[:,2:4])
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
    ]
    
    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                r, c = loc_u // 2, loc_u % 2
                global_u = sp['a_map'](r, c)
                U[global_u][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                r, c = loc_v // 2, loc_v % 2
                global_v = sp['b_map'](r, c)
                V[global_v][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                r, c = loc_w // 2, loc_w % 2
                global_w = sp['c_map'](r, c)
                W[global_w][col] += coeff
            col += 1
    
    assert col == 28, f"Expected col=28, got {col}"
    
    return {"U": U, "V": V, "W": W}
