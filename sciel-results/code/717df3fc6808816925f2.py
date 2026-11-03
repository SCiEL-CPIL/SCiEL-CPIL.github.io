import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 3, 3, 6
    # Dimensions of the flattened vectors
    dim_a = n * m  # 9
    dim_b = m * p  # 18
    dim_c = n * p  # 18
    
    # Build the tensor T
    # T[i, j, k] = 1 if a_j b_k contributes to c_i
    # i = x*p + z
    # j = x*m + y
    # k = y*p + z
    T = np.zeros((dim_c, dim_a, dim_b))
    for x in range(n):
        for y in range(m):
            for z in range(p):
                i = x * p + z
                j = x * m + y
                k = y * p + z
                T[i, j, k] = 1.0
                
    # Reshape T to a matrix for easier handling?
    # T is a 3-way tensor. We want T = sum_r u_r v_r w_r.
    # This is equivalent to finding U, V, W such that:
    # sum_r U[:, i, r] V[:, j, r] W[:, k, r] = T[i, j, k]
    # Or, for each r, the matrix M_r = u_r v_r^T w_r^T? No.
    # The equation is:
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    
    def residuals(params, R):
        # params: flattened U, V, W
        # U: dim_c x R
        # V: dim_a x R
        # W: dim_b x R
        U = params[:dim_c * R].reshape(dim_c, R)
        V = params[dim_c * R : dim_c * R + dim_a * R].reshape(dim_a, R)
        W = params[dim_c * R + dim_a * R : dim_c * R + dim_a * R + dim_b * R].reshape(dim_b, R)
        
        # Compute sum_r U[i, r] V[j, r] W[k, r]
        # This is the tensor product sum.
        # We can compute this as:
        # result[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # This can be done by iterating over r or using einsum.
        # np.einsum('ir,jr,kr->ijk', U, V, W)
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return (computed - T).flatten()
        
    # Try ranks from 30 down to 10
    for R in range(30, 9, -1):
        # Multiple random restarts
        for restart in range(20):
            start_time = time.time()
            if start_time - time.time() > 10: # Simple guard, though not strict
                break
                
            # Random initialization
            np.random.seed(restart)
            params0 = np.random.randn(dim_c * R + dim_a * R + dim_b * R)
            
            try:
                res = least_squares(residuals, params0, args=(R,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.success or res.cost < 1e-6:
                # Check if we can round to integers/half-integers
                U = res.x[:dim_c * R].reshape(dim_c, R)
                V = res.x[dim_c * R : dim_c * R + dim_a * R].reshape(dim_a, R)
                W = res.x[dim_c * R + dim_a * R : dim_c * R + dim_a * R + dim_b * R].reshape(dim_b, R)
                
                # Try rounding to integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                # Check exactness
                computed_int = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
                if np.allclose(computed_int, T, atol=1e-9):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    
                # Try rounding to half-integers
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                
                computed_half = np.einsum('ir,jr,kr->ijk', U_half, V_half, W_half)
                if np.allclose(computed_half, T, atol=1e-9):
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * p)]
    V = [[0] * rank for _ in range(n * m)]
    W = [[0] * rank for _ in range(m * p)]
    # Note: The task description says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # My U, V, W above are (n*p, R), (n*m, R), (m*p, R).
    # Let's re-read the task carefully.
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # This matches the standard:
    # U: n*m x R
    # V: m*p x R
    # W: p*n x R
    # And the equation is:
    # sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k]
    # Where i is index of A (n*m), j is index of B (m*p), k is index of C (p*n)?
    # Wait, C is n x p. So index of C is n*p.
    # But W is p*n x R.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This suggests W is indexed by (k, i) where k is col of B (0..p-1) and i is row of A (0..n-1).
    # So W is p*n x R.
    # And the equation is:
    # sum_r U[i, r] V[j, r] W[k, r] = ?
    # The product is C[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor T has indices (i, j, k) corresponding to (A, B, C).
    # T[i, j, k] = 1 if A[i, j] B[j, k] contributes to C[i, k].
    # This is always true for valid i, j, k.
    # So T[i, j, k] = 1 for all i, j, k?
    # No, the indices are:
    # i: index of A (0..n*m-1)
    # j: index of B (0..m*p-1)
    # k: index of C (0..p*n-1) -- but C is n x p, so index is i*p + k.
    # The task says W is p*n x R, indexed by k*n + i.
    # So the third index in the sum is k*n + i.
    # The equation is:
    # sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k]
    # Where T[i, j, k] is 1 if the term A[i] B[j] contributes to C[k].
    # A[i] is A[row, col] where i = row*m + col.
    # B[j] is B[row, col] where j = row*p + col.
    # C[k] is C[row, col] where k = col*n + row (based on W[k*n+i]).
    # Wait, W[k*n+i] means the index is k*n + i.
    # So the third index in the tensor is k*n + i.
    # Let's check the contribution.
    # A[row, col] * B[col, col2] contributes to C[row, col2].
    # i = row*m + col
    # j = col*p + col2
    # k_index = col2*n + row
    # So T[i, j, k_index] = 1 if i, j, k_index correspond to the same row, col, col2.
    # Otherwise 0.
    
    # My previous T construction was:
    # i = x*p + z (index of C)
    # j = x*m + y (index of A)
    # k = y*p + z (index of B)
    # This was for the equation C[i, k] = sum_j A[i, j] B[j, k].
    # But the task specifies U, V, W shapes and the W index convention.
    # Let's stick to the task's convention.
    # U: n*m x R
    # V: m*p x R
    # W: p*n x R
    # Equation: sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k]
    # Where T[i, j, k] = 1 if A[i] B[j] contributes to C[k].
    # i = row_A * m + col_A
    # j = col_A * p + col_B  (since B is m x p, row_B = col_A)
    # k = col_B * n + row_A  (since W is p*n x R, index is k*n + i? No, W[k*n+i] means index = k*n + i.
    # Wait, if W is p*n x R, and index is k*n + i, then k is 0..p-1 and i is 0..n-1.
    # So the third index in the sum is k*n + i.
    # Let's call it l = k*n + i.
    # So T[i, j, l] = 1 if A[i] B[j] contributes to C[l].
    # A[i] = A[row_A, col_A]
    # B[j] = B[row_B, col_B]
    # C[l] = C[row_C, col_C]
    # Contribution if row_A = row_C, col_A = row_B, col_B = col_C.
    # l = col_C * n + row_C.
    # So l = col_B * n + row_A.
    
    # Let's rebuild T with this convention.
    T_new = np.zeros((n*m, m*p, p*n))
    for row_A in range(n):
        for col_A in range(m):
            for col_B in range(p):
                i = row_A * m + col_A
                j = col_A * p + col_B
                l = col_B * n + row_A
                T_new[i, j, l] = 1.0
                
    # Now solve for U, V, W with this T.
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n  # Note: dim_c is now p*n, not n*p
    
    def residuals_new(params, R):
        U = params[:dim_a * R].reshape(dim_a, R)
        V = params[dim_a * R : dim_a * R + dim_b * R].reshape(dim_b, R)
        W = params[dim_a * R + dim_b * R : dim_a * R + dim_b * R + dim_c * R].reshape(dim_c, R)
        
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return (computed - T_new).flatten()
        
    for R in range(30, 9, -1):
        for restart in range(20):
            np.random.seed(restart)
            params0 = np.random.randn(dim_a * R + dim_b * R + dim_c * R)
            
            try:
                res = least_squares(residuals_new, params0, args=(R,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.success or res.cost < 1e-6:
                U = res.x[:dim_a * R].reshape(dim_a, R)
                V = res.x[dim_a * R : dim_a * R + dim_b * R].reshape(dim_b, R)
                W = res.x[dim_a * R + dim_b * R : dim_a * R + dim_b * R + dim_c * R].reshape(dim_c, R)
                
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                computed_int = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
                if np.allclose(computed_int, T_new, atol=1e-9):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                
                computed_half = np.einsum('ir,jr,kr->ijk', U_half, V_half, W_half)
                if np.allclose(computed_half, T_new, atol=1e-9):
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                
    # Fallback
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
