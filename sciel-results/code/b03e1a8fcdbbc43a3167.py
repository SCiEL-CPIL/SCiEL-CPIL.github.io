import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    subproducts = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col = 0
    for a_col_off, b_row_off, b_col_off, c_col_off in subproducts:
        for x_coeffs, y_coeffs, z_coeffs in strassen_factors:
            for (x_row, x_col), coeff in x_coeffs.items():
                i = x_row
                j = a_col_off + x_col
                U[i * m + j][col] += coeff
            for (y_row, y_col), coeff in y_coeffs.items():
                j = b_row_off + y_row
                k = b_col_off + y_col
                V[j * p + k][col] += coeff
            for (z_row, z_col), coeff in z_coeffs.items():
                k = c_col_off + z_col
                i = z_row
                W[k * n + i][col] += coeff
            col += 1
    
    # Quick verification with random matrices
    if time.time() < deadline:
        np.random.seed(42)
        A = np.random.randint(-5, 5, size=(n, m)).astype(float)
        B = np.random.randint(-5, 5, size=(m, p)).astype(float)
        C_expected = A @ B
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        # Reconstruct: C[i,k] = sum_c W[k*n+i, c] * ... 
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor: T[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] summed over c
        # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c])... no
        # The decomposition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,j,k} for the multiplication tensor
        # Actually the tensor T[i,j,k] = 1 if this is the multiplication tensor
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c (A[i,j] U[i*m+j,c]) (B[j,k] V[j*p+k,c]) W[k*n+i,c]
        # Wait, that's not right either. Let me think.
        
        # The tensor decomposition means: for all i,j,k:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 (the multiplication tensor has all 1s)
        # And then C[i,k] = sum_j A[i,j] * B[j,k] is recovered by:
        # C[i,k] = sum_c W[k*n+i, c] * sum_j (A[i,j] * U[i*m+j, c]) * (B[j,k] * V[j*p+k, c])
        # No wait, that's a bilinear form.
        
        # Actually the standard tensor decomposition for matrix mult:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] where T[i,j,k]=1
        # Then C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c (A[i,j] U[i,j,c]) (B[j,k] V[j,k,c]) W[i,k,c]
        # Hmm, that doesn't look right dimensionally.
        
        # Let me just verify the tensor identity directly:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k
        tensor_sum = np.einsum('ic,jc,kc->ijk', U_np, V_np, W_np)
        expected_tensor = np.ones((n, m, p))
        if not np.allclose(tensor_sum, expected_tensor, atol=1e-10):
            # Fallback to schoolbook
            rank_fallback = n * m * p
            U_f = [[0] * rank_fallback for _ in range(n * m)]
            V_f = [[0] * rank_fallback for _ in range(m * p)]
            W_f = [[0] * rank_fallback for _ in range(p * n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_f[i * m + j][c] = 1
                        V_f[j * p + k][c] = 1
                        W_f[k * n + i][c] = 1
                        c += 1
            return {"U": U_f, "V": V_f, "W": W_f}
    
    return {"U": U, "V": V, "W": W}
