import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 9
    dim_V = m * p  # 21
    dim_W = p * n  # 21
    
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def tensor_from_factors(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def is_close_to_half_integer(x, tol=1e-5):
        return np.all(np.abs(x - np.round(2*x)/2) < tol)

    def try_rank(R, max_iters=2000, tol=1e-8, seed=0):
        np.random.seed(seed)
        U = np.random.randn(dim_U, R) * 0.1
        V = np.random.randn(dim_V, R) * 0.1
        W = np.random.randn(dim_W, R) * 0.1
        
        # Structured initialization for the last 9 columns (schoolbook for k=6)
        if R >= 9:
            # The last 9 columns correspond to B_{:, 6}
            # U[:, 9:18] etc? No, the columns are just indices 0..R-1.
            # We can assign the last 9 columns to be the schoolbook factors for k=6.
            # But which columns of U, V, W correspond to which k?
            # In the tensor sum, each column r contributes T_r = U_r V_r W_r.
            # We want the sum of the last 9 rank-1 tensors to be the tensor for k=6.
            # The tensor for k=6 is T_6[i, j, 6] = 1 if a=i*m+j, b=j*p+6, c=6*n+i.
            # This is exactly the schoolbook decomposition for that slice.
            # So we can set U[:, R-9:R], V[:, R-9:R], W[:, R-9:R] to the schoolbook factors for k=6.
            
            # Schoolbook factors for k=6:
            # For each i, j:
            # a = i*m + j
            # b = j*p + 6
            # c = 6*n + i
            # U[a, col] = 1
            # V[b, col] = 1
            # W[c, col] = 1
            
            # We need to map the 9 (i,j) pairs to the 9 columns.
            col_offset = R - 9
            for idx, (i, j) in enumerate([(i, j) for i in range(n) for j in range(m)]):
                a = i * m + j
                b = j * p + 6
                c = 6 * n + i
                U[a, col_offset + idx] = 1.0
                V[b, col_offset + idx] = 1.0
                W[c, col_offset + idx] = 1.0
                # Zero out other entries in these columns?
                # The random initialization already put values there.
                # We should probably zero out the non-target entries to make it a valid rank-1 tensor for that slice.
                # But ALS will adjust them.
                # Actually, if we set them to 1, and others to 0, it's a valid rank-1 tensor for that slice.
                # Let's zero out the rest of the columns.
                U[:, col_offset + idx] = 0.0
                V[:, col_offset + idx] = 0.0
                W[:, col_offset + idx] = 0.0
                U[a, col_offset + idx] = 1.0
                V[b, col_offset + idx] = 1.0
                W[c, col_offset + idx] = 1.0

        for it in range(max_iters):
            # Update U
            M_U = np.einsum('jr,kr->jkr', V, W).reshape(dim_V*dim_W, R)
            T_2 = T.reshape(dim_U, dim_V*dim_W)
            try:
                U = np.linalg.lstsq(M_U.T, T_2.T, rcond=None)[0].T
            except:
                pass
            
            # Update V
            M_V = np.einsum('ir,kr->ikr', U, W).reshape(dim_U*dim_W, R)
            T_2 = T.transpose(1, 0, 2).reshape(dim_V, dim_U*dim_W)
            try:
                V = np.linalg.lstsq(M_V.T, T_2.T, rcond=None)[0].T
            except:
                pass
            
            # Update W
            M_W = np.einsum('ir,jr->ijr', U, V).reshape(dim_U*dim_V, R)
            T_2 = T.transpose(2, 0, 1).reshape(dim_W, dim_U*dim_V)
            try:
                W = np.linalg.lstsq(M_W.T, T_2.T, rcond=None)[0].T
            except:
                pass
            
            T_est = tensor_from_factors(U, V, W)
            err = np.linalg.norm(T - T_est)
            if err < tol:
                return U, V, W, err
        
        return None, None, None, np.inf

    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = 63
    best_factors = None
    
    # Schoolbook fallback
    U_school = np.zeros((dim_U, 63), dtype=float)
    V_school = np.zeros((dim_V, 63), dtype=float)
    W_school = np.zeros((dim_W, 63), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1

    # Try rank 55
    # Use multiple seeds
    for R in [55]:
        for seed in range(10):
            if time.time() > deadline:
                break
            U, V, W, err = try_rank(R, max_iters=1000, tol=1e-8, seed=seed)
            if U is not None and err < 1e-6:
                if is_close_to_half_integer(U) and is_close_to_half_integer(V) and is_close_to_half_integer(W):
                    U = np.round(2*U)/2
                    V = np.round(2*V)/2
                    W = np.round(2*W)/2
                    T_est = tensor_from_factors(U, V, W)
                    if np.allclose(T, T_est, atol=1e-5):
                        best_rank = R
                        best_factors = (U, V, W)
                        break
        if best_rank < 63:
            break

    if best_factors is not None:
        U, V, W = best_factors
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
