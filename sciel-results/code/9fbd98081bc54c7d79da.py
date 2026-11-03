import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor T
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T is 1 if j_a == j_b and i_a == i_c and k_b == k_c
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for a in range(dim_u):
        i_a = a // m
        j_a = a % m
        for b in range(dim_v):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(dim_w):
                k_c = c // n
                i_c = c % n
                if i_a == i_c and k_b == k_c:
                    T[a, b, c] = 1.0

    # Try to find a decomposition with rank R
    # We try R from 26 to 30
    for R in range(26, 31):
        # Try a few random initializations
        for trial in range(20):
            # Random integer initialization for U and V
            # Use small integers to increase chance of integer W
            U = np.random.randint(-1, 2, size=(dim_u, R))
            V = np.random.randint(-1, 2, size=(dim_v, R))
            
            # Ensure no zero columns
            for r in range(R):
                if np.all(U[:, r] == 0):
                    U[np.random.randint(dim_u), r] = 1
                if np.all(V[:, r] == 0):
                    V[np.random.randint(dim_v), r] = 1
            
            # Construct the linear system for W
            # Equation: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
            # Let's reshape to a matrix equation
            # We have 900 equations and R*dim_w unknowns? No.
            # W is (dim_w, R).
            # Unknowns are W[c, r].
            # Let's form the matrix A_sys of size (dim_u * dim_v * dim_w) x (dim_w * R)
            # This is too big to form explicitly if we do it naively, but 900x280 is small.
            
            # Actually, we can solve for each c independently?
            # No, the coefficients U[a, r] V[b, r] are shared across c.
            # But the equations are coupled.
            
            # Let's use least squares on the whole system.
            # Vectorize W: w_vec of size dim_w * R.
            # w_vec[c * R + r] = W[c, r].
            
            # A_sys[eq_idx, var_idx]
            # eq_idx corresponds to (a, b, c).
            # var_idx corresponds to (c', r).
            # The term is non-zero only if c == c'.
            # So A_sys is block diagonal with respect to c?
            # Let's check:
            # sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
            # For a fixed c, the equation involves only W[c, :].
            # So we can solve for each c independently!
            
            valid = True
            W = np.zeros((dim_w, R))
            for c in range(dim_w):
                # Extract T[:, :, c] of shape (dim_u, dim_v)
                T_c = T[:, :, c]
                # We need to solve: sum_r U[a, r] V[b, r] W[c, r] = T_c[a, b]
                # This is a linear system in W[c, :]
                # Let's form the matrix M of size (dim_u * dim_v) x R
                # M[eq_idx, r] = U[a, r] * V[b, r]
                M = np.zeros((dim_u * dim_v, R))
                for a in range(dim_u):
                    for b in range(dim_v):
                        eq_idx = a * dim_v + b
                        for r in range(R):
                            M[eq_idx, r] = U[a, r] * V[b, r]
                
                # Solve M * W_c = T_c_flat
                T_c_flat = T_c.flatten()
                
                # Use least squares
                try:
                    W_c, residuals, rank, s = np.linalg.lstsq(M, T_c_flat, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if solution is integer (close to integer)
                W_c_int = np.round(W_c)
                if not np.allclose(W_c, W_c_int, atol=1e-9):
                    valid = False
                    break
                
                W[c, :] = W_c_int
            
            if valid:
                # Verify the solution
                # Reconstruct T'
                T_prime = np.zeros((dim_u, dim_v, dim_w))
                for a in range(dim_u):
                    for b in range(dim_v):
                        for c in range(dim_w):
                            T_prime[a, b, c] = np.dot(U[a, :], V[b, :] * W[c, :])
                
                if np.allclose(T_prime, T, atol=1e-9):
                    # Convert to lists of integers
                    U_list = U.astype(int).tolist()
                    V_list = V.astype(int).tolist()
                    W_list = W.astype(int).tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
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
