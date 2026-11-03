import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    n, m, p = 3, 4, 6
    
    # Build the rank-66 decomposition
    U66 = [[0] * 66 for _ in range(n * m)]
    V66 = [[0] * 66 for _ in range(m * p)]
    W66 = [[0] * 66 for _ in range(p * n)]
    
    col = 0
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1
        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1
        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1
        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
        col += 1
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1
    
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)
    
    # Convert to numpy
    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)
    
    # Verify rank-66 decomposition
    def verify(U, V, W):
        # Check: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 if j==k else 0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for l in range(U.shape[1]):
                        s += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                    expected = 1.0 if j == k else 0.0
                    max_err = max(max_err, abs(s - expected))
        return max_err
    
    err66 = verify(U, V, W)
    if err66 > 1e-10:
        # Fallback: return rank-66
        return {"U": U66, "V": V66, "W": W66}
    
    # Try to reduce to rank 65 by finding a column that's a linear combination of others
    # For each candidate column c to eliminate, check if the "flattened" tensor contribution
    # of column c can be written as a linear combination of the other columns' contributions.
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = delta_{j,k}
    # Each rank-1 term l contributes: outer(U[:,l], V[:,l], W[:,l])
    # We want: outer(U[:,c], V[:,c], W[:,c]) = sum_{l != c} alpha_l * outer(U[:,l], V[:,l], W[:,l])
    
    # This is a system in the alpha_l's. Flatten each outer product to a vector.
    # The tensor has n*m * m*p * p*n = 12 * 24 * 18 = 5184 entries.
    # But many are zero by structure.
    
    # Better approach: for each (i,j,k) with j==k, we need:
    # U[i*m+j, c] * V[j*p+j, c] * W[j*n+i, c] = sum_{l != c} alpha_l * U[i*m+j, l] * V[j*p+j, l] * W[j*n+i, l]
    # And for j != k, we need:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = sum_{l != c} alpha_l * U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    
    # This gives 18*4*6 = 432 equations in 65 unknowns (alpha_l for l != c).
    # If the system is consistent, we can eliminate column c.
    
    # But we also need to update U, V, W to absorb the alpha_l's.
    # Actually, if column c = sum alpha_l * column l in the tensor sense, then we can just
    # drop column c and the remaining 65 columns still span the tensor.
    
    # Let's try each column as a candidate for elimination.
    rank = 66
    best_U, best_V, best_W = U, V, W
    
    for c in range(66):
        if time.time() > deadline - 5:
            break
        
        # Build the system: for each (i,j,k), the equation is:
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = sum_{l != c} alpha_l * U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
        
        # Collect equations
        A_rows = []
        b_vals = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(lhs) < 1e-15 and all(abs(U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]) < 1e-15 for l in range(66) if l != c):
                        continue  # trivially satisfied
                    row = np.zeros(65)
                    idx = 0
                    for l in range(66):
                        if l == c:
                            continue
                        row[idx] = U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
                        idx += 1
                    A_rows.append(row)
                    b_vals.append(lhs)
        
        if not A_rows:
            # All zero, can eliminate
            # Remove column c
            keep = [l for l in range(66) if l != c]
            U_new = U[:, keep]
            V_new = V[:, keep]
            W_new = W[:, keep]
            err = verify(U_new, V_new, W_new)
            if err < 1e-10:
                rank = 65
                best_U, best_V, best_W = U_new, V_new, W_new
                break
            continue
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        
        # Solve least squares
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if the solution actually works
        # Reconstruct: does column c equal sum alpha_l * column l in tensor sense?
        # Verify by checking the residual
        A_alpha = A @ alpha
        residual = np.max(np.abs(A_alpha - b))
        
        if residual < 1e-10:
            # We can eliminate column c
            keep = [l for l in range(66) if l != c]
            U_new = U[:, keep]
            V_new = V[:, keep]
            W_new = W[:, keep]
            err = verify(U_new, V_new, W_new)
            if err < 1e-10:
                rank = 65
                best_U, best_V, best_W = U_new, V_new, W_new
                break
    
    # Also try a more aggressive approach: use ALS to find a rank-65 solution
    # Initialize with rank-66, zero out one column, and optimize
    if rank == 66:
        # Try zeroing each column and running ALS
        for c in range(66):
            if time.time() > deadline - 5:
                break
            
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            U_try[:, c] = 0
            V_try[:, c] = 0
            W_try[:, c] = 0
            
            # Run a few ALS iterations
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        A_mat = np.zeros((p, 65))
                        b_vec = np.zeros(p)
                        for k in range(p):
                            for l in range(65):
                                A_mat[k, l] = V_try[j*p+k, l] * W_try[k*n+i, l]
                            b_vec[k] = 1.0 if j == k else 0.0
                        sol, res, r, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U_try[u_idx, :65] = sol
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        A_mat = np.zeros((n, 65))
                        b_vec = np.zeros(n)
                        for i in range(n):
                            for l in range(65):
                                A_mat[i, l] = U_try[i*m+j, l] * W_try[k*n+i, l]
                            b_vec[i] = 1.0 if j == k else 0.0
                        sol, res, r, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V_try[v_idx, :65] = sol
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        A_mat = np.zeros((m, 65))
                        b_vec = np.zeros(m)
                        for j in range(m):
                            for l in range(65):
                                A_mat[j, l] = U_try[i*m+j, l] * V_try[j*p+k, l]
                            b_vec[j] = 1.0 if j == k else 0.0
                        sol, res, r, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W_try[w_idx, :65] = sol
                
                # Check convergence
                err = verify(U_try[:, :65], V_try[:, :65], W_try[:, :65])
                if err < 1e-10:
                    rank = 65
                    best_U = U_try[:, :65]
                    best_V = V_try[:, :65]
                    best_W = W_try[:, :65]
                    break
            if rank == 65:
                break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
