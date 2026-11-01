import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy:
    # 1. Construct a rank-54 decomposition using Strassen on 2x2x2 blocks.
    # 2. Check for linear dependencies among the rank-1 tensors to reduce rank.
    # 3. If a dependency allows removing a term, return the reduced rank decomposition.
    # 4. Otherwise, return the rank-54 decomposition.

    # Step 1: Construct the rank-54 decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_standard(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    def add_strassen(j1, j2, k1, k2):
        nonlocal col
        u00 = 0*m + j1
        u01 = 0*m + j2
        u10 = 1*m + j1
        u11 = 1*m + j2
        
        v00 = j1*p + k1
        v01 = j1*p + k2
        v10 = j2*p + k1
        v11 = j2*p + k2
        
        w00 = k1*n + 0
        w01 = k2*n + 0
        w10 = k1*n + 1
        w11 = k2*n + 1
        
        # P1 = (a00+a11)(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # P2 = (a10+a11)b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] += -1
        col += 1
        
        # P3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] += -1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # P4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] += -1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # P5 = (a00+a01)b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1; W[w01][col] += 1
        col += 1
        
        # P6 = (a10-a00)(b00+b01)
        U[u10][col] += 1; U[u00][col] += -1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # P7 = (a01-a11)(b10+b11)
        U[u01][col] += 1; U[u11][col] += -1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    # Apply Strassen to 2x2x2 blocks
    # j splits: (0,1), (2,3). k splits: (0,1), (2,3), (4,5)
    # Remaining j=4 is handled by standard terms.
    
    add_strassen(0, 1, 0, 1)
    add_strassen(0, 1, 2, 3)
    add_strassen(0, 1, 4, 5)
    add_strassen(2, 3, 0, 1)
    add_strassen(2, 3, 2, 3)
    add_strassen(2, 3, 4, 5)
    
    # Standard terms for j=4
    for i in range(2):
        for k in range(6):
            add_standard(i, 4, k)

    # Verify rank 54
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify(U_mat, V_mat, W_mat):
        U_v = np.array(U_mat, dtype=float)
        V_v = np.array(V_mat, dtype=float)
        W_v = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_v[i * m + j, :]
                    v_row = V_v[j * p + k, :]
                    w_row = W_v[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to trivial rank 60
        rank_f = 60
        U_f = [[0] * rank_f for _ in range(n * m)]
        V_f = [[0] * rank_f for _ in range(m * p)]
        W_f = [[0] * rank_f for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    if time.time() > deadline - 20:
        return {"U": U, "V": V, "W": W}

    # Step 2: Check for rank reduction
    # Vectorize the rank-1 tensors
    # t_r[i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Flatten to 1D vector of size 60
    
    def vec_tensor(U_r, V_r, W_r):
        # U_r, V_r, W_r are 1D arrays of lengths n*m, m*p, p*n
        # We need to form the 3D tensor and flatten
        # But we can do it more efficiently by iterating over non-zero entries
        # Or just use numpy broadcasting
        # U_r is (n*m,), V_r is (m*p,), W_r is (p*n,)
        # Reshape to (n,m), (m,p), (p,n)
        u_3d = U_r.reshape(n, m)
        v_3d = V_r.reshape(m, p)
        w_3d = W_r.reshape(p, n)
        # Tensor product
        # T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        # This is a bit tricky with simple broadcasting if we want 3D.
        # Let's just loop for simplicity since dimensions are small.
        t = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t[i*m*p + j*p + k] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        return t

    M_cols = []
    for r in range(rank):
        U_r = np.array(U[:, r], dtype=float)
        V_r = np.array(V[:, r], dtype=float)
        W_r = np.array(W[:, r], dtype=float)
        M_cols.append(vec_tensor(U_r, V_r, W_r))
    
    M = np.column_stack(M_cols) # Shape (60, 54)
    
    # Target tensor T
    T_vec = np.zeros(n*m*p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_vec[i*m*p + j*p + k] = 1.0
                
    # Check if M * 1 = T
    # We know this is true by construction.
    
    # Find null space of M
    # If there is a vector x in Null(M) such that 1 + x has a zero component,
    # then we can remove that component.
    # M * (1 + x) = M*1 + M*x = T + 0 = T.
    # If (1+x)_k = 0, then the k-th term is not needed.
    
    try:
        # Use SVD to find null space
        U_svd, s, Vt = np.linalg.svd(M)
        # Rank of M
        tol = 1e-10
        num_rank = np.sum(s > tol)
        null_space_dim = M.shape[1] - num_rank
        
        if null_space_dim > 0:
            # Null space vectors are rows of Vt[num_rank:]
            null_vecs = Vt[num_rank:]
            
            # We want to find a linear combination of null_vecs
            # z = null_vecs @ c
            # such that 1 + z has a zero component.
            # This is a non-linear problem in c if we require exact zero.
            # However, if the null space is 1D, we can just check the single vector.
            # If null_space_dim > 1, we can try to find c such that 1 + null_vecs @ c has a zero.
            # This is equivalent to finding if -1 is in the range of the map c -> null_vecs @ c + 1.
            # Actually, simpler: iterate through components k.
            # For each k, check if there exists c such that (null_vecs @ c)_k = -1.
            # This is a linear system: null_vecs[:, k] @ c = -1.
            # If solvable, then we can remove term k.
            
            candidates = []
            for k in range(rank):
                # Check if we can make the k-th component of (1 + null_vecs @ c) zero
                # i.e., null_vecs @ c = -1 at index k
                # This is a system of 1 equation with null_space_dim unknowns.
                # It is always solvable if null_space_dim >= 1, UNLESS the k-th row of null_vecs is all zeros.
                # If null_vecs[:, k] is not all zero, we can find c.
                # But we need the FULL vector 1 + null_vecs @ c to be a valid solution?
                # Yes, any vector in the affine space {1 + null_vecs @ c} is a solution to M @ alpha = T.
                # So if we can find c such that (1 + null_vecs @ c)_k = 0, we are good.
                # This requires that the k-th component of the null space projection can be -1.
                # This is possible if the k-th row of the null space basis is not zero.
                
                row_k = null_vecs[:, k]
                if np.any(np.abs(row_k) > 1e-9):
                    # We can solve row_k @ c = -1
                    # Let's pick the simplest solution: set one c_i to -1/row_k_i and others to 0
                    # But we need to verify the exact rational solution later.
                    candidates.append(k)
            
            if candidates:
                # Try the first candidate
                k_rem = candidates[0]
                # Construct the reduced system
                # We want alpha_r for r != k_rem such that sum_{r != k_rem} alpha_r t_r = T
                # This is M_reduced @ alpha_reduced = T
                # M_reduced is M with column k_rem removed.
                M_red = np.delete(M, k_rem, axis=1)
                
                # Solve using exact rational arithmetic
                # Convert M_red and T to Fractions
                # This might be slow, so we limit the size or time.
                # M_red is 60 x 53.
                
                if time.time() > deadline - 10:
                    return {"U": U, "V": V, "W": W}
                    
                try:
                    # Exact solve
                    # We can use a simple Gaussian elimination with Fractions
                    # Or use numpy to get a float solution and then round/verify?
                    # Given the integer nature, if a solution exists, it might be rational.
                    # Let's try to solve M_red @ x = T using numpy first to see if it's consistent.
                    x_float, residuals, rank_svd, s = np.linalg.lstsq(M_red, T_vec, rcond=None)
                    
                    # Check residual
                    if np.linalg.norm(M_red @ x_float - T_vec) < 1e-9:
                        # It seems solvable. Now verify with exact arithmetic.
                        # We need to check if x_float is close to a rational number.
                        # Or just reconstruct the decomposition with x_float and verify?
                        # No, the task requires integer/half-integer entries.
                        # If x_float is not close to half-integers, this path fails.
                        # However, the original construction used integers.
                        # If a linear dependency exists, the coefficients in the dependency
                        # might be integers or half-integers.
                        
                        # Let's check if x_float is close to 0.5 * integers
                        x_rounded = np.round(x_float * 2) / 2.0
                        if np.allclose(M_red @ x_rounded, T_vec, atol=1e-9):
                            # We found a valid decomposition with rank 53
                            # Construct new U, V, W
                            new_rank = rank - 1
                            U_new = [[0] * new_rank for _ in range(n * m)]
                            V_new = [[0] * new_rank for _ in range(m * p)]
                            W_new = [[0] * new_rank for _ in range(p * n)]
                            
                            new_col = 0
                            for r in range(rank):
                                if r == k_rem:
                                    continue
                                coeff = x_rounded[new_col]
                                # Scale the r-th column of U, V, W by coeff
                                # Note: U, V, W entries are integers. coeff is half-integer.
                                # So new entries will be half-integers.
                                for i in range(n * m):
                                    U_new[i][new_col] = U[i][r] * coeff
                                for j in range(m * p):
                                    V_new[j][new_col] = V[j][r] * coeff
                                for k in range(p * n):
                                    W_new[k][new_col] = W[k][r] * coeff
                                new_col += 1
                            
                            # Verify the new decomposition
                            if verify(U_new, V_new, W_new):
                                return {"U": U_new, "V": V_new, "W": W_new}
                            else:
                                # Verification failed, likely due to floating point issues in rounding
                                # Return original
                                return {"U": U, "V": V, "W": W}
                        else:
                            return {"U": U, "V": V, "W": W}
                    else:
                        return {"U": U, "V": V, "W": W}
                except Exception:
                    return {"U": U, "V": V, "W": W}
            else:
                return {"U": U, "V": V, "W": W}
        else:
            return {"U": U, "V": V, "W": W}
    except Exception:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

