import time
import numpy as np
import random

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

        U[(i_base + 1) * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] -= 1
        col += 1

        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1

        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1

        U[(i_base + 1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

        U[i_base * m + (j_base + 1)][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] -= 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1

        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1

        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    assert col == 54, f"Expected rank 54 but got {col}"

    # Attempt to find a rank 53 decomposition
    # Convert to numpy arrays for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_flat = T.flatten()
    
    # Build the matrix A where A[:, r] is the flattened tensor of the r-th term
    A = np.zeros((3600, 54))
    for r in range(54):
        u_r = U_np[:, r]
        v_r = V_np[:, r]
        w_r = W_np[:, r]
        tensor_r = np.einsum('i,j,k->ijk', u_r, v_r, w_r)
        A[:, r] = tensor_r.flatten()
        
    # Try to drop each term and see if the rest can span T
    for d in range(54):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        mask = [i for i in range(54) if i != d]
        A_sub = A[:, mask]
        
        # Solve least squares
        try:
            c, residuals, rank, s = np.linalg.lstsq(A_sub, T_flat, rcond=None)
        except:
            continue
            
        # Check if the solution is exact
        if residuals and residuals[0] < 1e-5:
            # Check if coefficients are close to integers or half-integers
            valid = True
            for val in c:
                # Check if val is close to integer or half-integer
                if abs(val - round(val)) > 1e-5 and abs(val - (round(val*2)/2.0)) > 1e-5:
                    valid = False
                    break
            
            if valid:
                # Construct the new solution
                new_rank = 53
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                new_col = 0
                for r in mask:
                    scale = c[new_col]
                    # Round to nearest integer or half-integer for safety
                    if abs(scale - round(scale)) < 1e-5:
                        scale = float(round(scale))
                    else:
                        scale = round(scale * 2) / 2.0
                        
                    for i in range(n*m):
                        if U_np[i, r] != 0:
                            new_U[i][new_col] = U_np[i, r] * scale
                    for j in range(m*p):
                        if V_np[j, r] != 0:
                            new_V[j][new_col] = V_np[j, r] * scale
                    for k in range(p*n):
                        if W_np[k, r] != 0:
                            new_W[k][new_col] = W_np[k, r] * scale
                    new_col += 1
                    
                # Verify the new decomposition
                U_ver = np.array(new_U, dtype=float)
                V_ver = np.array(new_V, dtype=float)
                W_ver = np.array(new_W, dtype=float)
                
                T_ver = np.zeros((n*m, m*p, p*n))
                for r in range(new_rank):
                    tensor_r = np.einsum('i,j,k->ijk', U_ver[:, r], V_ver[:, r], W_ver[:, r])
                    T_ver += tensor_r
                    
                if np.allclose(T_ver.flatten(), T_flat, atol=1e-5):
                    return {"U": new_U, "V": new_V, "W": new_W}

    return {"U": U, "V": V, "W": W}
