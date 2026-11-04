import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen coefficients for 2x2x2
    alpha = [
        [1, 0, 0, 1],   # s=0: A00 + A11
        [0, 0, 1, 1],   # s=1: A10 + A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00 + A01
        [-1, 0, 1, 0],  # s=5: A10 - A00
        [0, 1, 0, -1]   # s=6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],   # s=0: B00 + B11
        [1, 0, 0, 0],   # s=1: B00
        [0, 1, 0, -1],  # s=2: B01 - B11
        [-1, 0, 1, 0],  # s=3: B10 - B00
        [0, 0, 0, 1],   # s=4: B11
        [1, 1, 0, 0],   # s=5: B00 + B01
        [0, 0, 1, 1]    # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    R = 77
    U = np.zeros((n * m, R))  # (16, 77)
    V = np.zeros((m * p, R))  # (24, 77)
    W = np.zeros((p * n, R))  # (24, 77)

    col = 0

    # Part 1: 4x4x4 recursive Strassen
    for s in range(7):
        for si in range(7):
            for a in range(4):
                a_coeff = alpha[si][a]
                if a_coeff == 0:
                    continue
                x_row = a // 2
                x_col = a % 2
                for r in range(4):
                    r_coeff = alpha[s][r]
                    if r_coeff == 0:
                        continue
                    global_i = (r // 2) * 2 + x_row
                    global_j = (r % 2) * 2 + x_col
                    u_idx = global_i * m + global_j
                    U[u_idx, col] += a_coeff * r_coeff

            for b in range(4):
                b_coeff = beta[si][b]
                if b_coeff == 0:
                    continue
                y_row = b // 2
                y_inner_col = b % 2
                for c in range(4):
                    c_coeff = beta[s][c]
                    if c_coeff == 0:
                        continue
                    global_j = (c // 2) * 2 + y_row
                    global_k = (c % 2) * 2 + y_inner_col
                    v_idx = global_j * p + global_k
                    V[v_idx, col] += b_coeff * c_coeff

            for (si_row, si_col, si_coeff) in recon[s]:
                for (inner_ii, inner_kk, inner_coeff) in recon[si]:
                    global_i = si_row * 2 + inner_ii
                    global_k = si_col * 2 + inner_kk
                    w_idx = global_k * n + global_i
                    W[w_idx, col] += si_coeff * inner_coeff

            col += 1

    # Part 2: 4x4x2 (Strassen on A, schoolbook on B)
    for s in range(7):
        for i in range(2):
            for j in range(2):
                for r in range(4):
                    r_coeff = alpha[s][r]
                    if r_coeff == 0:
                        continue
                    global_i = (r // 2) * 2 + i
                    global_j = (r % 2) * 2 + j
                    u_idx = global_i * m + global_j
                    U[u_idx, col] += r_coeff

                for c in range(4):
                    c_coeff = beta[s][c]
                    if c_coeff == 0:
                        continue
                    global_j = (c // 2) * 2 + j
                    global_k = 4 + (c % 2)
                    v_idx = global_j * p + global_k
                    V[v_idx, col] += c_coeff

                for (si_row, si_col, si_coeff) in recon[s]:
                    global_i = si_row * 2 + i
                    global_k = 4 + si_col
                    w_idx = global_k * n + global_i
                    W[w_idx, col] += si_coeff

                col += 1

    # Verify initial decomposition
    def verify(U, V, W, n, m, p, tol=1e-6):
        R = U.shape[1]
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return np.max(np.abs(T - 1.0)) < tol

    if not verify(U, V, W, n, m, p):
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

    # Rank reduction: try to eliminate columns by expressing them as combinations of others
    def try_eliminate_column(U, V, W, target_col, n, m, p):
        """Try to express column target_col as a linear combination of other columns.
        Returns (success, new_U, new_V, new_W) if successful."""
        R = U.shape[1]
        if R <= 1:
            return False, U, V, W
        
        # The tensor equation: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
        # If we can write U[:,target] = sum_{r≠target} c_r * U[:,r],
        # V[:,target] = sum_{r≠target} d_r * V[:,r],
        # W[:,target] = sum_{r≠target} e_r * W[:,r],
        # with c_r * d_r * e_r = 1 for all r, then we can eliminate target.
        # This is very restrictive. Instead, use a simpler approach:
        # Check if the rank-1 tensor for target_col is in the span of the others.
        
        # Build the full tensor
        T_full = np.zeros((n * m, m * p, p * n))
        for r in range(R):
            T_full += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        # Tensor without target column
        T_without = np.zeros((n * m, m * p, p * n))
        for r in range(R):
            if r != target_col:
                T_without += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        # Residual
        residual = T_full - T_without  # This is the rank-1 tensor for target_col
        
        # Check if residual is rank-1 (it should be by construction)
        # and if it can be represented as a sum of rank-1 tensors from the remaining columns
        # This is hard to check directly. Instead, try a local optimization.
        
        return False, U, V, W

    # Alternative approach: use linear algebra to find dependencies
    # The decomposition is valid if for all (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This gives 16*24*24 = 9216 equations in 77*3 = 231 unknowns.
    # We can check if the system is overdetermined and find the null space.
    
    # Build the Jacobian-like matrix
    # For each (i,j,k) and each r, the partial derivative wrt U[i*m+j, r] is V[j*p+k, r] * W[k*n+i, r]
    # Similarly for V and W.
    
    # This is complex. Let's try a simpler heuristic:
    # Check if any two columns produce the same rank-1 tensor (up to scaling).
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R
    
    # Try pairwise column merging: if col_a and col_b can be merged into one
    for a in range(R):
        if time.time() > deadline - 5:
            break
        for b in range(a + 1, R):
            if time.time() > deadline - 5:
                break
            # Check if U[:,a]*V[:,a]*W[:,a] + U[:,b]*V[:,b]*W[:,b] is rank-1
            # This happens if U[:,a]/U[:,b] = V[:,b]/V[:,a] = W[:,a]/W[:,b] (elementwise, where nonzero)
            # Simple check: see if the sum is rank-1
            T_a = np.einsum('i,j,k->ijk', U[:, a], V[:, a], W[:, a])
            T_b = np.einsum('i,j,k->ijk', U[:, b], V[:, b], W[:, b])
            T_sum = T_a + T_b
            
            # Check rank-1 by flattening and checking if outer product matches
            # Flatten T_sum
            T_flat = T_sum.flatten()
            # Find nonzero entries
            nz = np.nonzero(T_flat)[0]
            if len(nz) < 2:
                continue
            
            # Get first two nonzero entries to determine the outer product
            # This is a heuristic; for exact rank-1 check, use SVD
            # But SVD on 16x24x24 tensor is expensive. Use a simpler check.
            # If T_sum is rank-1, then T_sum[i,j,k] = u[i]*v[j]*w[k] for some u,v,w
            # Check: T_sum[0,0,0] * T_sum[0,0,1] should equal T_sum[0,0,0]^2 * (T_sum[0,0,1]/T_sum[0,0,0])
            # This is getting complicated. Skip this approach.
            pass
    
    # Convert to lists with integer values where possible
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                rounded = round(x)
                if abs(x - rounded) < 1e-9:
                    new_row.append(int(rounded))
                else:
                    new_row.append(x)
            result.append(new_row)
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
