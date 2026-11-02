import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 6

    # Strategy: Use recursive Strassen to compute 4x4 * 4x6.
    # Split A (4x4) into four 2x2 blocks, B (4x6) into two 2x6 halves.
    # C[0:2,:] = A[0:2,0:2]*B[0:2,:] + A[0:2,2:4]*B[2:4,:]
    # C[2:4,:] = A[2:4,0:2]*B[0:2,:] + A[2:4,2:4]*B[2:4,:]
    # Each 2x2*2x6 uses Strassen on three 2x2 column-blocks: 3*7 = 21 terms.
    # Total: 4 * 21 = 84 terms.

    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84

    col = 0

    def a_idx(r, c):
        return r * m + c

    def b_idx(r, c):
        return r * p + c

    def c_idx(r, c):
        return c * n + r  # W[k*n+i] where i=row, k=col

    def add_term(u_entries, v_entries, w_entries, c):
        for idx, val in u_entries:
            U[idx][c] += val
        for idx, val in v_entries:
            V[idx][c] += val
        for idx, val in w_entries:
            W[idx][c] += val

    # Strassen for 2x2 * 2x2 at A[r0:r0+2, c0:c0+2] * B[rb0:rb0+2, cb0:cb0+2] -> C[rc0:rc0+2, cc0:cc0+2]
    def add_strassen_2x2x2(r0, c0, rb0, cb0, rc0, cc0, c):
        A11 = a_idx(r0, c0)
        A12 = a_idx(r0, c0 + 1)
        A21 = a_idx(r0 + 1, c0)
        A22 = a_idx(r0 + 1, c0 + 1)

        B11 = b_idx(rb0, cb0)
        B12 = b_idx(rb0, cb0 + 1)
        B21 = b_idx(rb0 + 1, cb0)
        B22 = b_idx(rb0 + 1, cb0 + 1)

        C11 = c_idx(rc0, cc0)
        C12 = c_idx(rc0, cc0 + 1)
        C21 = c_idx(rc0 + 1, cc0)
        C22 = c_idx(rc0 + 1, cc0 + 1)

        # M1 = (A11+A22)(B11+B22)
        add_term([(A11, 1), (A22, 1)], [(B11, 1), (B22, 1)],
                 [(C11, 1), (C22, 1)], c)
        c += 1

        # M2 = (A21+A22)B11
        add_term([(A21, 1), (A22, 1)], [(B11, 1)],
                 [(C21, 1)], c)
        c += 1

        # M3 = A11(B12-B22)
        add_term([(A11, 1)], [(B12, 1), (B22, -1)],
                 [(C12, 1)], c)
        c += 1

        # M4 = A22(B21-B11)
        add_term([(A22, 1)], [(B21, 1), (B11, -1)],
                 [(C11, -1), (C21, 1)], c)
        c += 1

        # M5 = (A11+A12)B22
        add_term([(A11, 1), (A12, 1)], [(B22, 1)],
                 [(C11, -1), (C12, 1)], c)
        c += 1

        # M6 = (A21-A11)(B11+B12)
        add_term([(A21, 1), (A11, -1)], [(B11, 1), (B12, 1)],
                 [(C22, 1)], c)
        c += 1

        # M7 = (A12-A22)(B21+B22)
        add_term([(A12, 1), (A22, -1)], [(B21, 1), (B22, 1)],
                 [(C11, 1), (C22, -1)], c)
        c += 1

        return c

    # For 2x2 * 2x6: split B into three 2x2 column-blocks
    def add_strassen_2x2x6(r0, c0, rb0, cb0, rc0, cc0, c):
        c = add_strassen_2x2x2(r0, c0, rb0, cb0, rc0, cc0, c)
        c = add_strassen_2x2x2(r0, c0, rb0, cb0 + 2, rc0, cc0 + 2, c)
        c = add_strassen_2x2x2(r0, c0, rb0, cb0 + 4, rc0, cc0 + 4, c)
        return c

    # Build the four 2x2*2x6 products
    col = add_strassen_2x2x6(0, 0, 0, 0, 0, 0, col)
    col = add_strassen_2x2x6(0, 2, 2, 0, 0, 0, col)
    col = add_strassen_2x2x6(2, 0, 0, 0, 2, 0, col)
    col = add_strassen_2x2x6(2, 2, 2, 0, 2, 0, col)

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Verify with random test
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 6)
    C_expected = A_test @ B_test

    C_computed = np.zeros((4, 6))
    for r in range(rank):
        alpha_r = np.dot(U_np[:, r], A_test.flatten())
        beta_r = np.dot(V_np[:, r], B_test.flatten())
        C_computed += alpha_r * beta_r * W_np[:, r].reshape(6, 4).T

    error = np.max(np.abs(C_computed - C_expected))

    if error > 1e-10:
        # Fall back to naive if verification fails
        rank_naive = n * m * p
        U = [[0] * rank_naive for _ in range(n * m)]
        V = [[0] * rank_naive for _ in range(m * p)]
        W = [[0] * rank_naive for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank using fastest descent
    # For each term t, check if it can be expressed as linear combo of others
    # This means: U[:,t] is in col(U[:,others]) AND V[:,t] is in col(V[:,others]) AND W[:,t] is in col(W[:,others])
    # More precisely, we need: there exist coefficients c_1..c_{r-1} such that
    # U[:,t] = sum c_i U[:,i], V[:,t] = sum c_i V[:,i], W[:,t] = sum c_i W[:,i]
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    current_rank = rank
    U_cur = U_np.copy()
    V_cur = V_np.copy()
    W_cur = W_np.copy()
    
    # Try to eliminate terms one by one
    for _pass in range(10):  # Multiple passes
        if time.time() > deadline - 30:
            break
        eliminated = False
        for t in range(current_rank - 1, -1, -1):
            if time.time() > deadline - 30:
                break
            # Get all other terms
            others = [i for i in range(current_rank) if i != t]
            if len(others) < 3:
                continue
            
            U_others = U_cur[:, others]  # 16 x (r-1)
            V_others = V_cur[:, others]  # 24 x (r-1)
            W_others = W_cur[:, others]  # 24 x (r-1)
            
            U_t = U_cur[:, t]
            V_t = V_cur[:, t]
            W_t = W_cur[:, t]
            
            # Check if U_t is in column space of U_others
            # Use least squares to find coefficients
            try:
                # Solve for coefficients that best fit U_t
                coeffs_U, _, _, _ = np.linalg.lstsq(U_others, U_t, rcond=None)
                residual_U = np.linalg.norm(U_t - U_others @ coeffs_U)
                
                if residual_U > 1e-10:
                    continue
                
                # Check if same coefficients work for V_t
                V_pred = V_others @ coeffs_U
                residual_V = np.linalg.norm(V_t - V_pred)
                
                if residual_V > 1e-10:
                    continue
                
                # Check if same coefficients work for W_t
                W_pred = W_others @ coeffs_U
                residual_W = np.linalg.norm(W_t - W_pred)
                
                if residual_W > 1e-10:
                    continue
                
                # All three match! We can eliminate term t
                # Remove column t
                mask = [i for i in range(current_rank) if i != t]
                U_cur = U_cur[:, mask]
                V_cur = V_cur[:, mask]
                W_cur = W_cur[:, mask]
                current_rank -= 1
                eliminated = True
                print(f"Eliminated term {t}, new rank: {current_rank}")
                
            except np.linalg.LinAlgError:
                continue
        
        if not eliminated:
            break
    
    if current_rank < rank:
        # Convert back to list format
        U_final = U_cur.tolist()
        V_final = V_cur.tolist()
        W_final = W_cur.tolist()
        
        # Verify
        np.random.seed(42)
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 6)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((4, 6))
        for r in range(current_rank):
            alpha_r = np.dot(U_cur[:, r], A_test.flatten())
            beta_r = np.dot(V_cur[:, r], B_test.flatten())
            C_computed += alpha_r * beta_r * W_cur[:, r].reshape(6, 4).T
        
        error = np.max(np.abs(C_computed - C_expected))
        
        if error > 1e-10:
            # Fall back to original
            return {"U": U, "V": V, "W": W}
        
        return {"U": U_final, "V": V_final, "W": W_final}
    else:
        return {"U": U, "V": V, "W": W}
