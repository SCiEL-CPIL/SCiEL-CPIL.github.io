"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 115.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Build the full 3x4 * 4x4 decomposition using Strassen blocks.
    # Split strategy:
    #   Rows of A: top 2 (i=0,1), bottom 1 (i=2)
    #   Cols of B / rows of C: left 2 (k=0,1), right 2 (k=2,3)
    #   Middle dim j: j=0,1 and j=2,3
    #
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]  -> 7+7=14
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]  -> 7+7=14
    # C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]  -> 4+4=8
    # C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]  -> 4+4=8
    # Total: 44

    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 44

    def _build_rank7_2x2():
        """Return list of 7 (u_dict, v_dict, w_dict) for 2x2 matrix multiplication (Strassen)."""
        terms = []
        # p1 = (a11+a22)*(b11+b22)
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        # p2 = (a21+a22)*b11
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
        # p3 = a11*(b12-b22)
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
        # p4 = a22*(b21-b11)
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
        # p5 = (a11+a12)*b22
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
        # p6 = (a21-a11)*(b11+b12)
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        # p7 = (a12-a22)*(b21+b22)
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    rank7_terms = _build_rank7_2x2()

    def add_rank7_block(a_row_offset, b_col_offset, j_offset, c_row_offset, c_col_offset, col_start):
        """Add a rank-7 2x2*2x2 block.
        a_row_offset: row offset in A (0 or 1)
        b_col_offset: col offset in B (0 or 2)
        j_offset: row offset in B / col offset in A (0 or 2)
        c_row_offset: row offset in C (0 or 1)
        c_col_offset: col offset in C (0 or 2)
        """
        for r in range(7):
            u_dict, v_dict, w_dict = rank7_terms[r]
            col = col_start + r
            for (ai, aj), val in u_dict.items():
                U[(a_row_offset + ai) * m + (j_offset + aj)][col] += val
            for (bj, bk), val in v_dict.items():
                V[(j_offset + bj) * p + (b_col_offset + bk)][col] += val
            for (ci, ck), val in w_dict.items():
                W[(c_col_offset + ck) * n + (c_row_offset + ci)][col] += val
        return col_start + 7

    def add_rank4_block_1x2x2x2(a_row, b_col_offset, j_offset, c_col_offset, col_start):
        """Add a rank-4 1x2 * 2x2 -> 1x2 block (bottom row of A, 2 cols of B)."""
        for r in range(4):
            col = col_start + r
            j_idx = j_offset + (r % 2)
            k_idx = b_col_offset + (r // 2)
            U[a_row * m + j_idx][col] = 1
            V[j_idx * p + k_idx][col] = 1
            W[k_idx * n + a_row][col] = 1
        return col_start + 4

    col = 0
    # Top 2 rows of A, all 4x4 of B: 4 rank-7 blocks
    col = add_rank7_block(0, 0, 0, 0, 0, col)
    col = add_rank7_block(0, 0, 2, 0, 0, col)
    col = add_rank7_block(0, 2, 0, 0, 2, col)
    col = add_rank7_block(0, 2, 2, 0, 2, col)
    # Bottom row of A, all 4x4 of B: 4 rank-4 blocks
    col = add_rank4_block_1x2x2x2(2, 0, 0, 0, col)
    col = add_rank4_block_1x2x2x2(2, 0, 2, 0, col)
    col = add_rank4_block_1x2x2x2(2, 2, 0, 2, col)
    col = add_rank4_block_1x2x2x2(2, 2, 2, 2, col)

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verify the decomposition using vectorized numpy
    try:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        # Vectorized: a_r = sum_{i,j} U[i*m+j, r] * A[i,j]
        A_flat = A_test.flatten()  # length n*m
        B_flat = B_test.flatten()  # length m*p
        a_r = U_arr.T @ A_flat  # length total_rank
        b_r = V_arr.T @ B_flat  # length total_rank

        # C[i,k] = sum_r W[k*n+i, r] * a_r * b_r
        W_reshaped = W_arr.reshape(p, n, total_rank)  # [k, i, r]
        C_computed = np.einsum('kir,r,r->ik', W_reshaped, a_r, b_r)

        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-9:
            return {"U": U_school, "V": V_school, "W": W_school}
    except Exception:
        return {"U": U_school, "V": V_school, "W": W_school}

    # Now attempt rank reduction via linear dependency detection
    # Strategy: for each column, check if its tensor (u_col, v_col, w_col)
    # is a scalar multiple of another column's tensor. If so, it's redundant.
    # More generally, check if the set of columns has linear dependencies
    # in the tensor product space.
    
    try:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # First pass: check for exact scalar multiples (simplest case)
        removed = set()
        for c in range(total_rank - 1, -1, -1):
            if c in removed:
                continue
            u_c = U_arr[:, c]
            v_c = V_arr[:, c]
            w_c = W_arr[:, c]
            
            # Skip if this column is all zeros
            if np.allclose(u_c, 0) and np.allclose(v_c, 0) and np.allclose(w_c, 0):
                removed.add(c)
                continue
            
            dependent = False
            for c2 in range(c + 1, total_rank):
                if c2 in removed:
                    continue
                u_c2 = U_arr[:, c2]
                v_c2 = V_arr[:, c2]
                w_c2 = W_arr[:, c2]
                
                # Check if (u_c, v_c, w_c) = alpha * (u_c2, v_c2, w_c2)
                # Find alpha from u vectors
                mask = np.abs(u_c2) > 1e-15
                if np.any(mask):
                    alphas_u = u_c[mask] / u_c2[mask]
                    if np.std(alphas_u) < 1e-12:
                        alpha = alphas_u[0]
                        if np.allclose(v_c, alpha * v_c2, atol=1e-12) and \
                           np.allclose(w_c, alpha * w_c2, atol=1e-12):
                            dependent = True
                            break
                else:
                    # u_c2 is zero, check v
                    mask = np.abs(v_c2) > 1e-15
                    if np.any(mask):
                        alphas_v = v_c[mask] / v_c2[mask]
                        if np.std(alphas_v) < 1e-12:
                            alpha = alphas_v[0]
                            if np.allclose(u_c, alpha * u_c2, atol=1e-12) and \
                               np.allclose(w_c, alpha * w_c2, atol=1e-12):
                                dependent = True
                                break
                    else:
                        mask = np.abs(w_c2) > 1e-15
                        if np.any(mask):
                            alphas_w = w_c[mask] / w_c2[mask]
                            if np.std(alphas_w) < 1e-12:
                                alpha = alphas_w[0]
                                if np.allclose(u_c, alpha * u_c2, atol=1e-12) and \
                                   np.allclose(v_c, alpha * v_c2, atol=1e-12):
                                    dependent = True
                                    break
            
            if dependent:
                removed.add(c)
        
        if removed:
            keep = [c for c in range(total_rank) if c not in removed]
            new_rank = len(keep)
            U_new = [U[r][c] for r in range(n*m) for c in [0]] # placeholder
            # Actually need to properly slice
            U_new = [[U[r][c] for c in keep] for r in range(n*m)]
            V_new = [[V[r][c] for c in keep] for r in range(m*p)]
            W_new = [[W[r][c] for c in keep] for r in range(p*n)]
            
            # Verify the reduced decomposition
            U_arr_new = np.array(U_new, dtype=float)
            V_arr_new = np.array(V_new, dtype=float)
            W_arr_new = np.array(W_new, dtype=float)
            
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            a_r = U_arr_new.T @ A_flat
            b_r = V_arr_new.T @ B_flat
            
            W_reshaped = W_arr_new.reshape(p, n, new_rank)
            C_computed = np.einsum('kir,r,r->ik', W_reshaped, a_r, b_r)
            
            max_err = np.max(np.abs(C_computed - C_expected))
            if max_err < 1e-9:
                return {"U": U_new, "V": V_new, "W": W_new}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
