"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0  # Safety margin before 150s kill

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

    def _build_rank7_2x2():
        """Return list of 7 (u, v, w) dicts for 2x2 matrix multiplication (Strassen)."""
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

    # Build the full 3x4 * 4x4 decomposition.
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

    def add_rank7_block(a_row_offset, b_col_offset, j_offset, c_row_offset, c_col_offset, col_start):
        """Add a rank-7 2x2*2x2 block."""
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

    # Verify the decomposition
    def verify_decomposition(U_mat, V_mat, W_mat, rank):
        try:
            U_arr = np.array(U_mat, dtype=float)
            V_arr = np.array(V_mat, dtype=float)
            W_arr = np.array(W_mat, dtype=float)

            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test

            C_computed = np.zeros((n, p))
            for r in range(rank):
                a_r = np.sum(U_arr[:, r] * A_test.flatten())
                b_r = np.sum(V_arr[:, r] * B_test.flatten())
                C_computed += W_arr[:, r].reshape(p, n).T * a_r * b_r

            max_err = np.max(np.abs(C_computed - C_expected))
            return max_err < 1e-9
        except Exception:
            return False

    if not verify_decomposition(U, V, W, total_rank):
        return {"U": U_school, "V": V_school, "W": W_school}

    # Attempt rank reduction via linear dependency detection
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    flat_dim = dim_u * dim_v * dim_w  # 2304

    # Build flattened tensor vectors
    tensor_vecs = np.zeros((total_rank, flat_dim))
    for r in range(total_rank):
        t = np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r]).flatten()
        tensor_vecs[r] = t

    best_rank = total_rank
    best_keep = list(range(total_rank))
    
    # Try multiple random orders for greedy removal
    np.random.seed(123)
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
        
        removed = set()
        # Random order
        col_order = list(range(total_rank))
        np.random.shuffle(col_order)
        
        for c in col_order:
            if c in removed:
                continue
            if time.time() > deadline - 5:
                break
                
            remaining = [r for r in range(total_rank) if r not in removed and r != c]
            if not remaining:
                continue
            
            # Check if tensor_vecs[c] is in the span of {tensor_vecs[r] for r in remaining}
            M = tensor_vecs[remaining]
            
            try:
                sv = np.linalg.svd(M, compute_uv=False)
                rank_M = np.sum(sv > 1e-10)
                
                M_plus = np.vstack([M, tensor_vecs[c].reshape(1, -1)])
                sv_plus = np.linalg.svd(M_plus, compute_uv=False)
                rank_M_plus = np.sum(sv_plus > 1e-10)
                
                if rank_M_plus == rank_M:
                    removed.add(c)
            except Exception:
                continue
        
        current_rank = total_rank - len(removed)
        if current_rank < best_rank:
            best_rank = current_rank
            best_keep = [c for c in range(total_rank) if c not in removed]
            
    if best_rank < total_rank:
        U_new = U_arr[:, best_keep]
        V_new = V_arr[:, best_keep]
        W_new = W_arr[:, best_keep]
        
        if verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), best_rank):
            print(f"Rank reduced from {total_rank} to {best_rank}")
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
        else:
            print(f"Reduced decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
