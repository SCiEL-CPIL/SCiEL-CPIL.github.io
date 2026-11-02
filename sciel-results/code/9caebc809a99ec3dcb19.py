"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def build_strassen_36():
        rank = 36
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices

            # M1: (y00+y11)(z00+z11) -> X00, X11
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1

            # M2: (y10+y11)z00 -> X10, -X11
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1

            # M3: y00(z01-z11) -> X01, X11
            U[u_idx(y00[0], y00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = -1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1

            # M4: y11(z10-z00) -> X00, X10
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = -1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1

            # M5: (y00+y01)z11 -> -X00, X01
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y01[0], y01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1

            # M6: (y10-y00)(z00+z01) -> X11
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y00[0], y00[1])][col] = -1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1

            # M7: (y01-y11)(z10+z11) -> X00
            U[u_idx(y01[0], y01[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = -1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1

        # Block 1: A_L (cols 0,1) * B_11 (rows 0,1; cols 0,1)
        y1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
        z1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
        w1 = [(0, 0), (1, 0), (0, 1), (1, 1)]
        strassen_block(y1, z1, w1)

        # Block 2: A_R (cols 2,3) * B_21 (rows 2,3; cols 0,1)
        y2 = [(0, 2), (0, 3), (1, 2), (1, 3)]
        z2 = [(2, 0), (2, 1), (3, 0), (3, 1)]
        w2 = [(0, 0), (1, 0), (0, 1), (1, 1)]
        strassen_block(y2, z2, w2)

        # Block 3: A_L (cols 0,1) * B_12 (rows 0,1; cols 2,3)
        y3 = [(0, 0), (0, 1), (1, 0), (1, 1)]
        z3 = [(0, 2), (0, 3), (1, 2), (1, 3)]
        w3 = [(2, 0), (3, 0), (2, 1), (3, 1)]
        strassen_block(y3, z3, w3)

        # Block 4: A_R (cols 2,3) * B_22 (rows 2,3; cols 2,3)
        y4 = [(0, 2), (0, 3), (1, 2), (1, 3)]
        z4 = [(2, 2), (2, 3), (3, 2), (3, 3)]
        w4 = [(2, 0), (3, 0), (2, 1), (3, 1)]
        strassen_block(y4, z4, w4)

        # Block 5: A_L (cols 0,1) * B_15 (rows 0,1; col 4)
        U[u_idx(0, 0)][col] = 1
        V[v_idx(0, 4)][col] = 1
        W[w_idx(4, 0)][col] = 1
        col += 1

        U[u_idx(0, 1)][col] = 1
        V[v_idx(1, 4)][col] = 1
        W[w_idx(4, 0)][col] = 1
        col += 1

        U[u_idx(1, 0)][col] = 1
        V[v_idx(0, 4)][col] = 1
        W[w_idx(4, 1)][col] = 1
        col += 1

        U[u_idx(1, 1)][col] = 1
        V[v_idx(1, 4)][col] = 1
        W[w_idx(4, 1)][col] = 1
        col += 1

        # Block 6: A_R (cols 2,3) * B_25 (rows 2,3; col 4)
        U[u_idx(0, 2)][col] = 1
        V[v_idx(2, 4)][col] = 1
        W[w_idx(4, 0)][col] = 1
        col += 1

        U[u_idx(0, 3)][col] = 1
        V[v_idx(3, 4)][col] = 1
        W[w_idx(4, 0)][col] = 1
        col += 1

        U[u_idx(1, 2)][col] = 1
        V[v_idx(2, 4)][col] = 1
        W[w_idx(4, 1)][col] = 1
        col += 1

        U[u_idx(1, 3)][col] = 1
        V[v_idx(3, 4)][col] = 1
        W[w_idx(4, 1)][col] = 1
        col += 1

        return U, V, W

    U, V, W = build_strassen_36()

    def verify_decomposition(U_arr, V_arr, W_arr):
        """Check if the decomposition is valid by testing on random inputs."""
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(W_arr.shape[1]):
                C_computed += (U_arr @ V_arr[:, r]) * W_arr[:, r].reshape(p, n).T
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    def try_remove_columns(U_arr, V_arr, W_arr, max_removals=5):
        """Try to remove linearly dependent columns using structured block analysis."""
        current_U = U_arr.copy()
        current_V = V_arr.copy()
        current_W = W_arr.copy()
        current_rank = current_U.shape[1]

        removed = 0
        for attempt in range(max_removals):
            if time.time() > deadline:
                break
            if current_rank <= 1:
                break

            found_removal = False
            for col_idx in range(current_rank):
                if time.time() > deadline:
                    break

                other_cols = [c for c in range(current_rank) if c != col_idx]
                if len(other_cols) == 0:
                    continue

                # Structured check: for each block of 7 Strassen columns,
                # check if the last column (index 6 within block, i.e., col_idx if
                # it's the 7th of a block) can be expressed as a linear combination
                # of the first 6 columns of the same block.
                
                # Identify which block this column belongs to
                # Blocks are at columns: 0-6, 7-13, 14-20, 21-27
                block_start = None
                for bs in [0, 7, 14, 21]:
                    if bs <= col_idx < bs + 7:
                        block_start = bs
                        break
                
                if block_start is None:
                    # Not in a Strassen block, try generic check
                    # Use a smaller subsampled tensor check
                    n_i = n * m
                    n_j = m * p
                    n_k = p * n
                    
                    # Use a moderate subsample
                    rng = np.random.RandomState(42 + attempt + col_idx)
                    n_sub_i = min(15, n_i)
                    n_sub_j = min(15, n_j)
                    n_sub_k = min(15, n_k)
                    
                    i_sub = rng.choice(n_i, n_sub_i, replace=False)
                    j_sub = rng.choice(n_j, n_sub_j, replace=False)
                    k_sub = rng.choice(n_k, n_sub_k, replace=False)
                    
                    n_eq = n_sub_i * n_sub_j * n_sub_k
                    target = np.zeros(n_eq)
                    basis = np.zeros((n_eq, len(other_cols)))
                    
                    idx = 0
                    for ii in i_sub:
                        for jj in j_sub:
                            for kk in k_sub:
                                target[idx] = current_U[ii, col_idx] * current_V[jj, col_idx] * current_W[kk, col_idx]
                                for oc_idx, oc in enumerate(other_cols):
                                    basis[idx, oc_idx] = current_U[ii, oc] * current_V[jj, oc] * current_W[kk, oc]
                                idx += 1
                    
                    try:
                        coeffs, residuals, rank_basis, sv = np.linalg.lstsq(basis, target, rcond=None)
                        reconstruction = basis @ coeffs
                        if np.allclose(reconstruction, target, atol=1e-10):
                            new_U = np.delete(current_U, col_idx, axis=1)
                            new_V = np.delete(current_V, col_idx, axis=1)
                            new_W = np.delete(current_W, col_idx, axis=1)
                            if verify_decomposition(new_U, new_V, new_W):
                                current_U = new_U
                                current_V = new_V
                                current_W = new_W
                                current_rank -= 1
                                removed += 1
                                found_removal = True
                                break
                    except:
                        continue
                else:
                    # Structured block check: can column col_idx be expressed as
                    # linear combination of other columns in the same block?
                    block_cols = list(range(block_start, block_start + 7))
                    local_idx = col_idx - block_start
                    other_in_block = [c for c in block_cols if c != col_idx]
                    
                    # Build the linear system for this block only
                    # The tensor contribution of each column r is:
                    # T_r[i,j,k] = U[i,r] * V[j,r] * W[k,r]
                    # We want: T_col_idx = sum_{oc in other_in_block} alpha_oc * T_oc
                    
                    # Use a subsample of (i,j,k) indices
                    rng = np.random.RandomState(100 + attempt + col_idx)
                    n_i = n * m
                    n_j = m * p
                    n_k = p * n
                    
                    n_sub_i = min(12, n_i)
                    n_sub_j = min(12, n_j)
                    n_sub_k = min(12, n_k)
                    
                    i_sub = rng.choice(n_i, n_sub_i, replace=False)
                    j_sub = rng.choice(n_j, n_sub_j, replace=False)
                    k_sub = rng.choice(n_k, n_sub_k, replace=False)
                    
                    n_eq = n_sub_i * n_sub_j * n_sub_k
                    target = np.zeros(n_eq)
                    basis = np.zeros((n_eq, len(other_in_block)))
                    
                    idx = 0
                    for ii in i_sub:
                        for jj in j_sub:
                            for kk in k_sub:
                                target[idx] = current_U[ii, col_idx] * current_V[jj, col_idx] * current_W[kk, col_idx]
                                for oc_idx, oc in enumerate(other_in_block):
                                    basis[idx, oc_idx] = current_U[ii, oc] * current_V[jj, oc] * current_W[kk, oc]
                                idx += 1
                    
                    try:
                        coeffs, residuals, rank_basis, sv = np.linalg.lstsq(basis, target, rcond=None)
                        reconstruction = basis @ coeffs
                        if np.allclose(reconstruction, target, atol=1e-10):
                            new_U = np.delete(current_U, col_idx, axis=1)
                            new_V = np.delete(current_V, col_idx, axis=1)
                            new_W = np.delete(current_W, col_idx, axis=1)
                            if verify_decomposition(new_U, new_V, new_W):
                                current_U = new_U
                                current_V = new_V
                                current_W = new_W
                                current_rank -= 1
                                removed += 1
                                found_removal = True
                                break
                    except:
                        continue

            if not found_removal:
                break

        return current_U, current_V, current_W, current_rank

    # Try to remove columns from the Strassen decomposition
    best_U, best_V, best_W, best_rank = try_remove_columns(U, V, W, max_removals=5)

    # Convert back to lists
    best_U_list = [[best_U[i, j] for j in range(best_U.shape[1])] for i in range(best_U.shape[0])]
    best_V_list = [[best_V[i, j] for j in range(best_V.shape[1])] for i in range(best_V.shape[0])]
    best_W_list = [[best_W[i, j] for j in range(best_W.shape[1])] for i in range(best_W.shape[0])]

    # Additional targeted search: try to merge columns across the two 2x2 blocks
    # that share the same W output indices. Specifically, check if we can express
    # one M7-type column (which only contributes to C[0,0] or C[0,1]) as a combination
    # of columns from both blocks that also contribute to those outputs.
    if time.time() < deadline - 20:
        # Try removing each M7 column (indices 6, 13, 20, 27) using ALL other columns
        m7_indices = [6, 13, 20, 27]
        for m7 in m7_indices:
            if time.time() > deadline - 15:
                break
            if m7 >= best_rank:
                continue
            
            other_all = [c for c in range(best_rank) if c != m7]
            if len(other_all) == 0:
                continue
            
            # Build linear system with all other columns
            rng = np.random.RandomState(200 + m7)
            n_i = n * m
            n_j = m * p
            n_k = p * n
            
            n_sub_i = min(10, n_i)
            n_sub_j = min(10, n_j)
            n_sub_k = min(10, n_k)
            
            i_sub = rng.choice(n_i, n_sub_i, replace=False)
            j_sub = rng.choice(n_j, n_sub_j, replace=False)
            k_sub = rng.choice(n_k, n_sub_k, replace=False)
            
            n_eq = n_sub_i * n_sub_j * n_sub_k
            target = np.zeros(n_eq)
            basis = np.zeros((n_eq, len(other_all)))
            
            idx = 0
            for ii in i_sub:
                for jj in j_sub:
                    for kk in k_sub:
                        target[idx] = best_U[ii, m7] * best_V[jj, m7] * best_W[kk, m7]
                        for oc_idx, oc in enumerate(other_all):
                            basis[idx, oc_idx] = best_U[ii, oc] * best_V[jj, oc] * best_W[kk, oc]
                        idx += 1
            
            try:
                coeffs, residuals, rank_basis, sv = np.linalg.lstsq(basis, target, rcond=None)
                reconstruction = basis @ coeffs
                if np.allclose(reconstruction, target, atol=1e-10):
                    new_U = np.delete(best_U, m7, axis=1)
                    new_V = np.delete(best_V, m7, axis=1)
                    new_W = np.delete(best_W, m7, axis=1)
                    if verify_decomposition(new_U, new_V, new_W):
                        best_U = new_U
                        best_V = new_V
                        best_W = new_W
                        best_rank -= 1
                        # Rebuild lists
                        best_U_list = [[best_U[i, j] for j in range(best_U.shape[1])] for i in range(best_U.shape[0])]
                        best_V_list = [[best_V[i, j] for j in range(best_V.shape[1])] for i in range(best_V.shape[0])]
                        best_W_list = [[best_W[i, j] for j in range(best_W.shape[1])] for i in range(best_W.shape[0])]
            except:
                pass

    # Clean up floating point values to integers/half-integers if possible
    def clean_val(v):
        if abs(v - round(v)) < 1e-10:
            return int(round(v))
        elif abs(v * 2 - round(v * 2)) < 1e-10:
            return round(v * 2) / 2.0
        else:
            return v

    best_U_list = [[clean_val(v) for v in row] for row in best_U_list]
    best_V_list = [[clean_val(v) for v in row] for row in best_V_list]
    best_W_list = [[clean_val(v) for v in row] for row in best_W_list]

    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
