import numpy as np
import time
import itertools
import random

def construct():
    n, m, p = 2, 3, 9
    t_start = time.time()
    deadline = t_start + 142

    # Build the base rank-50 decomposition using Strassen's algorithm
    # for 2x2 blocks on the inner dimension, plus row augmentation.
    rank0 = 50
    U0 = [[0] * rank0 for _ in range(n * m)]
    V0 = [[0] * rank0 for _ in range(m * p)]
    W0 = [[0] * rank0 for _ in range(p * n)]

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    # 4 blocks of 7 columns each for k=0..7
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        U0[u_a][col] += 1
        U0[u_d][col] += 1
        V0[v_e][col] += 1
        V0[v_h][col] += 1

        U0[u_c][col+1] += 1
        U0[u_d][col+1] += 1
        V0[v_e][col+1] += 1

        U0[u_a][col+2] += 1
        V0[v_f][col+2] += 1
        V0[v_h][col+2] -= 1

        U0[u_d][col+3] += 1
        V0[v_g][col+3] += 1
        V0[v_e][col+3] -= 1

        U0[u_a][col+4] += 1
        U0[u_b][col+4] += 1
        V0[v_h][col+4] += 1

        U0[u_c][col+5] += 1
        U0[u_a][col+5] -= 1
        V0[v_e][col+5] += 1
        V0[v_f][col+5] += 1

        U0[u_b][col+6] += 1
        U0[u_d][col+6] -= 1
        V0[v_g][col+6] += 1
        V0[v_h][col+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W0[w_00][col] += 1
        W0[w_00][col+3] += 1
        W0[w_00][col+4] -= 1
        W0[w_00][col+6] += 1

        W0[w_01][col+2] += 1
        W0[w_01][col+4] += 1

        W0[w_10][col+1] += 1
        W0[w_10][col+3] += 1

        W0[w_11][col] += 1
        W0[w_11][col+1] -= 1
        W0[w_11][col+2] += 1
        W0[w_11][col+5] += 1

        col += 7

    # k=8 block (4 columns)
    k8 = 8
    v_e = k8
    v_g = p + k8

    U0[u_a][col] += 1
    V0[v_e][col] += 1
    U0[u_b][col+1] += 1
    V0[v_g][col+1] += 1
    U0[u_c][col+2] += 1
    V0[v_e][col+2] += 1
    U0[u_d][col+3] += 1
    V0[v_g][col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    W0[w_08][col] += 1
    W0[w_08][col+1] += 1
    W0[w_18][col+2] += 1
    W0[w_18][col+3] += 1

    col += 4

    # Row augmentation for a=0 and a=1 (18 columns)
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U0[u_ar0][col] += 1
        V0[v_b][col] += 1
        W0[k * n + 0][col] += 1
        U0[u_ar1][col+1] += 1
        V0[v_b][col+1] += 1
        W0[k * n + 1][col+1] += 1
        col += 2

    U0_np = np.array(U0, dtype=float)
    V0_np = np.array(V0, dtype=float)
    W0_np = np.array(W0, dtype=float)

    if time.time() > deadline - 5:
        return {"U": U0, "V": V0, "W": W0}

    # Build target tensor T[i,j,k] = 1 if A[i,a] * B[a,j] contributes to C[k]
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(nm):
        ai = i // m
        am = i % m
        for j in range(mp):
            bj = j // p
            bm = j % p
            if am == bj:
                k_idx = bm * n + ai
                T[i, j, k_idx] = 1.0

    # Verify base decomposition
    T_decomp = np.einsum('ic,jc,kc->ijk', U0_np, V0_np, W0_np, optimize=True)
    residual0 = np.linalg.norm(T_decomp - T)

    if residual0 > 1e-10:
        # Base decomposition is broken, return as-is
        return {"U": U0, "V": V0, "W": W0}

    # Convert to integer if all entries are close to integers
    def to_if_integer(arr):
        if np.allclose(arr, np.round(arr), atol=1e-12):
            return np.round(arr).astype(np.int64)
        return arr

    # Compute the support matrix: for each column, which (i,j,k) it touches
    # M[c, t] where t indexes T[i,j,k] flattened
    total_T = T.size
    M = np.zeros((rank0, total_T), dtype=np.float64)
    for c in range(rank0):
        uc = U0_np[:, c]
        vc = V0_np[:, c]
        wc = W0_np[:, c]
        # Outer product
        M[c, :] = np.einsum('i,j,k->ijk', uc, vc, wc).flatten()

    # Identify rows with exact integer coefficients
    int_rows = []
    for r in range(total_T):
        col_vals = M[:, r]
        if np.allclose(col_vals, np.round(col_vals), atol=1e-10):
            int_rows.append(r)

    int_M = M[int_rows]
    int_T_flat = T.flatten()
    # The values in T are 0 or 1, so check for exact match
    int_T_vals = int_T_flat[int_rows]

    # Check which integer rows have non-trivial structure
    # A row is "free" if it can be made zero without changing the tensor
    # (i.e., the corresponding T value is 0)
    free_rows = []
    constrained_rows = []
    for idx, r in enumerate(int_rows):
        if abs(int_T_vals[idx]) < 1e-10:
            free_rows.append(idx)
        else:
            constrained_rows.append(idx)

    # For free rows, we want to zero them out by finding linear combinations of columns
    # that cancel each other at those positions.
    # We can use the structure: if column c has a 1 at free position r,
    # we can pair it with another column that has a -1 at that position.
    
    best_rank = rank0
    best_U = U0_np.copy()
    best_V = V0_np.copy()
    best_W = W0_np.copy()

    # Strategy: try to drop columns that are "redundant" 
    # (i.e., removing them doesn't change the tensor because
    #  their contribution is exactly canceled by others)
    
    # Build a binary incidence matrix for free rows
    if len(free_rows) > 0 and time.time() < deadline - 40:
        # free_M[c, r] = 1 if column c touches free position r
        free_M = int_M[free_rows, :]  # shape (n_free, rank0)
        
        # Try dropping single columns: check if column c's contribution
        # to all constrained positions can be matched by adjusting
        # remaining columns. This is complex, so use a heuristic:
        # If column c only touches free positions (no constrained positions),
        # it can be dropped (but we still need to maintain tensor validity).
        
        # Actually, let's use a different approach:
        # Try all subsets of columns of size k to drop, and check if the
        # remaining columns still produce the correct tensor.
        
        # For efficiency, try dropping 1, 2, 3 columns at a time.
        
        # Precompute: for each subset of columns to keep, does it work?
        # This is exponential, so use greedy/heuristic.
        
        # Heuristic: find columns that are "similar" and can be merged.
        # Two columns c1, c2 are mergeable if their sum is zero on all
        # positions, or if one can be replaced by a combination.
        
        # Better: use linear algebra on the constraint equations.
        # We have total_T equations (one per tensor position).
        # The system is: M @ x = T_flat, where x is the coefficient vector.
        # We want to find sparse x (with x[c] = 1 for kept columns) that satisfies this.
        
        # Since we start with a valid solution (x = ones), we can try to
        # find alternative solutions with fewer non-zero entries.
        
        # For small rank reductions (rank-47, rank-48), try dropping 2-3 columns.
        
        if time.time() < deadline - 30:
            # Try dropping pairs
            for c1 in range(rank0):
                if time.time() > deadline - 10:
                    break
                for c2 in range(c1 + 1, rank0):
                    if time.time() > deadline - 10:
                        break
                    # Try dropping c1 and c2
                    cols_keep = [c for c in range(rank0) if c != c1 and c != c2]
                    U_new = U0_np[:, cols_keep]
                    V_new = V0_np[:, cols_keep]
                    W_new = W0_np[:, cols_keep]
                    
                    T_new = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new, optimize=True)
                    res = np.linalg.norm(T_new - T)
                    
                    if res < 1e-10:
                        best_rank = len(cols_keep)
                        best_U = U_new.copy()
                        best_V = V_new.copy()
                        best_W = W_new.copy()
    
    # If we found a reduction, also try to reduce further
    if best_rank < rank0 and time.time() < deadline - 15:
        current_rank = best_rank
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        
        # Try dropping one more column
        for c in range(current_rank):
            if time.time() > deadline - 8:
                break
            cols_keep = [cc for cc in range(current_rank) if cc != c]
            U_new = U_cur[:, cols_keep]
            V_new = V_cur[:, cols_keep]
            W_new = W_cur[:, cols_keep]
            
            T_new = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new, optimize=True)
            res = np.linalg.norm(T_new - T)
            
            if res < 1e-10:
                new_rank = len(cols_keep)
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U = U_new.copy()
                    best_V = V_new.copy()
                    best_W = W_new.copy()
    
    # Convert to lists
    U_final = to_if_integer(best_U).tolist()
    V_final = to_if_integer(best_V).tolist()
    W_final = to_if_integer(best_W).tolist()

    return {"U": U_final, "V": V_final, "W": W_final}
