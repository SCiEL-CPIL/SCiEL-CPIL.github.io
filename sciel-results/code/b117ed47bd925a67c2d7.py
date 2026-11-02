import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Block decomposition with block size 2
    block_size = 2
    n_blocks = n // block_size  # 3

    # Total rank is n_blocks^3 * 7
    rank = n_blocks ** 3 * 7

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    # Helper to get index in U for block (ib, kb) and local (r, c)
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    # Helper to get index in V for block (kb, jb) and local (r, c)
    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    # Helper to get index in W for block (ib, jb) and local (r, c)
    # W index is k*n + i, where k is col of C, i is row of C
    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Strassen definitions for 2x2 block matrices:
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6

    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                # M1
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M2
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] -= 1
                col += 1

                # M3
                U[u_idx(ib, kb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M4
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                col += 1

                # M5
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                col += 1

                # M6
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 0, 0), col] -= 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M7
                U[u_idx(ib, kb, 0, 1), col] += 1
                U[u_idx(ib, kb, 1, 1), col] -= 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                col += 1

    # Post-processing: greedily remove columns that are redundant
    # A column t is redundant if the tensor sum_{s != t} U_s ⊗ V_s ⊗ W_s 
    # still equals the multiplication tensor (all ones in valid positions)
    
    def is_valid_decomposition(U_arr, V_arr, W_arr, n, m, p):
        """Check if the decomposition exactly reproduces the multiplication tensor."""
        # For each (i, j, k) where 0<=i<n, 0<=j<p, 0<=k<m:
        # sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t] should equal 1
        # We check all combinations
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    total = 0
                    for t in range(U_arr.shape[1]):
                        total += U_arr[i * m + k, t] * V_arr[k * p + j, t] * W_arr[j * n + i, t]
                    if total != 1:
                        return False
        return True

    current_rank = U.shape[1]
    
    # Try to remove each column one at a time
    # Start from the end to preserve ordering
    keep_cols = list(range(current_rank))
    
    for t in range(current_rank - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        if t not in keep_cols:
            continue
        # Try removing column t
        new_keep = [c for c in keep_cols if c != t]
        if len(new_keep) == 0:
            break
        
        U_test = U[:, new_keep]
        V_test = V[:, new_keep]
        W_test = W[:, new_keep]
        
        if is_valid_decomposition(U_test, V_test, W_test, n, m, p):
            keep_cols = new_keep
    
    if len(keep_cols) < current_rank:
        U = U[:, keep_cols]
        V = V[:, keep_cols]
        W = W[:, keep_cols]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
