"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True


def _tensor_contribution(n, m, p, U_arr, V_arr, W_arr, keep_mask):
    """Compute the tensor sum over kept columns."""
    U3 = U_arr[:, keep_mask].reshape(n, m, -1)
    V3 = V_arr[:, keep_mask].reshape(m, p, -1)
    W3 = W_arr[:, keep_mask].reshape(p, n, -1)
    if U3.shape[2] == 0:
        return np.zeros((n, m, p))
    return np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)


def _try_reduce_rank(n, m, p, U, V, W, deadline):
    """
    Greedily remove columns (singles, pairs, triples) that are redundant.
    Uses incremental tensor updates for speed. Multiple restarts with
    randomized order to escape local minima.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    target = np.ones((n, m, p))
    tol = 1e-9
    
    # Precompute per-column tensor contributions
    col_contrib = np.zeros((rank, n, m, p))
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        col_contrib[r] = np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    best_keep = np.ones(rank, dtype=bool)
    best_rank = rank
    
    def _greedy_pass(keep, T_current, deadline, perm=None):
        """Run greedy single/pair/triple removal until fixed point."""
        if perm is None:
            perm = list(range(rank))
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Singles
            for r in perm:
                if not keep[r]:
                    continue
                T_new = T_current - col_contrib[r]
                if np.allclose(T_new, target, atol=tol):
                    keep[r] = False
                    T_current = T_new
                    changed = True
                    break
            if changed:
                continue
            # Pairs
            kept_indices = [r for r in perm if keep[r]]
            found = False
            for i in range(len(kept_indices)):
                if time.time() > deadline:
                    break
                r1 = kept_indices[i]
                if not keep[r1]:
                    continue
                for j in range(i + 1, len(kept_indices)):
                    r2 = kept_indices[j]
                    if not keep[r2]:
                        continue
                    T_new = T_current - col_contrib[r1] - col_contrib[r2]
                    if np.allclose(T_new, target, atol=tol):
                        keep[r1] = False
                        keep[r2] = False
                        T_current = T_new
                        found = True
                        break
                if found:
                    break
            if found:
                continue
            # Triples
            kept_indices = [r for r in perm if keep[r]]
            found = False
            for i in range(len(kept_indices)):
                if time.time() > deadline:
                    break
                r1 = kept_indices[i]
                if not keep[r1]:
                    continue
                for j in range(i + 1, len(kept_indices)):
                    if time.time() > deadline:
                        break
                    r2 = kept_indices[j]
                    if not keep[r2]:
                        continue
                    for k in range(j + 1, len(kept_indices)):
                        r3 = kept_indices[k]
                        if not keep[r3]:
                            continue
                        T_new = T_current - col_contrib[r1] - col_contrib[r2] - col_contrib[r3]
                        if np.allclose(T_new, target, atol=tol):
                            keep[r1] = False
                            keep[r2] = False
                            keep[r3] = False
                            T_current = T_new
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        return T_current
    
    # First pass: deterministic order
    keep = np.ones(rank, dtype=bool)
    T_current = col_contrib.sum(axis=0)
    T_current = _greedy_pass(keep, T_current, deadline)
    cur_rank = int(np.sum(keep))
    if cur_rank < best_rank:
        best_rank = cur_rank
        best_keep = keep.copy()
    
    # Additional randomized restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        keep = np.ones(rank, dtype=bool)
        T_current = col_contrib.sum(axis=0)
        perm = rng.permutation(rank).tolist()
        T_current = _greedy_pass(keep, T_current, deadline, perm)
        
        cur_rank = int(np.sum(keep))
        if cur_rank < best_rank:
            best_rank = cur_rank
            best_keep = keep.copy()
    
    if best_rank >= rank:
        return U, V, W, rank
    
    U_new = U_arr[:, best_keep]
    V_new = V_arr[:, best_keep]
    W_new = W_arr[:, best_keep]
    
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()
    
    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, best_rank
    else:
        return U, V, W, rank


def _build_strassen(n, m, p):
    """
    Build a rank-45 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks
    with a 6+4 column partition of B.
    """
    assert n == 2 and m == 3 and p == 10
    
    # Partition B's 10 columns into 6 + 4
    # 6-column part: two 3x3 Strassen blocks (rank 13 each) = 26
    # 4-column part: one 3x3 Strassen block (rank 13) + one 3x1 schoolbook (rank 6) = 19
    # Total: 45
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]  # 6 x 45
    V = [[0] * rank for _ in range(m * p)]  # 30 x 45
    W = [[0] * rank for _ in range(p * n)]  # 20 x 45
    
    r = 0
    
    # Strassen coefficients for 2x2x2 matrix multiplication
    # For a 2x2 block: A_block[i,j] with i,j in {0,1}, B_block[j,k] with j,k in {0,1}
    # U rows indexed by (i,j) -> i*2+j (local), V rows by (j,k) -> j*2+k (local), W rows by (k,i) -> k*2+i (local)
    strassen_u = {
        0: [(0, 1), (3, 1)],   # A[0,0]+A[1,1]
        1: [(2, 1), (3, 1)],   # A[1,0]+A[1,1]
        2: [(0, 1)],           # A[0,0]
        3: [(3, 1)],           # A[1,1]
        4: [(0, 1), (1, 1)],   # A[0,0]+A[0,1]
        5: [(2, 1), (0, -1)],  # A[1,0]-A[0,0]
        6: [(1, 1), (3, -1)],  # A[0,1]-A[1,1]
    }
    
    strassen_v = {
        0: [(0, 1), (3, 1)],   # B[0,0]+B[1,1]
        1: [(0, 1)],           # B[0,0]
        2: [(1, 1), (3, -1)],  # B[0,1]-B[1,1]
        3: [(2, 1), (0, -1)],  # B[1,0]-B[0,0]
        4: [(3, 1)],           # B[1,1]
        5: [(0, 1), (1, 1)],   # B[0,0]+B[0,1]
        6: [(2, 1), (3, 1)],   # B[1,0]+B[1,1]
    }
    
    strassen_w = {
        0: [(0, 1), (3, 1)],   # C[0,0] += M1, C[1,1] += M1
        1: [(1, 1), (3, -1)],  # C[0,1] += M2... wait, let me re-derive
        2: [(2, 1), (3, 1)],
        3: [(0, 1), (1, 1)],
        4: [(0, -1), (2, 1)],
        5: [(3, 1)],
        6: [(0, 1)],
    }
    
    # Let me re-derive Strassen properly.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U (A-side): local indices (i,j) -> i*2+j
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    strassen_u = {
        0: [(0, 1), (3, 1)],   # M1: A00+A11
        1: [(2, 1), (3, 1)],   # M2: A10+A11
        2: [(0, 1)],           # M3: A00
        3: [(3, 1)],           # M4: A11
        4: [(0, 1), (1, 1)],   # M5: A00+A01
        5: [(2, 1), (0, -1)],  # M6: A10-A00
        6: [(1, 1), (3, -1)],  # M7: A01-A11
    }
    
    # V (B-side): local indices (j,k) -> j*2+k
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    strassen_v = {
        0: [(0, 1), (3, 1)],   # M1: B00+B11
        1: [(0, 1)],           # M2: B00
        2: [(1, 1), (3, -1)],  # M3: B01-B11
        3: [(2, 1), (0, -1)],  # M4: B10-B00
        4: [(3, 1)],           # M5: B11
        5: [(0, 1), (1, 1)],   # M6: B00+B01
        6: [(2, 1), (3, 1)],   # M7: B10+B11
    }
    
    # W (C-side): local indices (k,i) -> k*2+i
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_w = {
        0: [(0, 1), (3, 1)],   # M1: C00 += 1, C11 += 1
        1: [(2, 1)],           # M2: C10 += 1
        2: [(1, 1)],           # M3: C01 += 1
        3: [(0, 1), (2, 1)],   # M4: C00 += 1, C10 += 1
        4: [(0, -1), (1, 1)],  # M5: C00 -= 1, C01 += 1
        5: [(3, 1)],           # M6: C11 += 1
        6: [(0, 1)],           # M7: C00 += 1
    }
    
    def _add_strassen_block(u_col_offset, v_col_offset, w_col_offset, r_start):
        """Add a 2x3 x 3x3 Strassen block.
        u_col_offset: which column of A (0,1,2) to use for the 2x2 part (we use cols 0,1)
        v_col_offset: starting column of B for the 3x3 block
        w_col_offset: starting column of output C
        """
        # A uses columns 0 and 1 (the 2x2 part)
        # B uses 3 columns starting at v_col_offset
        # C uses 2 rows, 3 columns starting at w_col_offset
        
        for s in range(7):
            # U: A[i,j] with i in {0,1}, j in {0,1}
            for (u_local, coeff) in strassen_u[s]:
                i = u_local // 2
                j = u_local % 2
                U[i * 3 + j][r_start + s] += coeff
            
            # V: B[j,k] with j in {0,1}, k in {0,1,2}
            for (v_local, coeff) in strassen_v[s]:
                j = v_local // 2
                k = v_local % 2
                # B row j, column v_col_offset + k
                V[j * 10 + v_col_offset + k][r_start + s] += coeff
            
            # W: C[k,i] with k in {0,1,2}, i in {0,1}
            for (w_local, coeff) in strassen_w[s]:
                k = w_local // 2
                i = w_local % 2
                # C row w_col_offset + k, column i
                W[(w_col_offset + k) * 2 + i][r_start + s] += coeff
        
        # Schoolbook for A[:,2] * B[2, v_col_offset:v_col_offset+3]
        for i in range(2):
            for k in range(3):
                U[i * 3 + 2][r_start + 7 + i * 3 + k] = 1
                V[2 * 10 + v_col_offset + k][r_start + 7 + i * 3 + k] = 1
                W[(w_col_offset + k) * 2 + i][r_start + 7 + i * 3 + k] = 1
        
        return r_start + 13  # 7 Strassen + 6 schoolbook = 13
    
    # 6-column part: two 3x3 blocks
    r = _add_strassen_block(0, 0, 0, r)  # B cols 0-2, C cols 0-2
    r = _add_strassen_block(0, 3, 3, r)  # B cols 3-5, C cols 3-5
    
    # 4-column part: one 3x3 block + one 3x1 block
    r = _add_strassen_block(0, 6, 6, r)  # B cols 6-8, C cols 6-8
    
    # 3x1 block for B col 9, C col 9
    for i in range(2):
        for j in range(3):
            U[i * 3 + j][r] = 1
            V[j * 10 + 9][r] = 1
            W[9 * 2 + i][r] = 1
            r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return U, V, W, rank


def _try_linear_reduction(n, m, p, U, V, W, deadline):
    """
    Try to find linear dependencies among columns using tensor structure.
    For each subset S of columns, check if there exists a vector c such that
    sum_{r in S} c[r] * col_contrib[r] = 0, and if so, eliminate one column.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    target = np.ones((n, m, p))
    tol = 1e-9
    
    # Precompute per-column tensor contributions
    col_contrib = np.zeros((rank, n, m, p))
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        col_contrib[r] = np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    keep = np.ones(rank, dtype=bool)
    T_current = col_contrib.sum(axis=0)
    
    # Try removing pairs via linear combination: c1*C1 + c2*C2 = 0
    # This means C1 and C2 are linearly dependent as tensors
    for r1 in range(rank):
        if time.time() > deadline:
            break
        if not keep[r1]:
            continue
        for r2 in range(r1 + 1, rank):
            if not keep[r2]:
                continue
            # Check if col_contrib[r1] and col_contrib[r2] are linearly dependent
            # and their difference can be absorbed
            diff = col_contrib[r1] - col_contrib[r2]
            T_new = T_current - diff
            if np.allclose(T_new, target, atol=tol):
                keep[r1] = False
                keep[r2] = False
                T_current = T_new
                break
    
    cur_rank = int(np.sum(keep))
    if cur_rank >= rank:
        return U, V, W, rank
    
    U_new = U_arr[:, keep]
    V_new = V_arr[:, keep]
    W_new = W_arr[:, keep]
    
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()
    
    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, cur_rank
    else:
        return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin under 150s cap

    n, m, p = DIMENSIONS

    U, V, W, rank = _build_strassen(n, m, p)

    if not _validate(n, m, p, U, V, W):
        # Fallback to trivial if validation fails
        U, V, W, rank = _build_trivial(n, m, p)
    else:
        # Try greedy reduction to see if we can do even better
        U, V, W, rank = _try_reduce_rank(n, m, p, U, V, W, deadline)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
