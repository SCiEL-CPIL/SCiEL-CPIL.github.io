"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_2x3x3_block_strassen():
    """Construct rank-17 decomposition for 2x3 x 3x3 matrix multiplication."""
    n_b, m_b, p_b = 2, 3, 3
    rank = 17
    U = np.zeros((n_b * m_b, rank))
    V = np.zeros((m_b * p_b, rank))
    W = np.zeros((p_b * n_b, rank))
    col = 0

    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals.items():
            U[idx, col] += val
        for idx, val in v_vals.items():
            V[idx, col] += val
        for idx, val in w_vals.items():
            W[idx, col] += val
        col += 1

    # U indices: i*m+j
    # 0:a00 1:a01 2:a02 3:a10 4:a11 5:a12
    # V indices: j*p+k
    # 0:b00 1:b01 2:b02 3:b10 4:b11 5:b12 6:b20 7:b21 8:b22
    # W indices: k*n+i
    # 0:c00 1:c10 2:c01 3:c11 4:c02 5:c12

    # Strassen for 2x2 sub-block (j,k in {0,1})
    # M1 = (a00+a11)(b00+b11) -> c00, c11
    add_term({0:1, 3:1}, {0:1, 4:1}, {0:1, 3:1})
    # M2 = (a10+a11)b00 -> c10, -c11
    add_term({3:1, 4:1}, {0:1}, {1:1, 3:-1})
    # M3 = a00(b01-b11) -> c01, c11
    add_term({0:1}, {1:1, 4:-1}, {2:1, 3:1})
    # M4 = a11(b10-b00) -> c00, c10
    add_term({4:1}, {3:1, 0:-1}, {0:1, 1:1})
    # M5 = (a00+a01)b11 -> -c00, c01
    add_term({0:1, 1:1}, {4:1}, {0:-1, 2:1})
    # M6 = (a10-a00)(b00+b01) -> c11
    add_term({3:1, 0:-1}, {0:1, 1:1}, {3:1})
    # M7 = (a01-a11)(b10+b11) -> c00
    add_term({1:1, 4:-1}, {3:1, 4:1}, {0:1})

    # Schoolbook: j in {0,1}, k=2
    add_term({0:1}, {2:1}, {4:1})   # a00*b02 -> c02
    add_term({1:1}, {5:1}, {4:1})   # a01*b12 -> c02
    add_term({3:1}, {2:1}, {5:1})   # a10*b02 -> c12
    add_term({4:1}, {5:1}, {5:1})   # a11*b12 -> c12

    # Schoolbook: j=2, k in {0,1,2}
    add_term({2:1}, {6:1}, {0:1})   # a02*b20 -> c00
    add_term({2:1}, {7:1}, {2:1})   # a02*b21 -> c01
    add_term({2:1}, {8:1}, {4:1})   # a02*b22 -> c02
    add_term({5:1}, {6:1}, {1:1})   # a12*b20 -> c10
    add_term({5:1}, {7:1}, {3:1})   # a12*b21 -> c11
    add_term({5:1}, {8:1}, {5:1})   # a12*b22 -> c12

    return U, V, W

def _verify_block(U, V, W, n_b, m_b, p_b):
    """Verify a block decomposition is exact."""
    dim_u = n_b * m_b
    dim_v = m_b * p_b
    dim_w = p_b * n_b
    err = 0.0
    # Check all (a,b,c) triples
    for a in range(dim_u):
        for b in range(dim_v):
            for c in range(dim_w):
                i = a // m_b
                j = a % m_b
                j2 = b // p_b
                k = b % p_b
                k2 = c // n_b
                i2 = c % n_b
                expected = 1.0 if (j == j2 and k == k2 and i == i2) else 0.0
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err += (val - expected) ** 2
    return err

def _try_rank_r_block(n_b, m_b, p_b, r, time_budget=5.0):
    """Try to find a rank-r decomposition via random init + linear solve.
    Returns (U, V, W) if found, else None."""
    start = time.time()
    dim_u = n_b * m_b
    dim_v = m_b * p_b
    dim_w = p_b * n_b
    num_terms = n_b * m_b * p_b

    # Build the constraint matrix: for each (a,b,c), sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
    # This is bilinear in (U,W) given V, and bilinear in (U,V) given W, etc.
    # Strategy: random init for U,V, then solve for W linearly.
    
    for attempt in range(50):
        if time.time() - start > time_budget:
            break
        U = np.random.randn(dim_u, r)
        V = np.random.randn(dim_v, r)
        
        # For each c, W[c,:] is determined by: for all (a,b) with T[a,b,c]!=0,
        # sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
        # This is a linear system in W[c,:].
        
        W = np.zeros((dim_w, r))
        feasible = True
        for c in range(dim_w):
            k = c // n_b
            i = c % n_b
            # Constraints: for all j, sum_d U[i*m_b+j, d] * V[j*p_b+k, d] * W[c, d] = 1
            # That's m_b constraints on r unknowns W[c, 0..r-1]
            A_mat = np.zeros((m_b, r))
            for j in range(m_b):
                a_idx = i * m_b + j
                b_idx = j * p_b + k
                A_mat[j, :] = U[a_idx, :] * V[b_idx, :]
            b_vec = np.ones(m_b)
            # Solve least squares
            try:
                sol, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                feasible = False
                break
            # Check if solution is exact
            if np.linalg.norm(A_mat @ sol - b_vec) > 1e-8:
                feasible = False
                break
            W[c, :] = sol
        
        if not feasible:
            continue
        
        # Verify full decomposition
        err = _verify_block(U, V, W, n_b, m_b, p_b)
        if err < 1e-8:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18

    start_time = time.time()
    deadline = start_time + 145  # leave 5s margin

    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank

    # Build the 2x3x3 block
    U_b, V_b, W_b = _build_2x3x3_block_strassen()
    n_b, m_b, p_b = 2, 3, 3
    rank_block = 17

    # Verify block
    err = _verify_block(U_b, V_b, W_b, n_b, m_b, p_b)
    if err > 1e-10:
        # Block construction failed, fall back to schoolbook
        return best_result

    # Try to find a lower-rank block decomposition
    if time.time() < deadline - 30:
        for r_try in [16, 15, 14]:
            if time.time() > deadline - 25:
                break
            result = _try_rank_r_block(n_b, m_b, p_b, r_try, time_budget=min(8.0, deadline - time.time() - 20))
            if result is not None:
                U_b, V_b, W_b = result
                rank_block = r_try
                break

    # Construct full decomposition by tiling the block 3 times (for the 3 k-blocks of size 3)
    full_rank = rank_block * 3
    U_full = np.zeros((dim_u, full_rank))
    V_full = np.zeros((dim_v, full_rank))
    W_full = np.zeros((dim_w, full_rank))

    for block_idx in range(3):
        k_offset = block_idx * p_b
        for c in range(rank_block):
            full_c = block_idx * rank_block + c
            U_full[:, full_c] = U_b[:, c]

            for j in range(m_b):
                for k in range(p_b):
                    v_block_idx = j * p_b + k
                    v_full_idx = j * p + (k_offset + k)
                    V_full[v_full_idx, full_c] = V_b[v_block_idx, c]

            for k in range(p_b):
                for i in range(n_b):
                    w_block_idx = k * n_b + i
                    w_full_idx = (k_offset + k) * n + i
                    W_full[w_full_idx, full_c] = W_b[w_block_idx, c]

    # Verify full decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.dot(U_full[a, :], V_full[b, :] * W_full[c, :])
                if abs(val - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    # Also check that non-target entries are zero
    if valid:
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and k == k2 and i == i2):
                        val = np.dot(U_full[a, :], V_full[b, :] * W_full[c, :])
                        if abs(val) > 1e-10:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break

    if valid:
        best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        best_rank = full_rank

    return best_result
# EVOLVE-BLOCK-END
