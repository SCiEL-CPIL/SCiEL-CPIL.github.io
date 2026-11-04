"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify a full decomposition is exact."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    err = 0.0
    for a in range(dim_u):
        for b in range(dim_v):
            for c in range(dim_w):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                expected = 1.0 if (j == j2 and k == k2 and i == i2) else 0.0
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err += (val - expected) ** 2
    return err

def _build_polynomial_ring_decomposition():
    """
    Construct a rank-18 decomposition using the polynomial ring Z[x]/(x^3-1).
    
    Key insight: Group B's columns into 3 groups of 3.
    B = [B_0, B_1, B_2] where B_g is 3x3, with B_g[j,k] = B[j, 3*g+k]
    
    We can write A as a 2x3 matrix and compute C = A * B by treating
    the column groups as polynomial coefficients.
    
    The product C[i,k] = sum_j A[i,j] * B[j,k]
    
    We decompose this using the fact that multiplication in Z[x]/(x^3-1)
    has a special structure that can be exploited.
    
    Strategy: Use 18 rank-1 terms, one for each (i, k) pair, but with
    shared structure across the 3 column groups.
    """
    n, m, p = 2, 3, 9
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each output column k (0..8) and row i (0..1), we need:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We'll use a decomposition that shares computation across the 3 groups.
    # Group g = k // 3, position within group = k % 3
    #
    # The key is to use the cyclic structure: x^3 = 1 in Z[x]/(x^3-1)
    
    # We'll construct 18 terms. For each (i, k) pair, we create a term
    # that contributes to C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Actually, let's use a different approach:
    # We can write the 2x3 x 3x9 multiplication as a sum of 18 rank-1 tensors
    # by exploiting the structure.
    
    # Term t for t = 0..17:
    # Let t = i * 9 + k, where i in {0,1}, k in {0..8}
    # But we need to share structure.
    
    # Better approach: Use the fact that we can decompose the 3x9 matrix B
    # into 3 blocks of 3x3, and use a decomposition that works on each block
    # with shared U factors.
    
    # Let's try: For each column group g (0,1,2) and each "mode" r (0,1,2),
    # we create 2 terms (one for each row i), giving 3*2*2 = 12 terms.
    # But we need 18, so let's use a different grouping.
    
    # Actually, the simplest rank-18 decomposition is just the schoolbook
    # with 18 terms (one per output element). But that's rank 54.
    
    # Let me think differently. The tensor T[i,j,k] = 1 if (i,j,k) is a valid
    # multiplication index, else 0. We want rank-1 terms u_i * v_j * w_k.
    
    # A rank-1 term (u, v, w) contributes to T[i,j,k] if u[i] * v[j] * w[k] != 0.
    # For the multiplication tensor, T[i,j,k] = 1 iff the indices are consistent.
    
    # The structure: A is 2x3, B is 3x9, C is 2x9.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can use 18 terms by noting that for each (i,k) pair, we need
    # sum_j A[i,j] * B[j,k]. If we could make each term compute this sum
    # for one (i,k) pair, we'd need 18 terms. But each term is rank-1,
    # so it can only compute one product A[i,j] * B[j,k] for fixed (i,j,k).
    
    # So rank-18 is not achievable with simple rank-1 terms for this problem.
    # The minimum rank is at least 2*3*9 / something.
    
    # Let me use the block approach: decompose into 3 blocks of 2x3 x 3x3,
    # each with rank 6 (schoolbook for 2x3x3), giving total rank 18.
    
    # For each block g (0,1,2), the subproblem is:
    # C_g[i, k'] = sum_j A[i,j] * B_g[j, k'] where k' in {0,1,2}, k = 3*g + k'
    
    # A 2x3 x 3x3 multiplication has rank at most 2*3*3 = 18 (schoolbook),
    # but can be done with rank 17 (Strassen-like) or lower.
    
    # If each block has rank 6, total rank is 18.
    # Can we do 2x3 x 3x3 with rank 6? That would be very efficient.
    
    # Actually, 2x3 x 3x3: the tensor has 2*3*3 = 18 nonzero entries.
    # The minimum rank is at least the max of the dimensions in some sense.
    
    # Let's just use rank-6 for each 2x3x3 block (if possible) or fall back.
    
    # For now, let's construct a valid rank-18 decomposition using 3 blocks
    # of rank 6 each.
    
    # Block g: A (2x3) * B_g (3x3) -> C_g (2x3)
    # Rank-6 decomposition for 2x3 x 3x3:
    # We need 6 rank-1 terms.
    # Term r for r=0..5:
    #   Let r = i * 3 + k' where i in {0,1}, k' in {0,1,2}
    #   U[i, j] = 1 for all j (or some pattern)
    #   V[j, k'] = 1 for all j
    #   W[k', i] = 1
    # But this gives C_g[i,k'] = sum_j U[i,j] * V[j,k'] * W[k',i]
    # If U[i,j]=1, V[j,k']=1, W[k',i]=1, then C_g[i,k'] = sum_j 1*1*1 = 3
    # That's wrong.
    
    # Correct rank-6 for 2x3x3:
    # For each (i, k') pair, use a term that computes sum_j A[i,j]*B_g[j,k']
    # But a rank-1 term can only compute A[i,j]*B_g[j,k'] for fixed (i,j,k').
    
    # So we need 2*3*3 = 18 terms for schoolbook. Rank 6 is not possible
    # with this simple approach.
    
    # Let me just return the schoolbook rank-54 decomposition for now,
    # and use the block Strassen approach for a better result.
    
    return None

def _build_block_strassen(n_b, m_b, p_b):
    """Construct a decomposition for n_b x m_b x m_b x p_b multiplication."""
    if n_b == 2 and m_b == 3 and p_b == 3:
        return _build_2x3x3_block_strassen()
    # Fallback to schoolbook
    rank = n_b * m_b * p_b
    U = np.zeros((n_b * m_b, rank))
    V = np.zeros((m_b * p_b, rank))
    W = np.zeros((p_b * n_b, rank))
    col = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[i*m_b+j, col] = 1
                V[j*p_b+k, col] = 1
                W[k*n_b+i, col] = 1
                col += 1
    return U, V, W

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
    add_term({0:1, 3:1}, {0:1, 4:1}, {0:1, 3:1})
    add_term({3:1, 4:1}, {0:1}, {1:1, 3:-1})
    add_term({0:1}, {1:1, 4:-1}, {2:1, 3:1})
    add_term({4:1}, {3:1, 0:-1}, {0:1, 1:1})
    add_term({0:1, 1:1}, {4:1}, {0:-1, 2:1})
    add_term({3:1, 0:-1}, {0:1, 1:1}, {3:1})
    add_term({1:1, 4:-1}, {3:1, 4:1}, {0:1})

    # Schoolbook: j in {0,1}, k=2
    add_term({0:1}, {2:1}, {4:1})
    add_term({1:1}, {5:1}, {4:1})
    add_term({3:1}, {2:1}, {5:1})
    add_term({4:1}, {5:1}, {5:1})

    # Schoolbook: j=2, k in {0,1,2}
    add_term({2:1}, {6:1}, {0:1})
    add_term({2:1}, {7:1}, {2:1})
    add_term({2:1}, {8:1}, {4:1})
    add_term({5:1}, {6:1}, {1:1})
    add_term({5:1}, {7:1}, {3:1})
    add_term({5:1}, {8:1}, {5:1})

    return U, V, W

def _try_rank_r_full(n, m, p, r, time_budget=5.0):
    """Try to find a rank-r decomposition for the full problem."""
    start = time.time()
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    for attempt in range(30):
        if time.time() - start > time_budget:
            break
        U = np.random.randn(dim_u, r)
        V = np.random.randn(dim_v, r)
        
        W = np.zeros((dim_w, r))
        feasible = True
        for c in range(dim_w):
            k = c // n
            i = c % n
            A_mat = np.zeros((m, r))
            for j in range(m):
                a_idx = i * m + j
                b_idx = j * p + k
                A_mat[j, :] = U[a_idx, :] * V[b_idx, :]
            b_vec = np.ones(m)
            try:
                sol, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                feasible = False
                break
            if np.linalg.norm(A_mat @ sol - b_vec) > 1e-8:
                feasible = False
                break
            W[c, :] = sol
        
        if not feasible:
            continue
        
        err = _verify_decomposition(U, V, W, n, m, p)
        if err < 1e-8:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18

    start_time = time.time()
    deadline = start_time + 145

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
    err = _verify_decomposition(U_b, V_b, W_b, n_b, m_b, p_b)
    if err > 1e-10:
        return best_result

    # Try to find a lower-rank block decomposition
    if time.time() < deadline - 30:
        for r_try in [16, 15, 14]:
            if time.time() > deadline - 25:
                break
            result = _try_rank_r_full(n_b, m_b, p_b, r_try, time_budget=min(8.0, deadline - time.time() - 20))
            if result is not None:
                U_b, V_b, W_b = result
                rank_block = r_try
                break

    # Construct full decomposition by tiling the block 3 times
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
    err_full = _verify_decomposition(U_full, V_full, W_full, n, m, p)
    if err_full < 1e-8:
        best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        best_rank = full_rank

    return best_result
# EVOLVE-BLOCK-END
