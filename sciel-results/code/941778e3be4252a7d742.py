"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_full(U, V, W, n, m, p, tol=1e-10):
    """Verify full decomposition is exact."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    r = U.shape[1]
    
    # Check all target entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check all non-target entries are 0
    # This is too slow for full check, do sampled check
    # Sampled check of zero entries
    rng = np.random.RandomState(123)
    for _ in range(500):
        a = rng.randint(dim_u)
        b = rng.randint(dim_v)
        c = rng.randint(dim_w)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if not (j == j2 and k == k2 and i == i2):
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val) > tol:
                return False
    return True

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_tiled_block(n, m, p, block_p=3):
    """Build decomposition by tiling a 2x3x3 block."""
    n_b, m_b, p_b = 2, 3, 3
    rank_block = 17
    U_b = np.zeros((n_b * m_b, rank_block))
    V_b = np.zeros((m_b * p_b, rank_block))
    W_b = np.zeros((p_b * n_b, rank_block))
    col = 0

    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals.items():
            U_b[idx, col] += val
        for idx, val in v_vals.items():
            V_b[idx, col] += val
        for idx, val in w_vals.items():
            W_b[idx, col] += val
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

    # Tile the block 3 times
    full_rank = rank_block * (p // p_b)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    U_full = np.zeros((dim_u, full_rank))
    V_full = np.zeros((dim_v, full_rank))
    W_full = np.zeros((dim_w, full_rank))

    for block_idx in range(p // p_b):
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

    return U_full, V_full, W_full, full_rank

def _try_find_decomposition(n, m, p, rank, time_budget=5.0, n_trials=30):
    """Try to find a rank-r decomposition via random init + linear solve."""
    start = time.time()
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    for trial in range(n_trials):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.RandomState(trial * 7 + 42)
        U = rng.randn(dim_u, rank)
        V = rng.randn(dim_v, rank)
        
        # Solve for W: for each c, W[c,:] must satisfy
        # sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c] for all (a,b) with T[a,b,c]=1
        # For fixed c, the constraints are: for all (i,j,k) with k*n+i=c,
        # i.e., for all j: sum_d U[i*m+j,d] * V[j*p+k,d] * W[c,d] = 1
        # This is m_b constraints on r unknowns W[c,0..r-1]
        
        W = np.zeros((dim_w, rank))
        feasible = True
        
        for c in range(dim_w):
            k = c // n
            i = c % n
            # Constraints: for all j in 0..m-1
            A_mat = np.zeros((m, rank))
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
        
        # Verify full decomposition
        if _verify_full(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18

    start_time = time.time()
    deadline = start_time + 140  # leave 10s margin

    # Schoolbook decomposition as fallback
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n * m * p  # 54

    # Tiled block decomposition (rank 51)
    U_tiled, V_tiled, W_tiled, rank_tiled = _build_tiled_block(n, m, p)
    if _verify_full(U_tiled, V_tiled, W_tiled, n, m, p, tol=1e-10):
        best_result = {"U": U_tiled.tolist(), "V": V_tiled.tolist(), "W": W_tiled.tolist()}
        best_rank = rank_tiled

    # Try to find lower rank decompositions
    # Try ranks from best_rank-1 down to some lower bound
    for r_try in range(best_rank - 1, max(best_rank - 15, 0), -1):
        if time.time() > deadline - 20:
            break
        
        time_budget = min(8.0, deadline - time.time() - 15)
        if time_budget < 2.0:
            break
        
        result = _try_find_decomposition(n, m, p, r_try, time_budget=time_budget, n_trials=40)
        if result is not None:
            U_found, V_found, W_found = result
            best_result = {"U": U_found.tolist(), "V": V_found.tolist(), "W": W_found.tolist()}
            best_rank = r_try
            break  # Found the best we can, stop

    return best_result
# EVOLVE-BLOCK-END
