import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct using random sampling."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for _ in range(100):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        if abs(val - 1.0) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS  # 5, 5, 5

    terms = []

    def add_term(u_dict, v_dict, w_dict):
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))

    # 1. Rank-49 algorithm for the 4x4 top-left block (indices 0..15)
    # This is the Kronecker product of two Strassen 2x2 algorithms.
    
    # Strassen 2x2 terms (local indices 0..3)
    # u: A indices, v: B indices, w: C indices
    # A is 2x2, B is 2x2, C is 2x2
    # Index mapping: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # Helper to map local 2x2 index to global 4x4 index given block position
    def local_to_global_4x4(local_idx, block_r, block_c):
        r_l = local_idx // 2
        c_l = local_idx % 2
        global_r = 2 * block_r + r_l
        global_c = 2 * block_c + c_l
        return global_r * 4 + global_c

    # Generate 49 terms
    for i, (u_outer, v_outer, w_outer) in enumerate(strassen_2x2):
        for j, (u_inner, v_inner, w_inner) in enumerate(strassen_2x2):
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # Combine coefficients
            for u_block_idx, u_val in u_outer.items():
                for u_local_idx, u_inner_val in u_inner.items():
                    global_idx = local_to_global_4x4(u_local_idx, u_block_idx // 2, u_block_idx % 2)
                    u_dict[global_idx] = u_dict.get(global_idx, 0) + u_val * u_inner_val
            
            for v_block_idx, v_val in v_outer.items():
                for v_local_idx, v_inner_val in v_inner.items():
                    global_idx = local_to_global_4x4(v_local_idx, v_block_idx // 2, v_block_idx % 2)
                    v_dict[global_idx] = v_dict.get(global_idx, 0) + v_val * v_inner_val
            
            for w_block_idx, w_val in w_outer.items():
                for w_local_idx, w_inner_val in w_inner.items():
                    global_idx = local_to_global_4x4(w_local_idx, w_block_idx // 2, w_block_idx % 2)
                    w_dict[global_idx] = w_dict.get(global_idx, 0) + w_val * w_inner_val
            
            add_term(u_dict, v_dict, w_dict)

    # 2. Schoolbook terms for the remaining parts of 5x5
    # Partition: 4x4 block (0..3), and 1x1, 1x4, 4x1 strips.
    # Indices:
    # A: 0..24 (5x5)
    # B: 0..24
    # C: 0..24
    
    # C_11 (4x4) = A_11 B_11 + A_12 B_21
    # C_12 (4x1) = A_11 B_12 + A_12 B_22
    # C_21 (1x4) = A_21 B_11 + A_22 B_21
    # C_22 (1x1) = A_21 B_12 + A_22 B_22
    
    # A_11: rows 0-3, cols 0-3. Indices 0..15.
    # A_12: rows 0-3, col 4. Indices 4, 9, 14, 19.
    # A_21: row 4, cols 0-3. Indices 20, 21, 22, 23.
    # A_22: row 4, col 4. Index 24.
    
    # B_11: rows 0-3, cols 0-3. Indices 0..15.
    # B_12: rows 0-3, col 4. Indices 4, 9, 14, 19.
    # B_21: row 4, cols 0-3. Indices 20, 21, 22, 23.
    # B_22: row 4, col 4. Index 24.
    
    # C_11: rows 0-3, cols 0-3. Indices 0..15. (Handled by rank-49)
    # C_12: rows 0-3, col 4. Indices 4, 9, 14, 19.
    # C_21: row 4, cols 0-3. Indices 20, 21, 22, 23.
    # C_22: row 4, col 4. Index 24.
    
    # Term: A_12 B_21 -> C_11
    # A_12 is 4x1, B_21 is 1x4.
    # C_11[i,j] += A_12[i,0] * B_21[0,j]
    for i in range(4):
        for j in range(4):
            a_idx = i * 5 + 4
            b_idx = 4 * 5 + j
            c_idx = i * 5 + j
            add_term({a_idx: 1}, {b_idx: 1}, {c_idx: 1})

    # Term: A_11 B_12 -> C_12
    # A_11 is 4x4, B_12 is 4x1.
    # C_12[i,0] += sum_k A_11[i,k] * B_12[k,0]
    for i in range(4):
        for k in range(4):
            a_idx = i * 5 + k
            b_idx = k * 5 + 4
            c_idx = i * 5 + 4
            add_term({a_idx: 1}, {b_idx: 1}, {c_idx: 1})

    # Term: A_12 B_22 -> C_12
    # A_12 is 4x1, B_22 is 1x1.
    # C_12[i,0] += A_12[i,0] * B_22[0,0]
    for i in range(4):
        a_idx = i * 5 + 4
        b_idx = 4 * 5 + 4
        c_idx = i * 5 + 4
        add_term({a_idx: 1}, {b_idx: 1}, {c_idx: 1})

    # Term: A_21 B_11 -> C_21
    # A_21 is 1x4, B_11 is 4x4.
    # C_21[0,j] += sum_k A_21[0,k] * B_11[k,j]
    for j in range(4):
        for k in range(4):
            a_idx = 4 * 5 + k
            b_idx = k * 5 + j
            c_idx = 4 * 5 + j
            add_term({a_idx: 1}, {b_idx: 1}, {c_idx: 1})

    # Term: A_22 B_21 -> C_21
    # A_22 is 1x1, B_21 is 1x4.
    # C_21[0,j] += A_22[0,0] * B_21[0,j]
    for j in range(4):
        a_idx = 4 * 5 + 4
        b_idx = 4 * 5 + j
        c_idx = 4 * 5 + j
        add_term({a_idx: 1}, {b_idx: 1}, {c_idx: 1})

    # Term: A_21 B_12 -> C_22
    # A_21 is 1x4, B_12 is 4x1.
    # C_22[0,0] += sum_k A_21[0,k] * B_12[k,0]
    for k in range(4):
        a_idx = 4 * 5 + k
        b_idx = k * 5 + 4
        c_idx = 4 * 5 + 4
        add_term({a_idx: 1}, {b_idx: 1}, {c_idx: 1})

    # Term: A_22 B_22 -> C_22
    # A_22 is 1x1, B_22 is 1x1.
    # C_22[0,0] += A_22[0,0] * B_22[0,0]
    add_term({24: 1}, {24: 1}, {24: 1})

    rank = len(terms)

    # Build factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to naive
        rank_naive = n * m * p
        U2 = [[0] * rank_naive for _ in range(n * m)]
        V2 = [[0] * rank_naive for _ in range(m * p)]
        W2 = [[0] * rank_naive for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j][col] = 1
                    V2[j*p+k][col] = 1
                    W2[k*n+i][col] = 1
                    col += 1
        return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
