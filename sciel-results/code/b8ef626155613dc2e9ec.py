"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 2x4 * 4x5 matrix multiplication
    with global linear dependency elimination to reduce rank.
    """
    import time
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    rank_terms = []
    
    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))
    
    def add_schoolbook_2x2x1(i0, j0, k0, sign=1):
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))
    
    add_strassen_2x2x2(i0=0, j0=0, k0=0, sign=1)
    add_strassen_2x2x2(i0=0, j0=0, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=0, k0=4, sign=1)
    add_strassen_2x2x2(i0=0, j0=2, k0=0, sign=1)
    add_strassen_2x2x2(i0=0, j0=2, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=2, k0=4, sign=1)
    
    import numpy as np
    
    # Build contribution matrix M: M[t, c] = contribution of term c to tensor entry t
    # where t = i*m*p + j*p + k indexes the (i,j,k) triple
    triples_idx = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    ntriples = len(triples_idx)
    
    M = np.zeros((ntriples, len(rank_terms)), dtype=float)
    for c, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for t, (i, j, k) in enumerate(triples_idx):
            u_val = u_dict.get((i, j), 0)
            v_val = v_dict.get((j, k), 0)
            w_val = w_dict.get((k, i), 0)
            M[t, c] = u_val * v_val * w_val
    
    # Target: all-ones tensor
    ones = np.ones(ntriples)
    
    # Verify full decomposition sums to ones
    col_sum = M.sum(axis=1)
    if not np.allclose(col_sum, ones, atol=1e-10):
        # Fallback: return full 36-term
        rank = len(rank_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][c] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][c] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][c] = coeff
        return {"U": U, "V": V, "W": W}
    
    # Search for subsets of terms whose sum equals ones
    # i.e., find terms to remove such that remaining sum to ones
    # Equivalently: find a subset S of terms with sum(S) = 0 (as tensor)
    
    best_indices = list(range(len(rank_terms)))
    best_rank = len(rank_terms)
    
    # Try removing 1 term
    for remove in range(len(rank_terms)):
        if time.time() > deadline - 10:
            break
        residual = ones - (col_sum - M[:, remove])
        if np.max(np.abs(residual)) < 1e-10:
            best_rank = best_rank - 1
            best_indices = [i for i in range(len(rank_terms)) if i != remove]
            break
    
    # Try removing 2 terms
    if best_rank == len(rank_terms) and time.time() < deadline - 10:
        for r1 in range(len(rank_terms)):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, len(rank_terms)):
                residual = ones - (col_sum - M[:, r1] - M[:, r2])
                if np.max(np.abs(residual)) < 1e-10:
                    best_rank = best_rank - 2
                    best_indices = [i for i in range(len(rank_terms)) if i != r1 and i != r2]
                    break
            if best_rank < len(rank_terms):
                break
    
    # Try removing 3 terms
    if best_rank == len(rank_terms) and time.time() < deadline - 10:
        for r1 in range(len(rank_terms)):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, len(rank_terms)):
                if time.time() > deadline - 10:
                    break
                for r3 in range(r2 + 1, len(rank_terms)):
                    residual = ones - (col_sum - M[:, r1] - M[:, r2] - M[:, r3])
                    if np.max(np.abs(residual)) < 1e-10:
                        best_rank = best_rank - 3
                        best_indices = [i for i in range(len(rank_terms)) if i not in (r1, r2, r3)]
                        break
                if best_rank < len(rank_terms):
                    break
            if best_rank < len(rank_terms):
                break
    
    # Build final U, V, W
    rank = len(best_indices)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for new_c, old_c in enumerate(best_indices):
        u_dict, v_dict, w_dict = rank_terms[old_c]
        for (i, j), coeff in u_dict.items():
            U[i * m + j][new_c] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][new_c] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][new_c] = coeff
    
    # Verify with random test
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
            C_computed[i, k] = val
    
    error = np.max(np.abs(C_computed - C_expected))
    if error > 1e-10:
        # Fallback to full decomposition
        U = [[0] * 36 for _ in range(n * m)]
        V = [[0] * 36 for _ in range(m * p)]
        W = [[0] * 36 for _ in range(p * n)]
        for c, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][c] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][c] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
